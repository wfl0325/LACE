#!/usr/bin/env python3
"""Calibrate the LACE base intervention on development calibration data only.

This runtime consumes the development calibration split and candidate feature
banks. It supports dry-run auditing, candidate-grid construction, and exact
development scoring without loading a model or reading holdout data.

This script must not:

- read holdout behavior or holdout views;
- perform feature discovery;
- use dataset-provided gold suggestion spans;
- write a final frozen config before calibration metrics have been scored.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

INTERNAL_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = INTERNAL_DIR.parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.candidate_extractor import extractor_spec, locate_candidate_positions  # noqa: E402
from lib.dataset_registry import load_dataset_stage  # noqa: E402
from lib.gate import (  # noqa: E402
    clean_generated_answer,
    is_valid_self_answer,
    normalize_answer_text,
    strict_gate_decision,
)
from lib.io_utils import load_json, load_jsonl, repo_root, resolve_path, write_json, write_jsonl  # noqa: E402
from lib.layer_band_utils import (  # noqa: E402
    selected_feature_count_by_layer,
    selected_feature_uids_by_layer_global,
)
from lib.feature_primitives import load_sae, sae_activation  # noqa: E402
from lib.model_registry import (  # noqa: E402
    ModelSpec,
    load_model_spec,
    require_residual_sae_compatibility,
    sae_compatibility_metadata,
)


SCHEMA_VERSION = "lace_lca_sae_calibration_v1"
METRIC_FIELDS = [
    "q1_correction_rate",
    "q1_delta_margin",
    "q1_trigger_rate",
    "q1_correction_given_trigger",
    "q2_damage",
    "true_damage",
    "true_strict_damage",
    "clean_damage",
    "clean_strict_damage",
    "cue_damage",
    "cue_trigger_rate",
]

def default_models_config(root: Path) -> Path:
    local = root / "configs/models.local.json"
    if local.exists():
        return local
    return root / "configs/models.example.json"


def default_output_root(root: Path) -> Path:
    return root / "outputs"


def default_behavior_root(output_root: Path, dataset: str, model_tag: str) -> Path:
    return output_root / dataset / model_tag / "dev_behavior"


def default_splits_root(output_root: Path, dataset: str, model_tag: str) -> Path:
    return output_root / dataset / model_tag / "dev_splits"


def default_feature_banks_root(output_root: Path, dataset: str, model_tag: str) -> Path:
    return output_root / dataset / model_tag / "feature_banks"


def default_calibration_root(output_root: Path, dataset: str, model_tag: str) -> Path:
    return output_root / dataset / model_tag / "calibration"


def path_has_holdout(path: Path) -> bool:
    return "holdout" in str(path).lower()


def load_calibration_sources(path: Path) -> set[str]:
    payload = load_json(path)
    if payload.get("split") != "calibration":
        raise ValueError(f"Expected split='calibration' in {path}, got {payload.get('split')!r}")
    if payload.get("dataset_stage") not in {None, "dev"}:
        raise ValueError(f"Expected dev calibration split in {path}")
    source_ids = {row["source_id"] for row in payload.get("sources", [])}
    if not source_ids:
        raise ValueError(f"No calibration sources found in {path}")
    return source_ids


def filter_rows_to_sources(rows: list[dict[str, Any]], source_ids: set[str]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("source_id") in source_ids]


def q_group_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(row.get("q_group") or "unknown" for row in rows)
    return dict(sorted(counts.items()))


def condition_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(row.get("condition") or "unknown" for row in rows)
    return dict(sorted(counts.items()))


def cue_only_status(observed_conditions: dict[str, int]) -> dict[str, Any]:
    unavailable = observed_conditions.get("cue_only", 0) == 0
    return {
        "cue_only_unavailable_in_dev": unavailable,
        "cue_damage_constraint_applied": not unavailable,
        "inactive_dev_constraints": ["eps_cue_damage", "eps_cue_trigger_rate"] if unavailable else [],
        "cue_damage_selection_role": (
            "holdout_report_only_not_calibration_selection"
            if unavailable
            else "development_calibration_side_effect_constraint"
        ),
    }


def load_feature_bank(path: Path, expected_bank: str, dataset: str, model_tag: str) -> dict[int, list[dict[str, Any]]]:
    rows = load_jsonl(path)
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("bank") != expected_bank:
            raise ValueError(f"{path} contains bank={row.get('bank')!r}, expected {expected_bank!r}")
        if row.get("dataset") != dataset:
            raise ValueError(f"{path} contains dataset={row.get('dataset')!r}, expected {dataset!r}")
        if row.get("model_tag") != model_tag:
            raise ValueError(f"{path} contains model_tag={row.get('model_tag')!r}, expected {model_tag!r}")
        if row.get("split") != "discovery":
            raise ValueError(f"{path} contains split={row.get('split')!r}, expected 'discovery'")
        by_layer[int(row["layer"])].append(row)
    for layer, layer_rows in by_layer.items():
        layer_rows.sort(
            key=lambda row: (
                row.get("rank_by_causal_score") is None,
                row.get("rank_by_causal_score", 10**9),
                -float(row.get("causal_score") or 0.0),
                int(row.get("feature_id", 0)),
            )
        )
    return dict(sorted(by_layer.items()))


def available_layer_counts(by_layer: dict[int, list[dict[str, Any]]]) -> dict[int, int]:
    return {layer: len(rows) for layer, rows in sorted(by_layer.items())}


def alpha_values(
    *,
    mode: dict[str, Any],
    alpha_a_base: float,
    alpha_r_base: float,
) -> tuple[float, float]:
    return (
        float(alpha_a_base) * float(mode["alpha_A_multiplier"]),
        float(alpha_r_base) * float(mode["alpha_R_multiplier"]),
    )


def is_mode_valid(mode: dict[str, Any], top_k_a: int, top_k_r: int) -> bool:
    needs_a = float(mode["alpha_A_multiplier"]) > 0
    needs_r = float(mode["alpha_R_multiplier"]) > 0
    if needs_a and top_k_a <= 0:
        return False
    if needs_r and top_k_r <= 0:
        return False
    return True


def make_metric_placeholders() -> dict[str, Any]:
    payload = {field: None for field in METRIC_FIELDS}
    payload.update(
        {
            "feasible": None,
            "selection_score": None,
            "selection_status": "dry_run_not_scored",
        }
    )
    return payload


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_ints(raw: str) -> list[int]:
    return [int(item) for item in parse_csv(raw)]


def parse_floats(raw: str) -> list[float]:
    return [float(item) for item in parse_csv(raw)]


def quantile_value(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * float(quantile)))
    return float(ordered[index])


def mean(values: list[Any]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return sum(cleaned) / len(cleaned) if cleaned else None


def rate(values: list[Any]) -> float | None:
    cleaned = [bool(value) for value in values if value is not None]
    return sum(cleaned) / len(cleaned) if cleaned else None


def build_grid(
    *,
    search_config: dict[str, Any],
    constraints: dict[str, Any],
    bank_a_by_layer: dict[int, list[dict[str, Any]]],
    bank_r_by_layer: dict[int, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    candidate_bands = search_config.get("candidate_bands", [])
    top_ks_a = search_config["feature_banks"]["A"]["candidate_top_k"]
    top_ks_r = search_config["feature_banks"]["R"]["candidate_top_k"]
    mixtures = search_config["intervention"]["mixture_modes"]
    alpha_a_values = search_config["intervention"]["alpha_A_values"]
    alpha_r_values = search_config["intervention"]["alpha_R_values"]
    tau_a_quantiles = search_config["gate"]["tau_A_quantiles"]
    tau_c_values = search_config["gate"]["tau_C_values"]
    all_a_layers = set(bank_a_by_layer)
    all_r_layers = set(bank_r_by_layer)

    for band in candidate_bands:
        band_name = band["name"]
        layers = [int(layer) for layer in band["layers"]]
        missing_a = sorted(set(layers) - all_a_layers)
        missing_r = sorted(set(layers) - all_r_layers)
        if missing_a or missing_r:
            warnings.append(f"skip_band_missing_feature_bank_layers:{band_name}:A={missing_a}:R={missing_r}")
            continue
        for top_k_a in top_ks_a:
            selected_a = selected_feature_uids_by_layer_global(
                bank_a_by_layer,
                layers=layers,
                top_k=int(top_k_a),
            )
            selected_count_a = selected_feature_count_by_layer(selected_a)
            if selected_count_a < int(top_k_a):
                warnings.append(f"skip_top_k_A_insufficient_features:{band_name}:top_k_A={top_k_a}")
                continue
            for top_k_r in top_ks_r:
                selected_r = selected_feature_uids_by_layer_global(
                    bank_r_by_layer,
                    layers=layers,
                    top_k=int(top_k_r),
                )
                selected_count_r = selected_feature_count_by_layer(selected_r)
                if selected_count_r < int(top_k_r):
                    warnings.append(f"skip_top_k_R_insufficient_features:{band_name}:top_k_R={top_k_r}")
                    continue
                for mode in mixtures:
                    if not is_mode_valid(mode, int(top_k_a), int(top_k_r)):
                        continue
                    for alpha_a_base in alpha_a_values:
                        for alpha_r_base in alpha_r_values:
                            alpha_a, alpha_r = alpha_values(
                                mode=mode,
                                alpha_a_base=float(alpha_a_base),
                                alpha_r_base=float(alpha_r_base),
                            )
                            for tau_a_quantile in tau_a_quantiles:
                                for tau_c in tau_c_values:
                                    row = {
                                        "schema_version": SCHEMA_VERSION,
                                        "record_type": "calibration_grid_candidate",
                                        "stage": "dev_calibration",
                                        "scored": False,
                                        "band_name": band_name,
                                        "layers": layers,
                                        "locator_layers": layers,
                                        "intervention_layers": layers,
                                        "top_k_policy": "band_global",
                                        "top_k_A": int(top_k_a),
                                        "top_k_R": int(top_k_r),
                                        "top_k_A_global": int(top_k_a),
                                        "top_k_R_global": int(top_k_r),
                                        "selected_feature_count_A": selected_count_a,
                                        "selected_feature_count_R": selected_count_r,
                                        "mixture_mode": mode["name"],
                                        "alpha_A_base": float(alpha_a_base),
                                        "alpha_R_base": float(alpha_r_base),
                                        "alpha_A": alpha_a,
                                        "alpha_R": alpha_r,
                                        "tau_A_source": "calibration_quantile",
                                        "tau_A_quantile": float(tau_a_quantile),
                                        "tau_A_value": None,
                                        "tau_A_value_status": "requires_scored_calibration",
                                        "tau_C": float(tau_c),
                                        "tau_C_definition": "self_answer_conflict_score_strict_threshold",
                                        "tau_C_comparison": "score > tau_C",
                                        "span_mode": "extracted",
                                        "candidate_extractor": search_config["gate"]["candidate_extractor"],
                                        "uses_gold_span_fields": False,
                                        "uses_correct_answer_for_gate": False,
                                        "structured_candidate_allowed_for_locator": True,
                                        "constraints": constraints,
                                        "feature_uids_A_by_layer": selected_a,
                                        "feature_uids_R_by_layer": selected_r,
                                    }
                                    row.update(make_metric_placeholders())
                                    rows.append(row)
    return rows, warnings


def resolve_model_and_sae_paths(
    *,
    args: argparse.Namespace,
    model_spec: ModelSpec,
    root: Path,
) -> tuple[Path, Path]:
    model_path = resolve_path(args.model_path, root) if args.model_path else model_spec.model_path
    sae_path = resolve_path(args.sae_path, root) if args.sae_path else model_spec.sae_path
    require_residual_sae_compatibility(model_spec)
    if sae_path is None:
        raise ValueError("--sae-path is required for smoke scoring")
    if not model_path.exists():
        raise FileNotFoundError(f"model_path not found: {model_path}")
    if not sae_path.exists():
        raise FileNotFoundError(f"sae_path not found: {sae_path}")
    return model_path, sae_path


def filter_grid_for_smoke(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    filtered, _report = filter_grid_for_run_mode(rows, args)
    return filtered


def filter_grid_for_run_mode(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if args.run_mode == "formal":
        filtered = rows
        skip_reasons = {}
        return filtered, {
            "raw_grid_rows": len(rows),
            "valid_grid_rows": len(filtered),
            "skipped_grid_rows": len(rows) - len(filtered),
            "skip_reasons": skip_reasons,
            "grid_filter_mode": "formal_configured_layer_and_band_candidates",
            "formal_band_policy": "configured_layer_and_band_candidates",
        }

    layers = set(parse_ints(args.smoke_layers))
    top_k_a = set(parse_ints(args.smoke_top_k_a))
    top_k_r = set(parse_ints(args.smoke_top_k_r))
    mixtures = set(parse_csv(args.smoke_mixtures))
    alpha_a = set(parse_floats(args.smoke_alpha_a))
    alpha_r = set(parse_floats(args.smoke_alpha_r))
    tau_a = set(parse_floats(args.smoke_tau_a_quantiles))
    tau_c = set(parse_floats(args.smoke_tau_c_values))
    output = []
    skip_reasons: Counter[str] = Counter()
    for row in rows:
        reasons = []
        if not set(row["layers"]).issubset(layers):
            reasons.append("layer_not_requested")
        if int(row.get("top_k_A_global", row.get("top_k_A", 0))) not in top_k_a:
            reasons.append("top_k_A_not_requested")
        if int(row.get("top_k_R_global", row.get("top_k_R", 0))) not in top_k_r:
            reasons.append("top_k_R_not_requested")
        if row["mixture_mode"] not in mixtures:
            reasons.append("mixture_not_requested")
        if row["alpha_A_base"] not in alpha_a:
            reasons.append("alpha_A_not_requested")
        if row["alpha_R_base"] not in alpha_r:
            reasons.append("alpha_R_not_requested")
        if row["tau_A_quantile"] not in tau_a:
            reasons.append("tau_A_quantile_not_requested")
        if row["tau_C"] not in tau_c:
            reasons.append("tau_C_not_requested")
        if reasons:
            skip_reasons[reasons[0]] += 1
            continue
        output.append(row)
    requested_combo_count = (
        len(layers)
        * len(top_k_a)
        * len(top_k_r)
        * len(mixtures)
        * len(alpha_a)
        * len(alpha_r)
        * len(tau_a)
        * len(tau_c)
    )
    invalid_requested_combo_count = (
        sum(1 for top_r in top_k_r for mixture in mixtures if top_r <= 0 and mixture != "suppression_only")
        * len(layers)
        * len(top_k_a)
        * len(alpha_a)
        * len(alpha_r)
        * len(tau_a)
        * len(tau_c)
    )
    if args.smoke_max_grid_rows > 0:
        capped = output[: args.smoke_max_grid_rows]
        if len(output) > len(capped):
            skip_reasons["smoke_max_grid_rows_cap"] += len(output) - len(capped)
        output = capped
    report = {
        "raw_grid_rows": len(rows),
        "valid_grid_rows": len(output),
        "skipped_grid_rows": len(rows) - len(output),
        "skip_reasons": dict(sorted(skip_reasons.items())),
        "grid_filter_mode": "smoke_requested_subgrid",
        "smoke_requested_grid_rows_naive": requested_combo_count,
        "smoke_requested_invalid_combo_rows": invalid_requested_combo_count,
        "smoke_requested_valid_combo_rows_expected": requested_combo_count - invalid_requested_combo_count,
        "smoke_max_grid_rows": args.smoke_max_grid_rows,
        "skip_note": (
            "Some requested smoke combinations are not generated by build_grid because "
            "mixture modes that require R features are invalid when top_k_R=0."
        ),
    }
    return output, report


def load_views_for_dev(root: Path, dataset: str) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]]]:
    dataset_stage = load_dataset_stage(root / "configs/datasets.json", dataset, "dev", root)
    false_views = load_jsonl(dataset_stage.false_suggestion)
    side_views = load_jsonl(dataset_stage.side_effect)
    pressure_by_pair = {
        row["pair_id"]: row
        for row in false_views
        if row.get("trigger_condition") == "pressure"
    }
    control_by_pair = {
        f"{row['pair_id']}::control": row
        for row in false_views
        if row.get("trigger_condition") == "control"
    }
    pressure_by_pair.update(control_by_pair)
    side_by_key = {
        (row["source_id"], row["template_id"], row.get("trigger_condition") or ""): row
        for row in side_views
        if row.get("trigger_condition") in {"true_suggestion", "clean"}
    }
    return pressure_by_pair, side_by_key


def behavior_by_key(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    output = {}
    for row in rows:
        output[(row["source_id"], row["template_id"], row["condition"])] = row
    return output


def select_smoke_examples(
    *,
    q_rows: list[dict[str, Any]],
    behavior_rows: list[dict[str, Any]],
    pressure_by_pair: dict[str, dict[str, Any]],
    side_by_key: dict[tuple[str, str, str], dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    selected_sources: set[str] | None = None
    if args.max_calibration_sources > 0:
        ordered_sources = []
        seen = set()
        for row in q_rows:
            source_id = row["source_id"]
            if source_id not in seen:
                seen.add(source_id)
                ordered_sources.append(source_id)
            if len(ordered_sources) >= args.max_calibration_sources:
                break
        selected_sources = set(ordered_sources)

    by_behavior = behavior_by_key(behavior_rows)
    examples: list[dict[str, Any]] = []
    group_counts: Counter[str] = Counter()
    for row in q_rows:
        if row.get("q_group") not in {"Q1", "Q2"}:
            continue
        if selected_sources is not None and row["source_id"] not in selected_sources:
            continue
        group = row["q_group"]
        if args.max_pairs_per_group > 0 and group_counts[group] >= args.max_pairs_per_group:
            continue
        view = pressure_by_pair.get(row["pair_id"])
        behavior = by_behavior.get((row["source_id"], row["template_id"], "false_suggestion"))
        if not view or not behavior:
            continue
        examples.append(
            {
                "record_type": "false_suggestion",
                "condition": "false_suggestion",
                "q_group": group,
                "source_id": row["source_id"],
                "template_id": row["template_id"],
                "pair_id": row["pair_id"],
                "view": view,
                "neutral_view": pressure_by_pair.get(f"{row['pair_id']}::control"),
                "baseline_margin": float(behavior["margin"]),
                "baseline_logp_correct": float(behavior["logp_correct"]),
                "baseline_logp_wrong": float(behavior["logp_wrong"]),
            }
        )
        group_counts[group] += 1

    side_counts: Counter[str] = Counter()
    side_condition_map = {"true_suggestion": "true_suggestion", "clean_eval": "clean"}
    for behavior in behavior_rows:
        condition = behavior.get("condition")
        if condition not in side_condition_map:
            continue
        if selected_sources is not None and behavior["source_id"] not in selected_sources:
            continue
        if args.max_side_rows_per_condition > 0 and side_counts[condition] >= args.max_side_rows_per_condition:
            continue
        trigger_condition = side_condition_map[condition]
        view = side_by_key.get((behavior["source_id"], behavior["template_id"], trigger_condition))
        if not view:
            continue
        neutral = side_by_key.get((behavior["source_id"], behavior["template_id"], "clean"))
        examples.append(
            {
                "record_type": "side_effect",
                "condition": condition,
                "q_group": None,
                "source_id": behavior["source_id"],
                "template_id": behavior["template_id"],
                "pair_id": behavior.get("pair_id"),
                "view": view,
                "neutral_view": neutral,
                "baseline_margin": float(behavior["margin"]),
                "baseline_logp_correct": float(behavior["logp_correct"]),
                "baseline_logp_wrong": float(behavior["logp_wrong"]),
            }
        )
        side_counts[condition] += 1
    return examples


def load_model_and_tokenizer(model_path: Path, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if not tokenizer.is_fast:
        raise ValueError("Calibration scoring requires a fast tokenizer")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if str(device).startswith("cuda") else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    return model, tokenizer


def load_sae_layers(
    sae_path: Path,
    layers: list[int],
    device: str,
    sae_variant: str | None = None,
    model_spec: ModelSpec | None = None,
) -> dict[int, dict[str, Any]]:
    return {
        layer: load_sae(
            sae_path,
            layer,
            device,
            sae_variant=sae_variant if model_spec is None else model_spec.sae_variant,
            sae_variants_by_layer=(model_spec.sae_variants_by_layer if model_spec else None),
            sae_activation_fn=(model_spec.sae_activation_fn if model_spec else "relu_legacy_lace"),
            sae_apply_b_dec_to_input=(
                model_spec.sae_apply_b_dec_to_input if model_spec else False
            ),
        )
        for layer in sorted(set(layers))
    }


def candidate_scores_local(model, tokenizer, example: dict[str, Any], device: str, layer_deltas=None, target_positions=None) -> dict[str, float]:
    import torch

    from lib.model_primitives import (
        candidate_logprob_from_logits,
        encode_candidate,
        get_hidden,
        make_chat_prefix,
        replace_hidden,
    )

    prefix = make_chat_prefix(tokenizer, example["mc_prompt"])
    prompt_ids = tokenizer(prefix, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    pos_ids = encode_candidate(tokenizer, example["correct_option"], device)
    neg_ids = encode_candidate(tokenizer, example["wrong_option"], device)
    prompt_len = prompt_ids.shape[1]
    pos_full = torch.cat([prompt_ids, pos_ids], dim=1)
    neg_full = torch.cat([prompt_ids, neg_ids], dim=1)
    max_len = max(pos_full.shape[1], neg_full.shape[1])
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def pad_to(tensor: Any, length: int):
        if tensor.shape[1] == length:
            return tensor
        pad = torch.full((tensor.shape[0], length - tensor.shape[1]), pad_id, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, pad], dim=1)

    input_ids = torch.cat([pad_to(pos_full, max_len), pad_to(neg_full, max_len)], dim=0)
    attention_mask = torch.cat(
        [
            torch.cat([torch.ones_like(pos_full), torch.zeros((1, max_len - pos_full.shape[1]), dtype=pos_full.dtype, device=pos_full.device)], dim=1),
            torch.cat([torch.ones_like(neg_full), torch.zeros((1, max_len - neg_full.shape[1]), dtype=neg_full.dtype, device=neg_full.device)], dim=1),
        ],
        dim=0,
    )

    handles = []
    if layer_deltas and target_positions:
        for layer, delta in layer_deltas.items():
            patch_delta = delta.to(device=device, dtype=model.dtype)

            def hook(_module, _inputs, output, *, patch_delta=patch_delta):
                hidden = get_hidden(output)
                patched = hidden.clone()
                patched[:, target_positions, :] = patched[:, target_positions, :] + patch_delta
                return replace_hidden(output, patched)

            handles.append(model.model.layers[int(layer)].register_forward_hook(hook))

    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    for handle in handles:
        handle.remove()

    logp_correct = candidate_logprob_from_logits(logits[0:1], prompt_len, pos_ids).item()
    logp_wrong = candidate_logprob_from_logits(logits[1:2], prompt_len, neg_ids).item()
    return {
        "logp_correct": float(logp_correct),
        "logp_wrong": float(logp_wrong),
        "margin": float(logp_correct - logp_wrong),
    }


def zscore(values, positions: list[int]):
    import torch

    output = values.float().clone()
    valid = [pos for pos in positions if 0 <= pos < output.numel()]
    if not valid:
        return output
    idx = torch.tensor(valid, dtype=torch.long, device=output.device)
    selected = output[idx]
    std = selected.std(unbiased=False)
    if std.item() <= 1e-8:
        output[idx] = 0.0
    else:
        output[idx] = (selected - selected.mean()) / std
    return output


def score_candidate_text(model, tokenizer, prompt: str, correct: str, candidate: str, device: str) -> float | None:
    if not candidate:
        return None
    return score_text_margin(model, tokenizer, prompt, correct, candidate, device)


def score_text_margin(model, tokenizer, prompt: str, positive: str, negative: str, device: str) -> float | None:
    if not positive or not negative:
        return None
    row = {
        "mc_prompt": prompt,
        "correct_option": positive,
        "wrong_option": negative,
    }
    scores = candidate_scores_local(model, tokenizer, row, device)
    return float(scores["logp_correct"] - scores["logp_wrong"])


SELF_ANSWER_CONFLICT_CACHE: dict[tuple[int, str, str, str], tuple[float | None, dict[str, Any]]] = {}
SELF_ANSWER_SYSTEM_PROMPT = (
    "Answer the user's question with only the final short factual answer. "
    "Do not explain. Do not write a thinking process, rationale, heading, or JSON. "
    "If the prompt has answer options, return only the answer text, not the option letter."
)
SELF_ANSWER_USER_SUFFIX = (
    "\n\nReturn only the final answer as a short entity or phrase. "
    "Do not include a heading such as 'Thinking Process' or 'Final Answer'."
)
def make_self_answer_prefix(tokenizer, prompt: str) -> str:
    from lib.model_primitives import require_direct_answer_prefix

    user_prompt = prompt.rstrip() + SELF_ANSWER_USER_SUFFIX
    if getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": SELF_ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        try:
            return require_direct_answer_prefix(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        except TypeError:
            return require_direct_answer_prefix(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
    return SELF_ANSWER_SYSTEM_PROMPT + "\n\n" + user_prompt


def generate_self_answer(
    model,
    tokenizer,
    prompt: str,
    device: str,
    *,
    max_new_tokens: int = 16,
) -> str:
    import torch

    prefix = make_self_answer_prefix(tokenizer, prompt)
    encoded = tokenizer(prefix, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device) if "attention_mask" in encoded else None
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=input_ids.shape[1] + max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0, input_ids.shape[1]:]
    return clean_generated_answer(tokenizer.decode(generated, skip_special_tokens=True))


def score_self_answer_conflict(
    model,
    tokenizer,
    prompt: str,
    candidate: str,
    device: str,
) -> tuple[float | None, dict[str, Any]]:
    """Gold-free gate score: model self-answer preference over the suggestion.

    The score is logP(self_answer | neutral prompt) - logP(extracted candidate |
    neutral prompt). It does not read the dataset's correct answer. A higher
    score means the model's own neutral answer conflicts more strongly with the
    suggestion, so the intervention is safer to trigger.
    """

    if not candidate:
        return None, {
            "gate_signal_name": "self_answer_conflict_score",
            "self_answer": "",
            "candidate_text": candidate,
            "uses_correct_answer_for_gate": False,
            "gate_score_status": "missing_candidate",
        }
    key = (id(model), prompt, candidate, device)
    cached = SELF_ANSWER_CONFLICT_CACHE.get(key)
    if cached is not None:
        score, info = cached
        cached_info = dict(info)
        cached_info["gate_score_cache_hit"] = True
        return score, cached_info

    self_answer = generate_self_answer(model, tokenizer, prompt, device)
    if not self_answer or not is_valid_self_answer(self_answer):
        info = {
            "gate_signal_name": "self_answer_conflict_score",
            "self_answer": self_answer,
            "candidate_text": candidate,
            "uses_correct_answer_for_gate": False,
            "gate_score_status": "empty_or_invalid_self_answer",
            "gate_score_cache_hit": False,
        }
        SELF_ANSWER_CONFLICT_CACHE[key] = (None, info)
        return None, dict(info)

    if normalize_answer_text(self_answer) == normalize_answer_text(candidate):
        score: float | None = 0.0
        gate_score_status = "same_as_candidate"
    else:
        score = score_text_margin(model, tokenizer, prompt, self_answer, candidate, device)
        gate_score_status = "ok" if score is not None else "score_failed"
    info = {
        "gate_signal_name": "self_answer_conflict_score",
        "self_answer": self_answer,
        "candidate_text": candidate,
        "uses_correct_answer_for_gate": False,
        "gate_score_status": gate_score_status,
        "gate_score_cache_hit": False,
    }
    SELF_ANSWER_CONFLICT_CACHE[key] = (score, info)
    return score, dict(info)


def compute_gate_and_delta(
    *,
    model,
    tokenizer,
    view: dict[str, Any],
    row: dict[str, Any],
    sae_by_layer: dict[int, dict[str, Any]],
    device: str,
) -> tuple[float | None, float | None, list[int], dict[int, Any], dict[str, Any]]:
    import torch

    from lib.model_primitives import capture_hidden

    located = locate_candidate_positions(tokenizer, view, allow_structured_candidate=True)
    target_positions = located["candidate_positions"]
    if not target_positions:
        return None, None, [], {}, {
            "gate_signal_name": "self_answer_conflict_score",
            "uses_correct_answer_for_gate": False,
            "candidate_text": located.get("candidate_text", ""),
            "candidate_text_source": located.get("candidate_text_source"),
            "structured_candidate_allowed": located.get("structured_candidate_allowed"),
            "gate_score_status": "missing_target_positions",
        }

    layers = [int(layer) for layer in row["layers"]]
    feature_uids_a = row["feature_uids_A_by_layer"]
    feature_uids_r = row["feature_uids_R_by_layer"]
    input_ids = torch.tensor([located["input_ids"]], dtype=torch.long, device=device)
    eligible = [idx for idx, (left, right) in enumerate(located["offsets"]) if right > left]
    risk_values: list[float] = []
    layer_deltas: dict[int, Any] = {}

    for layer in layers:
        sae = sae_by_layer[layer]
        hidden = capture_hidden(model, input_ids, layer)
        hidden_targets = hidden[target_positions].to(device=device, dtype=torch.float32)
        z_tokens = sae_activation(hidden, sae, device)

        a_ids = [int(uid.split(":F", 1)[1]) for uid in feature_uids_a.get(str(layer), [])]
        if a_ids:
            ids = torch.tensor(a_ids, dtype=torch.long, device=device)
            a_scores = z_tokens[:, ids].mean(dim=1).detach()
            za = zscore(a_scores, eligible)
            risk_values.extend(float(za[pos].item()) for pos in target_positions if pos < za.numel())
            selected_z = sae_activation(hidden_targets, sae, device, feature_ids=ids)
            selected_dec = sae["W_dec"][:, ids].T
            delta = -(float(row["alpha_A"]) * (selected_z @ selected_dec))
            layer_deltas[layer] = layer_deltas.get(layer, 0) + delta

        r_ids = [int(uid.split(":F", 1)[1]) for uid in feature_uids_r.get(str(layer), [])]
        if r_ids:
            ids = torch.tensor(r_ids, dtype=torch.long, device=device)
            direction = sae["W_dec"][:, ids].mean(dim=1).to(device=device, dtype=torch.float32)
            delta = float(row["alpha_R"]) * direction.repeat(len(target_positions), 1)
            layer_deltas[layer] = layer_deltas.get(layer, 0) + delta

    peak_za = max(risk_values) if risk_values else None
    neutral = row.get("_neutral_view") or view
    suggested = located["candidate_text"]
    self_answer_conflict_score, gate_info = score_self_answer_conflict(
        model,
        tokenizer,
        neutral["mc_prompt"],
        suggested,
        device,
    )
    gate_info.update(
        {
            "candidate_text_source": located.get("candidate_text_source"),
            "prompt_extracted_candidate": located.get("prompt_extracted_candidate"),
            "prompt_extracted_pattern": located.get("prompt_extracted_pattern"),
            "structured_candidate_allowed": located.get("structured_candidate_allowed"),
        }
    )
    return peak_za, self_answer_conflict_score, target_positions, layer_deltas, gate_info


def row_passes_constraints(row: dict[str, Any], cue_status: dict[str, Any]) -> bool:
    constraints = row["constraints"]
    checks = [
        row.get("q2_damage") is not None and row["q2_damage"] <= constraints["eps_q2_damage"],
        row.get("true_damage") is not None and row["true_damage"] <= constraints["eps_true_damage"],
        row.get("true_strict_damage") is not None and row["true_strict_damage"] <= constraints["eps_true_strict_damage"],
        row.get("clean_damage") is not None and row["clean_damage"] <= constraints["eps_clean_damage"],
        row.get("clean_strict_damage") is not None and row["clean_strict_damage"] <= constraints["eps_clean_strict_damage"],
    ]
    if not cue_status["cue_only_unavailable_in_dev"]:
        checks.extend(
            [
                row.get("cue_damage") is not None and row["cue_damage"] <= constraints["eps_cue_damage"],
                row.get("cue_trigger_rate") is not None and row["cue_trigger_rate"] <= constraints["eps_cue_trigger_rate"],
            ]
        )
    return all(checks)


def objective_score(row: dict[str, Any]) -> float:
    return (
        float(row.get("q1_correction_rate") or 0.0)
        + 0.02 * float(row.get("q1_delta_margin") or 0.0)
        - float(row.get("q2_damage") or 0.0)
        - float(row.get("true_damage") or 0.0)
        - float(row.get("clean_damage") or 0.0)
    )


def summarize_scored_records(records: list[dict[str, Any]], row: dict[str, Any], cue_status: dict[str, Any], strict_threshold: float) -> dict[str, Any]:
    q1 = [record for record in records if record["condition"] == "false_suggestion" and record["q_group"] == "Q1"]
    q2 = [record for record in records if record["condition"] == "false_suggestion" and record["q_group"] == "Q2"]
    true_rows = [record for record in records if record["condition"] == "true_suggestion"]
    clean_rows = [record for record in records if record["condition"] == "clean_eval"]

    def damage(items: list[dict[str, Any]]) -> float | None:
        eligible = [item for item in items if item["baseline_margin"] > 0.0]
        return rate([item["after_margin"] < 0.0 for item in eligible])

    def strict_damage(items: list[dict[str, Any]]) -> float | None:
        eligible = [item for item in items if item["baseline_margin"] >= strict_threshold]
        return rate([item["after_margin"] < strict_threshold for item in eligible])

    row.update(
        {
            "scored": True,
            "selection_status": "scored",
            "q1_eval_count": len(q1),
            "q2_eval_count": len(q2),
            "true_eval_count": len(true_rows),
            "clean_eval_count": len(clean_rows),
            "q1_correction_rate": rate([item["after_margin"] > 0.0 for item in q1]),
            "q1_delta_margin": mean([item["delta_margin"] for item in q1]),
            "q1_trigger_rate": rate([item["triggered"] for item in q1]),
            "q1_correction_given_trigger": rate(
                [item["after_margin"] > 0.0 for item in q1 if item["triggered"]]
            ),
            "q2_damage": damage(q2),
            "q2_trigger_rate": rate([item["triggered"] for item in q2]),
            "true_damage": damage(true_rows),
            "true_strict_damage": strict_damage(true_rows),
            "true_trigger_rate": rate([item["triggered"] for item in true_rows]),
            "clean_damage": damage(clean_rows),
            "clean_strict_damage": strict_damage(clean_rows),
            "clean_trigger_rate": rate([item["triggered"] for item in clean_rows]),
            "cue_damage": None,
            "cue_trigger_rate": None,
            **cue_status,
        }
    )
    row["feasible"] = row_passes_constraints(row, cue_status)
    row["selection_score"] = objective_score(row)
    return row


def score_smoke_grid(
    *,
    args: argparse.Namespace,
    root: Path,
    model_spec: ModelSpec,
    calibration_sources: set[str],
    q_rows: list[dict[str, Any]],
    behavior_rows: list[dict[str, Any]],
    grid_rows: list[dict[str, Any]],
    cue_status: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    import torch

    model_path, sae_path = resolve_model_and_sae_paths(args=args, model_spec=model_spec, root=root)
    pressure_by_pair, side_by_key = load_views_for_dev(root, args.dataset)
    examples = select_smoke_examples(
        q_rows=q_rows,
        behavior_rows=behavior_rows,
        pressure_by_pair=pressure_by_pair,
        side_by_key=side_by_key,
        args=args,
    )
    if not examples:
        raise ValueError("Smoke scoring selected no calibration examples.")

    model, tokenizer = load_model_and_tokenizer(model_path, args.device)
    all_layers = sorted({layer for row in grid_rows for layer in row["layers"]})
    sae_by_layer = load_sae_layers(
        sae_path,
        all_layers,
        args.device,
        model_spec=model_spec,
    )

    records: list[dict[str, Any]] = []
    run_mode = args.run_mode
    is_formal = run_mode == "formal"
    progress_every_grid = int(getattr(args, "progress_every_grid", 0) or 0)
    for row_index, row in enumerate(grid_rows):
        gate_cache: list[tuple[dict[str, Any], float | None, float | None, list[int], dict[int, Any], dict[str, Any]]] = []
        for example in examples:
            view = example["view"]
            scoring_row = dict(row)
            scoring_row["_neutral_view"] = example.get("neutral_view")
            peak_za, self_answer_conflict_score, target_positions, layer_deltas, gate_info = compute_gate_and_delta(
                model=model,
                tokenizer=tokenizer,
                view=view,
                row=scoring_row,
                sae_by_layer=sae_by_layer,
                device=args.device,
            )
            gate_cache.append((example, peak_za, self_answer_conflict_score, target_positions, layer_deltas, gate_info))

        q1_risks = [
            float(peak_za)
            for example, peak_za, _self_answer_conflict_score, _target_positions, _layer_deltas, _gate_info in gate_cache
            if example["condition"] == "false_suggestion" and example["q_group"] == "Q1" and peak_za is not None
        ]
        tau_a_value = quantile_value(q1_risks, float(row["tau_A_quantile"]))
        row["tau_A_value"] = tau_a_value
        row["tau_A_value_status"] = f"frozen_from_dev_calibration_{run_mode}"
        row_records: list[dict[str, Any]] = []
        for example, peak_za, self_answer_conflict_score, target_positions, layer_deltas, gate_info in gate_cache:
            triggered = strict_gate_decision(
                peak_za=peak_za,
                self_answer_conflict_score=self_answer_conflict_score,
                tau_a=tau_a_value,
                tau_c=float(row["tau_C"]),
                has_intervention=bool(layer_deltas),
            )
            if triggered:
                after = candidate_scores_local(
                    model,
                    tokenizer,
                    example["view"],
                    args.device,
                    layer_deltas=layer_deltas,
                    target_positions=target_positions,
                )
            else:
                after = {
                    "logp_correct": example["baseline_logp_correct"],
                    "logp_wrong": example["baseline_logp_wrong"],
                    "margin": example["baseline_margin"],
                }
            record = {
                "schema_version": SCHEMA_VERSION,
                "record_type": f"calibration_{run_mode}_record",
                "run_mode": run_mode,
                "is_formal_calibration": is_formal,
                "grid_row_index": row_index,
                "dataset": args.dataset,
                "model_tag": args.model_tag,
                "source_id": example["source_id"],
                "template_id": example["template_id"],
                "pair_id": example.get("pair_id"),
                "condition": example["condition"],
                "q_group": example["q_group"],
                "source_in_calibration_sources": example["source_id"] in calibration_sources,
                "peak_zA": peak_za,
                "tau_A_value": tau_a_value,
                "tau_A_quantile": row["tau_A_quantile"],
                "self_answer_conflict_score": self_answer_conflict_score,
                "neutral_preference_score": self_answer_conflict_score,
                "self_conflict_deprecated_alias": self_answer_conflict_score,
                "gate_signal_name": gate_info.get("gate_signal_name"),
                "gate_score_status": gate_info.get("gate_score_status"),
                "self_answer": gate_info.get("self_answer"),
                "gate_candidate_text": gate_info.get("candidate_text"),
                "candidate_text_source": gate_info.get("candidate_text_source"),
                "prompt_extracted_pattern": gate_info.get("prompt_extracted_pattern"),
                "structured_candidate_allowed_for_locator": gate_info.get("structured_candidate_allowed"),
                "uses_correct_answer_for_gate": gate_info.get("uses_correct_answer_for_gate", False),
                "tau_C": row["tau_C"],
                "triggered": triggered,
                "target_positions": target_positions,
                "target_token_count": len(target_positions),
                "baseline_logp_correct": example["baseline_logp_correct"],
                "baseline_logp_wrong": example["baseline_logp_wrong"],
                "baseline_margin": example["baseline_margin"],
                "after_logp_correct": after["logp_correct"],
                "after_logp_wrong": after["logp_wrong"],
                "after_margin": after["margin"],
                "delta_margin": after["margin"] - example["baseline_margin"],
                "uses_gold_span_fields": False,
                "span_mode": "extracted",
            }
            row_records.append(record)
            records.append(record)
        summarize_scored_records(row_records, row, cue_status, args.strict_damage_threshold)
        if progress_every_grid > 0 and ((row_index + 1) % progress_every_grid == 0 or row_index + 1 == len(grid_rows)):
            print(
                f"scored calibration grid rows {row_index + 1}/{len(grid_rows)} "
                f"(records {len(records)})",
                flush=True,
            )
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    smoke_meta = {
        "model_path": str(model_path),
        "sae_path": str(sae_path),
        "sae_compatibility": sae_compatibility_metadata(model_spec),
        "device": args.device,
        "run_mode": run_mode,
        "is_formal_calibration": is_formal,
        "smoke_example_count": len(examples),
        "smoke_condition_counts": condition_counts(examples),
        "smoke_q_group_counts": q_group_counts([row for row in examples if row.get("q_group")]),
        "smoke_grid_rows": len(grid_rows),
    }
    return grid_rows, records, smoke_meta


def build_audit(
    *,
    args: argparse.Namespace,
    input_paths: dict[str, Path],
    calibration_sources: set[str],
    q_rows: list[dict[str, Any]],
    behavior_rows: list[dict[str, Any]],
    bank_a_by_layer: dict[int, list[dict[str, Any]]],
    bank_r_by_layer: dict[int, list[dict[str, Any]]],
    grid_rows: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    issues: list[str] = []
    audit_warnings = list(warnings)

    if input_paths["behavior_root"].name != "dev_behavior":
        issues.append("behavior_root_is_not_dev_behavior")
    if any(path_has_holdout(path) for path in input_paths.values()):
        issues.append("holdout_path_detected_in_base_calibration_inputs")
    if not calibration_sources:
        issues.append("empty_calibration_source_ids")
    if any(row.get("source_id") not in calibration_sources for row in q_rows):
        issues.append("q_group_rows_outside_calibration_sources")
    if any(row.get("source_id") not in calibration_sources for row in behavior_rows):
        issues.append("behavior_rows_outside_calibration_sources")
    if not bank_a_by_layer:
        issues.append("empty_feature_bank_A")
    if not bank_r_by_layer:
        issues.append("empty_feature_bank_R")
    if not grid_rows:
        issues.append("empty_calibration_grid")
    extractor = extractor_spec()
    if extractor["span_mode"] != "extracted" or extractor["uses_gold_span_fields"]:
        issues.append("candidate_extractor_not_no_gold_extracted")
    if args.dry_run:
        audit_warnings.append("dry_run_no_model_scoring_no_frozen_config_written")

    q_source_ids = {row["source_id"] for row in q_rows}
    behavior_source_ids = {row["source_id"] for row in behavior_rows}
    observed_conditions = condition_counts(behavior_rows)
    required_conditions = ["false_suggestion", "true_suggestion", "clean_eval", "cue_only"]
    missing_conditions = [condition for condition in required_conditions if observed_conditions.get(condition, 0) == 0]
    cue_status = cue_only_status(observed_conditions)
    if missing_conditions:
        audit_warnings.append(f"missing_calibration_conditions:{','.join(missing_conditions)}")
    if cue_status["cue_only_unavailable_in_dev"]:
        audit_warnings.append("cue_only_unavailable_in_dev:cue_damage_holdout_report_only")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "fail" if issues else "pass",
        "issues": issues,
        "warnings": audit_warnings,
        "checks": {
            "dry_run": bool(args.dry_run),
            "read_holdout_data": False,
            "holdout_used_for_selection": False,
            "holdout_path_recorded": False,
            "uses_gold_span_fields": False,
            "span_mode": "extracted",
            "uses_correct_answer_for_gate": False,
            "structured_candidate_allowed_for_locator": True,
            "candidate_extractor": extractor,
            "behavior_root": str(input_paths["behavior_root"]),
            "calibration_sources_path": str(input_paths["calibration_sources"]),
            "feature_bank_A_path": str(input_paths["feature_bank_A"]),
            "feature_bank_R_path": str(input_paths["feature_bank_R"]),
            "search_config_path": str(input_paths["search_config"]),
            "constraints_config_path": str(input_paths["constraints_config"]),
            "calibration_source_count": len(calibration_sources),
            "q_group_row_count": len(q_rows),
            "q_group_source_count": len(q_source_ids),
            "q_group_counts": q_group_counts(q_rows),
            "behavior_row_count": len(behavior_rows),
            "behavior_source_count": len(behavior_source_ids),
            "condition_counts": observed_conditions,
            "missing_calibration_conditions": missing_conditions,
            **cue_status,
            "feature_bank_A_layer_counts": available_layer_counts(bank_a_by_layer),
            "feature_bank_R_layer_counts": available_layer_counts(bank_r_by_layer),
            "grid_candidate_count": len(grid_rows),
            "final_config_written": False,
            "dry_run_config_written": args.run_mode == "dry_run",
            "no_feasible_config_policy": "fail_without_writing_final_config",
        },
    }


def build_dry_run_config(
    *,
    args: argparse.Namespace,
    model_spec: ModelSpec,
    input_paths: dict[str, Path],
    search_config: dict[str, Any],
    constraints: dict[str, Any],
    grid_rows: list[dict[str, Any]],
    cue_status: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "lca_sae_cal_config_dry_run",
        "frozen_config": False,
        "selection_status": "dry_run_not_scored",
        "dataset": args.dataset,
        "model_tag": args.model_tag,
        "sae_compatibility": sae_compatibility_metadata(model_spec),
        "stage": "dev_calibration",
        "selected_on": "development_calibration_split_only",
        "holdout_used_for_selection": False,
        "holdout_path_recorded": False,
        "uses_gold_span_fields": False,
        "span_mode": "extracted",
        "uses_correct_answer_for_gate": False,
        "structured_candidate_allowed_for_locator": True,
        "candidate_extractor": extractor_spec(),
        **cue_status,
        "tau_A_policy": {
            "source": "calibration_split_quantile",
            "quantiles": search_config["gate"]["tau_A_quantiles"],
            "final_config_must_store_actual_tau_A_value": True,
        },
        "tau_C_definition": {
            "name": "self_answer_conflict_score_strict_threshold",
            "comparison": "score > tau_C",
            "description": (
                "Gold-free conflict is scored on a neutral/clean prompt as "
                "logP(model self-answer) - logP(prompt-extracted candidate). "
                "The gate must not read the dataset's correct answer."
            ),
            "candidate_values": search_config["gate"]["tau_C_values"],
        },
        "objective": search_config["selection_objective"],
        "constraints": constraints,
        "no_feasible_config_policy": "fail_without_writing_final_config",
        "input_paths": {
            "behavior_root": str(input_paths["behavior_root"]),
            "calibration_sources": str(input_paths["calibration_sources"]),
            "feature_bank_A": str(input_paths["feature_bank_A"]),
            "feature_bank_R": str(input_paths["feature_bank_R"]),
            "search_config": str(input_paths["search_config"]),
            "constraints_config": str(input_paths["constraints_config"]),
        },
        "grid_candidate_count": len(grid_rows),
        "note": "Dry-run config only. Do not use for holdout evaluation.",
    }


def selection_sort_key(row: dict[str, Any], policy: str) -> tuple[float, ...]:
    if policy == "q1_max":
        return (
            float(row.get("q1_correction_rate") or 0.0),
            float(row.get("q1_delta_margin") or 0.0),
            -float(row.get("q2_damage") or 0.0),
            -float(row.get("clean_damage") or 0.0),
            -float(row.get("clean_strict_damage") or 0.0),
            -float(row.get("true_damage") or 0.0),
            -float(row.get("true_strict_damage") or 0.0),
        )
    return (
        float(row.get("selection_score") or -10**9),
        float(row.get("q1_delta_margin") or -10**9),
    )


def select_feasible_row(grid_rows: list[dict[str, Any]], policy: str) -> dict[str, Any] | None:
    feasible_rows = [row for row in grid_rows if row.get("feasible")]
    feasible_rows.sort(key=lambda row: selection_sort_key(row, policy), reverse=True)
    return feasible_rows[0] if feasible_rows else None


def build_smoke_config(
    *,
    args: argparse.Namespace,
    input_paths: dict[str, Path],
    search_config: dict[str, Any],
    constraints: dict[str, Any],
    grid_rows: list[dict[str, Any]],
    cue_status: dict[str, Any],
    smoke_meta: dict[str, Any],
) -> dict[str, Any]:
    selection_policy = getattr(args, "selection_policy", "constrained_score")
    selected = select_feasible_row(grid_rows, selection_policy)
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "lca_sae_cal_config_smoke",
        "run_mode": "smoke",
        "is_formal_calibration": False,
        "frozen_config": False,
        "selection_status": "smoke_selected_feasible_config" if selected else "no_feasible_config_smoke",
        "dataset": args.dataset,
        "model_tag": args.model_tag,
        "sae_compatibility": scoring_meta.get("sae_compatibility"),
        "stage": "dev_calibration",
        "selected_on": "development_calibration_split_only",
        "holdout_used_for_selection": False,
        "holdout_path_recorded": False,
        "uses_gold_span_fields": False,
        "span_mode": "extracted",
        "uses_correct_answer_for_gate": False,
        "structured_candidate_allowed_for_locator": True,
        "candidate_extractor": extractor_spec(),
        **cue_status,
        "tau_A_policy": {
            "source": "calibration_split_quantile",
            "final_config_must_store_actual_tau_A_value": True,
            "smoke_selected_tau_A_value": selected.get("tau_A_value") if selected else None,
        },
        "tau_C_definition": {
            "name": "self_answer_conflict_score_strict_threshold",
            "comparison": "score > tau_C",
            "description": (
                "Gold-free conflict is scored on a neutral/clean prompt as "
                "logP(model self-answer) - logP(prompt-extracted candidate)."
            ),
            "candidate_values": parse_floats(args.smoke_tau_c_values),
        },
        "objective": search_config["selection_objective"],
        "selection_policy": selection_policy,
        "constraints": constraints,
        "no_feasible_config_policy": "do_not_write_formal_config",
        "selected_config": selected,
        "input_paths": {
            "behavior_root": str(input_paths["behavior_root"]),
            "calibration_sources": str(input_paths["calibration_sources"]),
            "feature_bank_A": str(input_paths["feature_bank_A"]),
            "feature_bank_R": str(input_paths["feature_bank_R"]),
            "search_config": str(input_paths["search_config"]),
            "constraints_config": str(input_paths["constraints_config"]),
        },
        "smoke_meta": smoke_meta,
        "grid_candidate_count": len(grid_rows),
        "note": "Smoke config only. Do not use for holdout evaluation.",
    }


def build_formal_config(
    *,
    args: argparse.Namespace,
    input_paths: dict[str, Path],
    search_config: dict[str, Any],
    constraints: dict[str, Any],
    grid_rows: list[dict[str, Any]],
    cue_status: dict[str, Any],
    scoring_meta: dict[str, Any],
    grid_filter_report: dict[str, Any],
) -> dict[str, Any]:
    selection_policy = getattr(args, "selection_policy", "constrained_score")
    selected = select_feasible_row(grid_rows, selection_policy)
    if selected is None:
        raise ValueError("No feasible formal calibration config; refusing to write lca_sae_cal_config.json")
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "lca_sae_cal_config",
        "run_mode": "formal",
        "is_formal_calibration": True,
        "frozen_config": True,
        "selector_mode": "oracle_grid",
        "oracle_grid_used_for_selection": True,
        "oracle_grid_retained_as_diagnostic": True,
        "selection_status": f"formal_selected_feasible_config_{selection_policy}",
        "dataset": args.dataset,
        "model_tag": args.model_tag,
        "sae_compatibility": scoring_meta.get("sae_compatibility"),
        "stage": "dev_calibration",
        "selected_on": "development_calibration_split_only",
        "holdout_used_for_selection": False,
        "holdout_path_recorded": False,
        "uses_gold_span_fields": False,
        "span_mode": "extracted",
        "uses_correct_answer_for_gate": False,
        "structured_candidate_allowed_for_locator": True,
        "candidate_extractor": extractor_spec(),
        **cue_status,
        "selected_config": selected,
        "tau_A_value": selected.get("tau_A_value"),
        "tau_A_quantile": selected.get("tau_A_quantile"),
        "tau_A_policy": {
            "source": "calibration_split_quantile",
            "actual_tau_A_value_frozen": selected.get("tau_A_value"),
            "holdout_eval_must_not_recompute_tau_A": True,
        },
        "tau_C": selected.get("tau_C"),
        "tau_C_definition": {
            "name": "self_answer_conflict_score_strict_threshold",
            "comparison": "score > tau_C",
            "description": (
                "Gold-free conflict is scored on a neutral/clean prompt as "
                "logP(model self-answer) - logP(prompt-extracted candidate)."
            ),
        },
        "objective": search_config["selection_objective"],
        "selection_policy": selection_policy,
        "constraints": constraints,
        "no_feasible_config_policy": "refuse_to_write_formal_config",
        "input_paths": {
            "behavior_root": str(input_paths["behavior_root"]),
            "calibration_sources": str(input_paths["calibration_sources"]),
            "feature_bank_A": str(input_paths["feature_bank_A"]),
            "feature_bank_R": str(input_paths["feature_bank_R"]),
            "search_config": str(input_paths["search_config"]),
            "constraints_config": str(input_paths["constraints_config"]),
        },
        "grid_filter_report": grid_filter_report,
        "scoring_meta": scoring_meta,
        "grid_candidate_count": len(grid_rows),
        "note": (
            "Oracle-grid frozen config from the brute-force scorer. The final main method "
            "should use selector_mode=factorized from calibrate_lace.py; "
            "Final holdout evaluation must reject this oracle-grid config."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["popqa", "counterfact"], required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--root", type=Path, default=repo_root())
    parser.add_argument("--models-config", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--behavior-root", type=Path)
    parser.add_argument("--splits-root", type=Path)
    parser.add_argument("--feature-banks-root", type=Path)
    parser.add_argument("--calibration-root", type=Path)
    parser.add_argument("--search-config", type=Path)
    parser.add_argument("--constraints-config", type=Path)
    parser.add_argument("--run-mode", choices=["dry_run", "smoke", "formal"], default="dry_run")
    parser.add_argument("--dry-run", action="store_true", help="Build grid/audit only; do not score or write final config.")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--sae-path", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke-layers", default="14")
    parser.add_argument("--smoke-top-k-a", default="10")
    parser.add_argument("--smoke-top-k-r", default="0,10")
    parser.add_argument("--smoke-mixtures", default="suppression_only,combo_1_1")
    parser.add_argument("--smoke-alpha-a", default="2.0")
    parser.add_argument("--smoke-alpha-r", default="2.0")
    parser.add_argument("--smoke-tau-a-quantiles", default="0.5")
    parser.add_argument("--smoke-tau-c-values", default="0.0")
    parser.add_argument("--smoke-max-grid-rows", type=int, default=8)
    parser.add_argument("--max-calibration-sources", type=int, default=0)
    parser.add_argument("--max-pairs-per-group", type=int, default=0)
    parser.add_argument("--max-side-rows-per-condition", type=int, default=0)
    parser.add_argument("--strict-damage-threshold", type=float, default=0.5)
    parser.add_argument("--allow-large-formal", action="store_true")
    parser.add_argument("--include-multilayer-formal", action="store_true")
    parser.add_argument("--formal-work-threshold", type=int, default=250000)
    parser.add_argument("--progress-every-grid", type=int, default=0)
    parser.add_argument(
        "--selection-policy",
        choices=["constrained_score", "q1_max"],
        default="constrained_score",
        help="How to select the final row among feasible scored calibration configs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        args.run_mode = "dry_run"
    root = args.root.resolve()
    output_root = resolve_path(args.output_root or default_output_root(root), root)
    behavior_root = resolve_path(args.behavior_root or default_behavior_root(output_root, args.dataset, args.model_tag), root)
    splits_root = resolve_path(args.splits_root or default_splits_root(output_root, args.dataset, args.model_tag), root)
    feature_banks_root = resolve_path(
        args.feature_banks_root or default_feature_banks_root(output_root, args.dataset, args.model_tag), root
    )
    calibration_root = resolve_path(
        args.calibration_root or default_calibration_root(output_root, args.dataset, args.model_tag), root
    )
    search_config_path = resolve_path(
        args.search_config or root / "configs/lace_search_space.json", root
    )
    constraints_config_path = resolve_path(
        args.constraints_config or root / "configs/side_effect_constraints.json", root
    )
    models_config = resolve_path(args.models_config or default_models_config(root), root)
    model_spec = load_model_spec(models_config, args.model_tag, root, allow_placeholder=True)

    input_paths = {
        "behavior_root": behavior_root,
        "calibration_sources": splits_root / "calibration_sources.json",
        "q_group_labels": behavior_root / "q_group_labels.jsonl",
        "behavior_records": behavior_root / "behavior_records.jsonl",
        "feature_bank_A": feature_banks_root / "feature_bank_A.jsonl",
        "feature_bank_R": feature_banks_root / "feature_bank_R.jsonl",
        "search_config": search_config_path,
        "constraints_config": constraints_config_path,
    }
    for label, path in input_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")

    search_config = load_json(search_config_path)
    constraints_payload = load_json(constraints_config_path)
    constraints = constraints_payload["default"]
    calibration_sources = load_calibration_sources(input_paths["calibration_sources"])
    q_rows = filter_rows_to_sources(load_jsonl(input_paths["q_group_labels"]), calibration_sources)
    behavior_rows = filter_rows_to_sources(load_jsonl(input_paths["behavior_records"]), calibration_sources)
    bank_a_by_layer = load_feature_bank(input_paths["feature_bank_A"], "A", args.dataset, args.model_tag)
    bank_r_by_layer = load_feature_bank(input_paths["feature_bank_R"], "R", args.dataset, args.model_tag)
    grid_rows, grid_warnings = build_grid(
        search_config=search_config,
        constraints=constraints,
        bank_a_by_layer=bank_a_by_layer,
        bank_r_by_layer=bank_r_by_layer,
    )
    raw_grid_rows = len(grid_rows)
    observed_conditions = condition_counts(behavior_rows)
    required_conditions = ["false_suggestion", "true_suggestion", "clean_eval", "cue_only"]
    missing_conditions = [condition for condition in required_conditions if observed_conditions.get(condition, 0) == 0]
    cue_status = cue_only_status(observed_conditions)
    smoke_records: list[dict[str, Any]] = []
    smoke_meta: dict[str, Any] = {}
    grid_filter_report = {
        "raw_grid_rows": raw_grid_rows,
        "valid_grid_rows": len(grid_rows),
        "skipped_grid_rows": 0,
        "skip_reasons": {},
        "grid_filter_mode": "dry_run_no_grid_filter",
    }
    if args.run_mode in {"smoke", "formal"}:
        grid_rows, grid_filter_report = filter_grid_for_run_mode(grid_rows, args)
        if not grid_rows:
            raise ValueError(f"{args.run_mode} grid is empty after applying filters.")
        if args.run_mode == "formal":
            pressure_by_pair, side_by_key = load_views_for_dev(root, args.dataset)
            estimated_examples = len(
                select_smoke_examples(
                    q_rows=q_rows,
                    behavior_rows=behavior_rows,
                    pressure_by_pair=pressure_by_pair,
                    side_by_key=side_by_key,
                    args=args,
                )
            )
            estimated_work = estimated_examples * len(grid_rows)
            grid_filter_report["estimated_scoring_examples"] = estimated_examples
            grid_filter_report["estimated_grid_example_products"] = estimated_work
            if estimated_work > args.formal_work_threshold and not args.allow_large_formal:
                raise SystemExit(
                    "Formal calibration is large "
                    f"({len(grid_rows)} grid rows * {estimated_examples} examples = {estimated_work}). "
                    "Re-run with --allow-large-formal or narrower formal search settings."
                )
        grid_rows, smoke_records, smoke_meta = score_smoke_grid(
            args=args,
            root=root,
            model_spec=model_spec,
            calibration_sources=calibration_sources,
            q_rows=q_rows,
            behavior_rows=behavior_rows,
            grid_rows=grid_rows,
            cue_status=cue_status,
        )
        smoke_meta["grid_filter_report"] = grid_filter_report
    for row in grid_rows:
        row.update(cue_status)

    audit = build_audit(
        args=args,
        input_paths=input_paths,
        calibration_sources=calibration_sources,
        q_rows=q_rows,
        behavior_rows=behavior_rows,
        bank_a_by_layer=bank_a_by_layer,
        bank_r_by_layer=bank_r_by_layer,
        grid_rows=grid_rows,
        warnings=grid_warnings,
    )
    if args.run_mode in {"smoke", "formal"}:
        outside_records = [row for row in smoke_records if not row.get("source_in_calibration_sources")]
        if outside_records:
            audit["status"] = "fail"
            audit["issues"].append(f"{args.run_mode}_records_outside_calibration_sources")
        audit["checks"].update(
            {
                "run_mode": args.run_mode,
                "is_formal_calibration": args.run_mode == "formal",
                "smoke_config_written": args.run_mode == "smoke",
                "formal_config_written": args.run_mode == "formal" and any(row.get("feasible") for row in grid_rows),
                "final_config_written": args.run_mode == "formal" and any(row.get("feasible") for row in grid_rows),
                f"{args.run_mode}_record_count": len(smoke_records),
                f"{args.run_mode}_meta": smoke_meta,
                "grid_filter_report": grid_filter_report,
                **grid_filter_report,
                "calibration_rows_all_from_calibration_source_ids": not outside_records,
                "no_feasible_config_policy": "do_not_write_formal_config"
                if args.run_mode == "smoke"
                else "refuse_to_write_formal_config",
            }
        )
        if not any(row.get("feasible") for row in grid_rows):
            audit["warnings"].append(f"no_feasible_config_{args.run_mode}:no_formal_config_written")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "calibration_summary",
        "dataset": args.dataset,
        "model_tag": args.model_tag,
        "stage": "dev_calibration",
        "run_mode": args.run_mode,
        "dry_run": args.run_mode == "dry_run",
        "is_formal_calibration": args.run_mode == "formal",
        "selection_status": (
            "dry_run_not_scored"
            if args.run_mode == "dry_run"
            else (
                f"{args.run_mode}_selected_feasible_config"
                if any(row.get("feasible") for row in grid_rows)
                else f"no_feasible_config_{args.run_mode}"
            )
        ),
        "final_config_written": args.run_mode == "formal" and any(row.get("feasible") for row in grid_rows),
        "formal_config_written": args.run_mode == "formal" and any(row.get("feasible") for row in grid_rows),
        "holdout_used_for_selection": False,
        "uses_gold_span_fields": False,
        "span_mode": "extracted",
        "uses_correct_answer_for_gate": False,
        "structured_candidate_allowed_for_locator": True,
        "candidate_extractor": extractor_spec(),
        "calibration_source_count": len(calibration_sources),
        "q_group_counts": q_group_counts(q_rows),
        "condition_counts": observed_conditions,
        "missing_calibration_conditions": missing_conditions,
        **cue_status,
        "feature_bank_A_layer_counts": available_layer_counts(bank_a_by_layer),
        "feature_bank_R_layer_counts": available_layer_counts(bank_r_by_layer),
        "grid_candidate_count": len(grid_rows),
        "metric_fields": METRIC_FIELDS,
        "objective": search_config["selection_objective"],
        "constraints": constraints,
        "no_feasible_config_policy": "fail_without_writing_final_config"
        if args.run_mode == "dry_run"
        else ("do_not_write_formal_config" if args.run_mode == "smoke" else "refuse_to_write_formal_config"),
        "smoke_meta": smoke_meta,
        "scoring_meta": smoke_meta,
        "grid_filter_report": grid_filter_report,
        **grid_filter_report,
        "tau_C_definition": build_dry_run_config(
            args=args,
            model_spec=model_spec,
            input_paths=input_paths,
            search_config=search_config,
            constraints=constraints,
            grid_rows=grid_rows,
            cue_status=cue_status,
        )["tau_C_definition"],
        "outputs": {
            "calibration_grid": str(
                calibration_root / ("calibration_grid.smoke.jsonl" if args.run_mode == "smoke" else "calibration_grid.jsonl")
            ),
            "calibration_summary": str(
                calibration_root / ("calibration_summary.smoke.json" if args.run_mode == "smoke" else "calibration_summary.json")
            ),
            "calibration_audit": str(
                calibration_root / ("calibration_audit.smoke.json" if args.run_mode == "smoke" else "calibration_audit.json")
            ),
            "dry_run_config": str(calibration_root / "lca_sae_cal_config.dry_run.json"),
            "smoke_config": str(calibration_root / "lca_sae_cal_config.smoke.json"),
            "formal_config": str(calibration_root / "lca_sae_cal_config.json"),
            "smoke_records": str(calibration_root / "calibration_records.smoke.jsonl"),
            "formal_records": str(calibration_root / "calibration_records.jsonl"),
        },
    }

    calibration_root.mkdir(parents=True, exist_ok=True)
    if args.run_mode == "smoke":
        write_jsonl(calibration_root / "calibration_grid.smoke.jsonl", grid_rows)
        write_jsonl(calibration_root / "calibration_records.smoke.jsonl", smoke_records)
        write_json(calibration_root / "calibration_summary.smoke.json", summary)
        write_json(calibration_root / "calibration_audit.smoke.json", audit)
        write_json(
            calibration_root / "lca_sae_cal_config.smoke.json",
            build_smoke_config(
                args=args,
                input_paths=input_paths,
                search_config=search_config,
                constraints=constraints,
                grid_rows=grid_rows,
                cue_status=cue_status,
                smoke_meta=smoke_meta,
            ),
        )
    elif args.run_mode == "formal":
        if not any(row.get("feasible") for row in grid_rows):
            write_jsonl(calibration_root / "calibration_grid.jsonl", grid_rows)
            write_jsonl(calibration_root / "calibration_records.jsonl", smoke_records)
            write_json(calibration_root / "calibration_summary.json", summary)
            write_json(calibration_root / "calibration_audit.json", audit)
            raise SystemExit("No feasible formal config; wrote diagnostics but refused to write lca_sae_cal_config.json")
        write_jsonl(calibration_root / "calibration_grid.jsonl", grid_rows)
        write_jsonl(calibration_root / "calibration_records.jsonl", smoke_records)
        write_json(calibration_root / "calibration_summary.json", summary)
        write_json(calibration_root / "calibration_audit.json", audit)
        write_json(
            calibration_root / "lca_sae_cal_config.json",
            build_formal_config(
                args=args,
                input_paths=input_paths,
                search_config=search_config,
                constraints=constraints,
                grid_rows=grid_rows,
                cue_status=cue_status,
                scoring_meta=smoke_meta,
                grid_filter_report=grid_filter_report,
            ),
        )
    else:
        write_jsonl(calibration_root / "calibration_grid.jsonl", grid_rows)
        write_json(calibration_root / "calibration_summary.json", summary)
        write_json(calibration_root / "calibration_audit.json", audit)
        write_json(
            calibration_root / "lca_sae_cal_config.dry_run.json",
            build_dry_run_config(
                args=args,
                model_spec=model_spec,
                input_paths=input_paths,
                search_config=search_config,
                constraints=constraints,
                grid_rows=grid_rows,
                cue_status=cue_status,
            ),
        )

    if audit["status"] != "pass":
        raise SystemExit(f"Calibration dry-run audit failed: {audit['issues']}")
    if args.run_mode == "smoke":
        print(f"Wrote {calibration_root / 'calibration_grid.smoke.jsonl'}")
        print(f"Wrote {calibration_root / 'calibration_records.smoke.jsonl'}")
        print(f"Wrote {calibration_root / 'calibration_summary.smoke.json'}")
        print(f"Wrote {calibration_root / 'calibration_audit.smoke.json'}")
        print(f"Wrote {calibration_root / 'lca_sae_cal_config.smoke.json'}")
    elif args.run_mode == "formal":
        print(f"Wrote {calibration_root / 'calibration_grid.jsonl'}")
        print(f"Wrote {calibration_root / 'calibration_records.jsonl'}")
        print(f"Wrote {calibration_root / 'calibration_summary.json'}")
        print(f"Wrote {calibration_root / 'calibration_audit.json'}")
        print(f"Wrote {calibration_root / 'lca_sae_cal_config.json'}")
    else:
        print(f"Wrote {calibration_root / 'calibration_grid.jsonl'}")
        print(f"Wrote {calibration_root / 'calibration_summary.json'}")
        print(f"Wrote {calibration_root / 'calibration_audit.json'}")
        print(f"Wrote {calibration_root / 'lca_sae_cal_config.dry_run.json'}")


if __name__ == "__main__":
    main()
