#!/usr/bin/env python3
"""Internal scoring runtime shared by LACE calibration and final evaluation.

The public final entry point fixes the vector construction and gate to the
values stored in a frozen development configuration. Historical alternate
branches remain internal because the development selector shares their
low-level scoring primitives; they are not public reproduction commands.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

INTERNAL_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = INTERNAL_DIR.parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.candidate_extractor import extractor_spec, locate_candidate_positions  # noqa: E402
from lib.dataset_registry import load_dataset_stage  # noqa: E402
from lib.gate import strict_gate_decision  # noqa: E402
from lib.feature_primitives import pair_lookup, pooled_claim_hidden, sae_activation  # noqa: E402
from lib import holdout_utils as PHASE5  # noqa: E402
from lib.io_utils import load_json, load_jsonl, repo_root, resolve_path, write_json, write_jsonl  # noqa: E402
from lib.model_primitives import capture_hidden  # noqa: E402
from lib.model_registry import load_model_spec, sae_compatibility_metadata  # noqa: E402
from lib.span_utils import locate_regions  # noqa: E402


SCHEMA_VERSION = "lace_intervention_runtime_v1"
VALID_GATE_MODES = {
    "peak_zA_tau",
    "peak_zA_only",
    "no_gate_false_only",
    "no_gate_all_extracted",
    "oracle_q1",
}
VALID_MIXTURE_MODES = {
    "all",
    "selected",
    "enhancement_only",
    "suppression_only",
    "combo_1_1",
    "suppress_heavy_2_1",
    "enhance_heavy_1_2",
}
VALID_VECTOR_CONSTRUCTIONS = {
    "full_va",
    "no_alignment",
    "random_feature_alignment",
    "dense_q2_q1_direction",
    "random_dense_direction",
}
ALL_MIXTURE_MODES = [
    "enhancement_only",
    "suppression_only",
    "combo_1_1",
    "suppress_heavy_2_1",
    "enhance_heavy_1_2",
]


def load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PHASE4 = load_module("lace_calibration_runtime_for_vector_method", INTERNAL_DIR / "calibration_runtime.py")


def default_models_config(root: Path) -> Path:
    local = root / "configs/models.local.json"
    if local.exists():
        return local
    return root / "configs/models.example.json"


def default_output_root(root: Path) -> Path:
    return root / "outputs"


def default_config_path(output_root: Path, dataset: str, model_tag: str) -> Path:
    return output_root / dataset / model_tag / "calibration" / "lca_sae_cal_config.json"


def default_behavior_root(output_root: Path, dataset: str, model_tag: str) -> Path:
    return output_root / dataset / model_tag / "holdout_behavior"


def default_dev_behavior_root(output_root: Path, dataset: str, model_tag: str) -> Path:
    return output_root / dataset / model_tag / "dev_behavior"


def default_splits_root(output_root: Path, dataset: str, model_tag: str) -> Path:
    return output_root / dataset / model_tag / "dev_splits"


def default_feature_banks_root(output_root: Path, dataset: str, model_tag: str) -> Path:
    return output_root / dataset / model_tag / "feature_banks"


def default_output_dir(output_root: Path, dataset: str, model_tag: str) -> Path:
    return output_root / dataset / model_tag / "holdout_lace_eval"


def resolve_cli_path(path: str | Path | None, root: Path) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return resolve_path(candidate, root)


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_floats(raw: str) -> list[float]:
    return [float(item) for item in parse_csv(raw)]


def expand_mixture_modes(raw: str, selected_mode: str | None = None) -> list[str]:
    requested = parse_csv(raw)
    unknown = sorted(set(requested) - VALID_MIXTURE_MODES)
    if unknown:
        raise ValueError(f"Unknown mixture modes: {unknown}")
    output: list[str] = []
    for mode in requested:
        if mode == "all":
            output.extend(ALL_MIXTURE_MODES)
            if selected_mode:
                output.append(selected_mode)
        elif mode == "selected":
            if selected_mode:
                output.append(selected_mode)
            else:
                output.append(mode)
        else:
            output.append(mode)
    deduped: list[str] = []
    seen = set()
    for mode in output:
        if mode in seen:
            continue
        seen.add(mode)
        deduped.append(mode)
    return deduped


def parse_vector_constructions(raw: str) -> list[str]:
    requested = parse_csv(raw)
    unknown = sorted(set(requested) - VALID_VECTOR_CONSTRUCTIONS)
    if unknown:
        raise ValueError(f"Unknown vector constructions: {unknown}")
    output: list[str] = []
    seen = set()
    for item in requested:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output or ["full_va"]


def load_discovery_sources(path: Path) -> set[str]:
    payload = load_json(path)
    if payload.get("split") == "discovery":
        source_ids = {row["source_id"] for row in payload.get("sources", [])}
    else:
        source_ids = set(payload.get("discovery_source_ids", []))
    if not source_ids:
        raise ValueError(f"No discovery source ids found in {path}")
    return source_ids


def ordered_group_limited(records: list[dict[str, Any]], max_per_group: int) -> list[dict[str, Any]]:
    if max_per_group <= 0:
        return records
    counts: Counter[str] = Counter()
    output: list[dict[str, Any]] = []
    for row in records:
        group = row.get("q_group")
        if group not in {"Q1", "Q2"}:
            continue
        if counts[group] >= max_per_group:
            continue
        output.append(row)
        counts[group] += 1
    return output


def selected_feature_rows(
    *,
    feature_bank_path: Path,
    selected_by_layer: dict[str, list[str]],
    expected_bank: str,
    dataset: str,
    model_tag: str,
) -> dict[int, list[dict[str, Any]]]:
    rows = PHASE4.load_feature_bank(feature_bank_path, expected_bank, dataset, model_tag)
    by_uid = {
        row["feature_uid"]: row
        for layer_rows in rows.values()
        for row in layer_rows
    }
    output: dict[int, list[dict[str, Any]]] = {}
    missing: list[str] = []
    for layer_raw, uids in selected_by_layer.items():
        layer = int(layer_raw)
        output[layer] = []
        for uid in uids:
            row = by_uid.get(uid)
            if row is None:
                missing.append(uid)
            else:
                output[layer].append(row)
    if missing:
        raise ValueError(f"Selected {expected_bank} feature UIDs missing from feature bank: {missing[:10]}")
    return output


def feature_ids_and_coeffs(rows: list[dict[str, Any]], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.tensor([int(row["feature_id"]) for row in rows], dtype=torch.long, device=device)
    coeffs = torch.tensor(
        [max(0.0, float(row.get("causal_score") or 0.0)) for row in rows],
        dtype=torch.float32,
        device=device,
    )
    if coeffs.numel() and coeffs.float().norm().item() <= 1e-12:
        coeffs = torch.ones_like(coeffs)
    return ids, coeffs


def match_norm(vector: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    vector = vector.float()
    reference = reference.float()
    norm = vector.norm()
    ref_norm = reference.norm()
    if norm.item() <= 1e-12 or ref_norm.item() <= 1e-12:
        return vector
    return vector / norm * ref_norm


def sparse_vector(
    *,
    sae: dict[str, torch.Tensor],
    rows: list[dict[str, Any]],
    robust_direction: torch.Tensor,
    sign: float,
    device: str,
) -> torch.Tensor:
    if not rows:
        return torch.zeros(sae["W_dec"].shape[0], dtype=torch.float32, device=device)
    ids, coeffs = feature_ids_and_coeffs(rows, device)
    residual_norm = robust_direction.float().norm() + 1e-12
    vector = residual_norm * (sae["W_dec"][:, ids].float() @ coeffs.float())
    return match_norm(float(sign) * vector, robust_direction)


def raw_sparse_vector(
    *,
    sae: dict[str, torch.Tensor],
    rows: list[dict[str, Any]],
    sign: float,
    device: str,
) -> torch.Tensor:
    if not rows:
        return torch.zeros(sae["W_dec"].shape[0], dtype=torch.float32, device=device)
    ids, coeffs = feature_ids_and_coeffs(rows, device)
    return float(sign) * (sae["W_dec"][:, ids].float() @ coeffs.float())


def random_feature_vector(
    *,
    sae: dict[str, torch.Tensor],
    rows: list[dict[str, Any]],
    robust_direction: torch.Tensor,
    sign: float,
    seed: int,
    device: str,
) -> torch.Tensor:
    if not rows:
        return torch.zeros(sae["W_dec"].shape[0], dtype=torch.float32, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    feature_count = sae["W_dec"].shape[1]
    sample_count = min(len(rows), feature_count)
    ids = torch.randperm(feature_count, generator=generator, device=device)[:sample_count].long()
    coeffs = torch.tensor(
        [max(0.0, float(row.get("causal_score") or 0.0)) for row in rows[:sample_count]],
        dtype=torch.float32,
        device=device,
    )
    if coeffs.numel() and coeffs.float().norm().item() <= 1e-12:
        coeffs = torch.ones_like(coeffs)
    vector = sae["W_dec"][:, ids].float() @ coeffs.float()
    return match_norm(float(sign) * vector, robust_direction)


def random_dense_vector(*, robust_direction: torch.Tensor, seed: int, device: str) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    vector = torch.randn(
        robust_direction.shape,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    return match_norm(vector, robust_direction)


def compose_vector(
    *,
    robust_vector: torch.Tensor,
    suppress_vector: torch.Tensor,
    robust_direction: torch.Tensor,
    mixture_mode: str,
) -> torch.Tensor:
    if mixture_mode == "enhancement_only":
        return robust_vector
    if mixture_mode == "suppression_only":
        return suppress_vector
    if mixture_mode == "combo_1_1":
        return match_norm(robust_vector + suppress_vector, robust_direction)
    if mixture_mode == "suppress_heavy_2_1":
        return match_norm(robust_vector + 2.0 * suppress_vector, robust_direction)
    if mixture_mode == "enhance_heavy_1_2":
        return match_norm(2.0 * robust_vector + suppress_vector, robust_direction)
    raise ValueError(f"Unknown mixture_mode={mixture_mode!r}")


def compose_raw_vector(
    *,
    robust_vector: torch.Tensor,
    suppress_vector: torch.Tensor,
    mixture_mode: str,
) -> torch.Tensor:
    if mixture_mode == "enhancement_only":
        return robust_vector
    if mixture_mode == "suppression_only":
        return suppress_vector
    if mixture_mode == "combo_1_1":
        return robust_vector + suppress_vector
    if mixture_mode == "suppress_heavy_2_1":
        return robust_vector + 2.0 * suppress_vector
    if mixture_mode == "enhance_heavy_1_2":
        return 2.0 * robust_vector + suppress_vector
    raise ValueError(f"Unknown mixture_mode={mixture_mode!r}")


def layer_vector_metadata(vector: torch.Tensor, robust_direction: torch.Tensor) -> dict[str, Any]:
    vector = vector.float()
    robust_direction = robust_direction.float()
    denom = vector.norm() * robust_direction.norm()
    cosine = float((vector @ robust_direction / denom).item()) if denom.item() > 1e-12 else None
    return {
        "vector_norm": float(vector.norm().item()),
        "robust_direction_norm": float(robust_direction.norm().item()),
        "cosine_with_robust_direction": cosine,
    }


def load_direction_records(
    *,
    output_root: Path,
    dataset: str,
    model_tag: str,
    max_direction_pairs_per_group: int,
) -> tuple[list[dict[str, Any]], Path, Path]:
    dev_behavior_root = default_dev_behavior_root(output_root, dataset, model_tag)
    splits_root = default_splits_root(output_root, dataset, model_tag)
    discovery_sources_path = splits_root / "discovery_sources.json"
    discovery_sources = load_discovery_sources(discovery_sources_path)
    q_rows = load_jsonl(dev_behavior_root / "q_group_labels.jsonl")
    selected = [
        row
        for row in q_rows
        if row.get("source_id") in discovery_sources and row.get("q_group") in {"Q1", "Q2"}
    ]
    selected = ordered_group_limited(selected, max_direction_pairs_per_group)
    if not selected:
        raise ValueError("No development discovery Q1/Q2 records selected for robust direction")
    return selected, dev_behavior_root, discovery_sources_path


def compute_robust_directions(
    *,
    model,
    tokenizer,
    dataset: str,
    layers: list[int],
    direction_records: list[dict[str, Any]],
    root: Path,
    device: str,
    progress_every: int,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    dataset_stage = load_dataset_stage(root / "configs/datasets.json", dataset, "dev", root)
    false_views = load_jsonl(dataset_stage.false_suggestion)
    by_pair = pair_lookup(false_views)
    needed_pair_ids = {row["pair_id"] for row in direction_records}
    for pair_id in sorted(needed_pair_ids):
        by_pair[pair_id]["pressure"]["regions"] = locate_regions(tokenizer, by_pair[pair_id]["pressure"])

    directions: dict[int, torch.Tensor] = {}
    layer_counts: dict[str, dict[str, int]] = {}
    zero_target_count = 0
    for layer in layers:
        states: dict[str, list[torch.Tensor]] = {"Q1": [], "Q2": []}
        for index, row in enumerate(direction_records, start=1):
            pressure = by_pair[row["pair_id"]]["pressure"]
            target = pressure["regions"]["pressure_claim"]
            if not target:
                zero_target_count += 1
                continue
            input_ids = torch.tensor([pressure["regions"]["input_ids"]], dtype=torch.long, device=device)
            hidden = capture_hidden(model, input_ids, layer)
            pooled = pooled_claim_hidden(hidden, target)
            if pooled is None:
                continue
            states[row["q_group"]].append(pooled.detach())
            if progress_every > 0 and (index % progress_every == 0 or index == len(direction_records)):
                print(
                    f"direction layer={layer} captured {index}/{len(direction_records)}",
                    flush=True,
                )
        if not states["Q1"] or not states["Q2"]:
            raise ValueError(
                f"Cannot compute robust direction for layer {layer}: "
                f"Q1={len(states['Q1'])}, Q2={len(states['Q2'])}"
            )
        q1_hidden = torch.stack(states["Q1"]).to(device)
        q2_hidden = torch.stack(states["Q2"]).to(device)
        directions[layer] = q2_hidden.mean(dim=0) - q1_hidden.mean(dim=0)
        layer_counts[str(layer)] = {"Q1": len(states["Q1"]), "Q2": len(states["Q2"])}
    return directions, {
        "dataset_stage": dataset_stage.name,
        "direction_record_count": len(direction_records),
        "direction_q_group_counts": dict(Counter(row["q_group"] for row in direction_records)),
        "direction_layer_counts": layer_counts,
        "zero_target_count": zero_target_count,
        "span_policy": "development_direction_norm_uses_candidate_regions; holdout_eval_uses_extracted_span_only",
    }


def build_layer_vectors(
    *,
    selected: dict[str, Any],
    feature_root: Path,
    sae_by_layer: dict[int, dict[str, torch.Tensor]],
    robust_directions: dict[int, torch.Tensor],
    mixture_modes: list[str],
    vector_constructions: list[str],
    random_seed: int,
    dataset: str,
    model_tag: str,
    device: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    a_rows = selected_feature_rows(
        feature_bank_path=feature_root / "feature_bank_A.jsonl",
        selected_by_layer=selected.get("feature_uids_A_by_layer", {}),
        expected_bank="A",
        dataset=dataset,
        model_tag=model_tag,
    )
    r_rows = selected_feature_rows(
        feature_bank_path=feature_root / "feature_bank_R.jsonl",
        selected_by_layer=selected.get("feature_uids_R_by_layer", {}),
        expected_bank="R",
        dataset=dataset,
        model_tag=model_tag,
    )
    requested = expand_mixture_modes(",".join(mixture_modes), str(selected["mixture_mode"]))

    vectors: dict[str, dict[str, Any]] = {}
    metadata: dict[str, Any] = {
        "by_construction": {},
        "selected_feature_counts_by_layer": {},
        "random_seed": random_seed,
    }
    for layer in [int(layer) for layer in selected["layers"]]:
        robust_direction = robust_directions[layer]
        sae = sae_by_layer[layer]
        layer_a_rows = a_rows.get(layer, [])
        layer_r_rows = r_rows.get(layer, [])
        metadata["selected_feature_counts_by_layer"][str(layer)] = {
            "A": len(layer_a_rows),
            "R": len(layer_r_rows),
        }
        for construction in vector_constructions:
            if construction == "full_va":
                suppress = sparse_vector(
                    sae=sae,
                    rows=layer_a_rows,
                    robust_direction=robust_direction,
                    sign=-1.0,
                    device=device,
                )
                robust = sparse_vector(
                    sae=sae,
                    rows=layer_r_rows,
                    robust_direction=robust_direction,
                    sign=+1.0,
                    device=device,
                )
                composer = compose_vector
            elif construction == "no_alignment":
                suppress = raw_sparse_vector(
                    sae=sae,
                    rows=layer_a_rows,
                    sign=-1.0,
                    device=device,
                )
                robust = raw_sparse_vector(
                    sae=sae,
                    rows=layer_r_rows,
                    sign=+1.0,
                    device=device,
                )
                composer = None
            elif construction == "random_feature_alignment":
                suppress = random_feature_vector(
                    sae=sae,
                    rows=layer_a_rows,
                    robust_direction=robust_direction,
                    sign=-1.0,
                    seed=random_seed + layer * 1009 + 17,
                    device=device,
                )
                robust = random_feature_vector(
                    sae=sae,
                    rows=layer_r_rows,
                    robust_direction=robust_direction,
                    sign=+1.0,
                    seed=random_seed + layer * 1009 + 53,
                    device=device,
                )
                composer = compose_vector
            elif construction == "dense_q2_q1_direction":
                suppress = robust_direction.float()
                robust = robust_direction.float()
                composer = compose_vector
            elif construction == "random_dense_direction":
                dense = random_dense_vector(
                    robust_direction=robust_direction,
                    seed=random_seed + layer * 1009 + 97,
                    device=device,
                )
                suppress = dense
                robust = dense
                composer = compose_vector
            else:
                raise ValueError(f"Unknown vector construction={construction!r}")

            for mode in requested:
                run_key = f"{construction}::{mode}"
                if run_key not in vectors:
                    vectors[run_key] = {
                        "construction": construction,
                        "mixture_mode": mode,
                        "layer_vectors": {},
                    }
                if composer is None:
                    vector = compose_raw_vector(
                        robust_vector=robust,
                        suppress_vector=suppress,
                        mixture_mode=mode,
                    )
                else:
                    vector = composer(
                        robust_vector=robust,
                        suppress_vector=suppress,
                        robust_direction=robust_direction,
                        mixture_mode=mode,
                    )
                vectors[run_key]["layer_vectors"][layer] = vector
                metadata["by_construction"].setdefault(construction, {}).setdefault(mode, {})[
                    str(layer)
                ] = layer_vector_metadata(vector, robust_direction)
    return vectors, metadata


def compute_gate_base_payload(
    *,
    model,
    tokenizer,
    example: dict[str, Any],
    selected: dict[str, Any],
    sae_by_layer: dict[int, dict[str, torch.Tensor]],
    device: str,
) -> dict[str, Any]:
    """Compute all gate inputs that do not depend on the intervention mixture.

    The extracted candidate span, peak-zA risk score, and neutral preference
    score are the same for enhancement-only, suppression-only, and combo
    vectors. Caching them once per example is therefore exact, not an
    approximation.
    """
    located = locate_candidate_positions(
        tokenizer,
        example["view"],
        allow_structured_candidate=True,
    )
    target_positions = located["candidate_positions"]
    if not target_positions:
        return {
            "peak_zA": None,
            "neutral_preference_score": None,
            "self_answer_conflict_score": None,
            "target_positions": [],
            "gate_signal_name": "self_answer_conflict_score",
            "gate_score_status": "missing_target_positions",
            "candidate_text": located.get("candidate_text", ""),
            "candidate_text_source": located.get("candidate_text_source"),
            "prompt_extracted_pattern": located.get("prompt_extracted_pattern"),
            "structured_candidate_allowed": located.get("structured_candidate_allowed"),
            "uses_correct_answer_for_gate": False,
        }

    input_ids = torch.tensor([located["input_ids"]], dtype=torch.long, device=device)
    eligible = [idx for idx, (left, right) in enumerate(located["offsets"]) if right > left]
    risk_values: list[float] = []
    feature_uids_a = selected.get("feature_uids_A_by_layer", {})
    for layer in [int(layer) for layer in selected["layers"]]:
        sae = sae_by_layer[layer]
        hidden = capture_hidden(model, input_ids, layer)
        z_tokens = sae_activation(hidden, sae, device)
        a_ids = [int(uid.split(":F", 1)[1]) for uid in feature_uids_a.get(str(layer), [])]
        if a_ids:
            ids = torch.tensor(a_ids, dtype=torch.long, device=device)
            a_scores = z_tokens[:, ids].mean(dim=1).detach()
            za = PHASE4.zscore(a_scores, eligible)
            risk_values.extend(float(za[pos].item()) for pos in target_positions if pos < za.numel())

    neutral = example.get("neutral_view") or example["view"]
    suggested = located["candidate_text"]
    self_answer_conflict_score, gate_info = PHASE4.score_self_answer_conflict(
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
    return {
        "peak_zA": max(risk_values) if risk_values else None,
        "neutral_preference_score": self_answer_conflict_score,
        "self_answer_conflict_score": self_answer_conflict_score,
        "target_positions": target_positions,
        **gate_info,
    }


def layer_deltas_for_positions(
    *,
    layer_vectors: dict[int, torch.Tensor],
    target_positions: list[int],
    intervention_strength: float,
    device: str,
) -> dict[int, torch.Tensor]:
    if not target_positions:
        return {}
    return {
        int(layer): (
            float(intervention_strength)
            * vector.to(device=device, dtype=torch.float32).repeat(len(target_positions), 1)
        )
        for layer, vector in layer_vectors.items()
    }


def run_specs_from_args(args: argparse.Namespace, frozen_tau_a: float, frozen_tau_c: float) -> list[dict[str, Any]]:
    gate_modes = parse_csv(args.gate_modes)
    unknown = sorted(set(gate_modes) - VALID_GATE_MODES)
    if unknown:
        raise ValueError(f"Unknown gate modes: {unknown}")
    tau_values: list[tuple[str, float | None, float | None]] = []
    for scale in parse_floats(args.tau_a_scales):
        tau_values.append((f"scale{scale:g}", float(frozen_tau_a) * scale, scale))
    for value in parse_floats(args.tau_a_values):
        tau_values.append((f"value{value:g}", value, None))
    if not tau_values:
        tau_values.append(("frozen", float(frozen_tau_a), 1.0))

    tau_c_requested = parse_floats(args.tau_c_values)
    tau_c_values: list[tuple[str, float | None]] = []
    if tau_c_requested:
        tau_c_values = [(f"value{value:g}", value) for value in tau_c_requested]
    else:
        tau_c_values = [("frozen", float(frozen_tau_c))]

    specs: list[dict[str, Any]] = []
    for mode in gate_modes:
        if mode == "peak_zA_only":
            for label, value, scale in tau_values:
                specs.append(
                    {
                        "gate_mode": mode,
                        "run_id": f"{mode}_{label}",
                        "tau_A_value": value,
                        "tau_A_scale_from_frozen": scale,
                        "tau_C_value": None,
                        "tau_C_label": None,
                        "oracle_gate_used": False,
                    }
                )
        elif mode == "peak_zA_tau":
            for label, value, scale in tau_values:
                for tau_c_label, tau_c_value in tau_c_values:
                    if tau_c_requested:
                        run_id = f"{mode}_{label}_tauC_{tau_c_label}"
                    else:
                        run_id = f"{mode}_{label}"
                    specs.append(
                        {
                            "gate_mode": mode,
                            "run_id": run_id,
                            "tau_A_value": value,
                            "tau_A_scale_from_frozen": scale,
                            "tau_C_value": tau_c_value,
                            "tau_C_label": tau_c_label,
                            "oracle_gate_used": False,
                        }
                    )
        else:
            specs.append(
                {
                    "gate_mode": mode,
                    "run_id": mode,
                    "tau_A_value": None,
                    "tau_A_scale_from_frozen": None,
                    "tau_C_value": None,
                    "tau_C_label": None,
                    "oracle_gate_used": mode == "oracle_q1",
                }
            )
    return specs


def is_lace_config(config: dict[str, Any]) -> bool:
    return config.get("record_type") in {"lace_config", "lca_sae_vector_aligned_config"}


def apply_vector_config_selection(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Normalize a formal vector config into the frozen base-config shape.

    When ``--config`` points to a formal LACE config, the command line must not
    reselect mixture, thresholds, or strength on holdout.
    This function freezes the eval args to the dev-calibrated selection.
    """
    if not is_lace_config(config):
        return config
    issues = []
    for key, expected in [
        ("run_mode", "formal"),
        ("is_formal_calibration", True),
        ("frozen_config", True),
        ("holdout_used_for_selection", False),
        ("uses_gold_span_fields", False),
        ("span_mode", "extracted"),
        ("uses_correct_answer_for_gate", False),
    ]:
        if config.get(key) != expected:
            issues.append(f"vector_config_guard_failed:{key}={config.get(key)!r}:expected={expected!r}")
    selected_vector = config.get("selected_method_config") or config.get("selected_vector_config")
    base_selected = config.get("base_selected_config")
    if not isinstance(selected_vector, dict):
        issues.append("missing_selected_method_config")
    if not isinstance(base_selected, dict):
        issues.append("missing_base_selected_config")
    if isinstance(selected_vector, dict):
        if selected_vector.get("vector_construction", "full_va") != "full_va":
            issues.append("final_config_vector_construction_is_not_full_va")
        if selected_vector.get("gate_mode") != "peak_zA_tau":
            issues.append("final_config_gate_mode_is_not_peak_zA_tau")
    if issues:
        raise ValueError(f"LACE config guard failed: {issues}")
    args.mixture_modes = str(selected_vector["mixture_mode"])
    if not getattr(args, "allow_diagnostic_gate_overrides", False):
        args.gate_modes = "peak_zA_tau"
        args.tau_a_scales = ""
        args.tau_a_values = str(float(selected_vector["tau_A_value"]))
        args.tau_c_values = str(float(selected_vector.get("tau_C", 0.0)))
        args.frozen_vector_config_tau_c_injected = True
    args.intervention_strength = float(selected_vector["intervention_strength"])
    args.random_seed = int(config.get("random_seed", args.random_seed))
    return {
        "selected_config": dict(base_selected),
        "vector_config_path": str(args.config),
        "selected_method_config": dict(selected_vector),
    }


def is_diagnostic_variant_eval(args: argparse.Namespace) -> bool:
    vector_constructions = parse_vector_constructions(args.vector_constructions)
    tau_c_values_are_user_override = bool(parse_floats(args.tau_c_values)) and not bool(
        getattr(args, "frozen_vector_config_tau_c_injected", False)
    )
    return (
        vector_constructions != ["full_va"]
        or bool(set(parse_csv(args.gate_modes)) - {"peak_zA_tau"})
        or bool(parse_floats(args.tau_a_scales))
        or tau_c_values_are_user_override
        or bool(getattr(args, "allow_diagnostic_gate_overrides", False))
    )


def should_trigger(
    *,
    spec: dict[str, Any],
    example: dict[str, Any],
    peak_za: float | None,
    neutral_preference_score: float | None,
    tau_c: float,
    has_delta: bool,
) -> bool:
    if not has_delta:
        return False
    gate_mode = spec["gate_mode"]
    if gate_mode == "no_gate_false_only":
        return example["condition"] == "false_suggestion"
    if gate_mode == "no_gate_all_extracted":
        return True
    if gate_mode == "oracle_q1":
        return example["condition"] == "false_suggestion" and example.get("q_group") == "Q1"
    if gate_mode == "peak_zA_only":
        tau_a = spec["tau_A_value"]
        return (
            peak_za is not None
            and tau_a is not None
            and peak_za >= float(tau_a)
        )
    if gate_mode == "peak_zA_tau":
        tau_a = spec["tau_A_value"]
        tau_c_value = spec.get("tau_C_value")
        if tau_c_value is None:
            tau_c_value = tau_c
        return strict_gate_decision(
            peak_za=peak_za,
            self_answer_conflict_score=neutral_preference_score,
            tau_a=float(tau_a) if tau_a is not None else None,
            tau_c=float(tau_c_value),
            has_intervention=has_delta,
        )
    raise ValueError(f"Unknown gate mode={gate_mode!r}")


def evaluate_lace(
    *,
    args: argparse.Namespace,
    root: Path,
    config: dict[str, Any],
    model_spec: Any,
    examples: list[dict[str, Any]],
    feature_root: Path,
    output_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = dict(config["selected_config"])
    selected["tau_A_value"] = config.get("tau_A_value", selected.get("tau_A_value"))
    selected["tau_C"] = config.get("tau_C", selected.get("tau_C", 0.0))
    selected_layers = [int(layer) for layer in selected["layers"]]
    mixture_modes = expand_mixture_modes(args.mixture_modes, str(selected.get("mixture_mode")))
    vector_constructions = parse_vector_constructions(args.vector_constructions)

    model_path, sae_path = PHASE4.resolve_model_and_sae_paths(args=args, model_spec=model_spec, root=root)
    model, tokenizer = PHASE4.load_model_and_tokenizer(model_path, args.device)
    sae_by_layer = PHASE4.load_sae_layers(
        sae_path,
        selected_layers,
        args.device,
        model_spec=model_spec,
    )

    direction_records, dev_behavior_root, discovery_sources_path = load_direction_records(
        output_root=output_root,
        dataset=args.dataset,
        model_tag=args.model_tag,
        max_direction_pairs_per_group=args.max_direction_pairs_per_group,
    )
    robust_directions, direction_meta = compute_robust_directions(
        model=model,
        tokenizer=tokenizer,
        dataset=args.dataset,
        layers=selected_layers,
        direction_records=direction_records,
        root=root,
        device=args.device,
        progress_every=args.progress_every_direction,
    )
    layer_vectors_by_mode, vector_meta = build_layer_vectors(
        selected=selected,
        feature_root=feature_root,
        sae_by_layer=sae_by_layer,
        robust_directions=robust_directions,
        mixture_modes=mixture_modes,
        vector_constructions=vector_constructions,
        random_seed=args.random_seed,
        dataset=args.dataset,
        model_tag=args.model_tag,
        device=args.device,
    )
    tau_c = float(selected.get("tau_C") or 0.0)
    run_specs = run_specs_from_args(args, float(selected["tau_A_value"]), tau_c)

    gate_cache: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for index, example in enumerate(examples, start=1):
        gate_payload = compute_gate_base_payload(
            model=model,
            tokenizer=tokenizer,
            example=example,
            selected=selected,
            sae_by_layer=sae_by_layer,
            device=args.device,
        )
        gate_cache.append((example, gate_payload))
        if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(examples)):
            print(
                f"cached gate inputs holdout rows {index}/{len(examples)}",
                flush=True,
            )

    records: list[dict[str, Any]] = []
    for vector_run in layer_vectors_by_mode.values():
        construction = vector_run["construction"]
        mixture_mode = vector_run["mixture_mode"]
        layer_vectors = vector_run["layer_vectors"]
        for index, (example, gate_payload) in enumerate(gate_cache, start=1):
            layer_deltas = layer_deltas_for_positions(
                layer_vectors=layer_vectors,
                target_positions=gate_payload["target_positions"],
                intervention_strength=args.intervention_strength,
                device=args.device,
            )
            has_delta = bool(gate_payload["target_positions"]) and bool(layer_deltas)
            triggered_by_spec = {
                spec["run_id"]: should_trigger(
                    spec=spec,
                    example=example,
                    peak_za=gate_payload["peak_zA"],
                    neutral_preference_score=gate_payload["neutral_preference_score"],
                    tau_c=tau_c,
                    has_delta=has_delta,
                )
                for spec in run_specs
            }
            intervened_scores: dict[str, float] | None = None
            if any(triggered_by_spec.values()):
                intervened_scores = PHASE4.candidate_scores_local(
                    model,
                    tokenizer,
                    example["view"],
                    args.device,
                    layer_deltas=layer_deltas,
                    target_positions=gate_payload["target_positions"],
                )

            for spec in run_specs:
                triggered = triggered_by_spec[spec["run_id"]]
                after = intervened_scores if triggered and intervened_scores is not None else {
                    "logp_correct": example["baseline_logp_correct"],
                    "logp_wrong": example["baseline_logp_wrong"],
                    "margin": example["baseline_margin"],
                }
                records.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "record_type": "lace_intervention_record",
                        "stage": "holdout",
                        "dataset": args.dataset,
                        "model_tag": args.model_tag,
                        "source_id": example["source_id"],
                        "template_id": example["template_id"],
                        "pair_id": example.get("pair_id"),
                        "condition": example["condition"],
                        "q_group": example.get("q_group"),
                        "relation": example.get("relation"),
                        "subject": example.get("subject"),
                        "config_path": str(args.config),
                        "selected_band": selected.get("band_name"),
                        "selected_layers": selected.get("layers"),
                        "top_k_policy": selected.get("top_k_policy"),
                        "top_k_A_global": selected.get("top_k_A_global"),
                        "top_k_R_global": selected.get("top_k_R_global"),
                        "selected_config_mixture_mode": selected.get("mixture_mode"),
                        "vector_construction": construction,
                        "mixture_mode": mixture_mode,
                        "intervention_construction": construction,
                        "intervention_strength": args.intervention_strength,
                        "gate_mode": spec["gate_mode"],
                        "run_id": f"{construction}__{mixture_mode}__{spec['run_id']}",
                        "tau_A_value": spec["tau_A_value"],
                        "tau_A_scale_from_frozen": spec["tau_A_scale_from_frozen"],
                        "frozen_tau_A_value": selected["tau_A_value"],
                        "tau_C": spec.get("tau_C_value") if spec.get("tau_C_value") is not None else tau_c,
                        "tau_C_label": spec.get("tau_C_label"),
                        "frozen_tau_C": tau_c,
                        "oracle_gate_used": spec["oracle_gate_used"],
                        "peak_zA": gate_payload["peak_zA"],
                        "self_answer_conflict_score": gate_payload["self_answer_conflict_score"],
                        "neutral_preference_score": gate_payload["neutral_preference_score"],
                        "gate_signal_name": gate_payload.get("gate_signal_name"),
                        "gate_score_status": gate_payload.get("gate_score_status"),
                        "self_answer": gate_payload.get("self_answer"),
                        "gate_candidate_text": gate_payload.get("candidate_text"),
                        "candidate_text_source": gate_payload.get("candidate_text_source"),
                        "prompt_extracted_pattern": gate_payload.get("prompt_extracted_pattern"),
                        "structured_candidate_allowed_for_locator": gate_payload.get("structured_candidate_allowed"),
                        "uses_correct_answer_for_gate": gate_payload.get("uses_correct_answer_for_gate", False),
                        "triggered": triggered,
                        "target_positions": gate_payload["target_positions"],
                        "target_token_count": len(gate_payload["target_positions"]),
                        "baseline_logp_correct": example["baseline_logp_correct"],
                        "baseline_logp_wrong": example["baseline_logp_wrong"],
                        "baseline_margin": example["baseline_margin"],
                        "baseline_is_correct": example["baseline_is_correct"],
                        "baseline_strict_correct": example["baseline_strict_correct"],
                        "after_logp_correct": after["logp_correct"],
                        "after_logp_wrong": after["logp_wrong"],
                        "after_margin": after["margin"],
                        "after_is_correct": after["margin"] > 0.0,
                        "delta_margin": after["margin"] - example["baseline_margin"],
                        "uses_gold_span_fields": False,
                        "span_mode": "extracted",
                    }
                )
            if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(examples)):
                print(
                    f"scored construction={construction} mixture={mixture_mode} holdout rows {index}/{len(examples)}",
                    flush=True,
                )

    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    meta = {
        "model_path": str(model_path),
        "sae_path": str(sae_path),
        "sae_compatibility": sae_compatibility_metadata(model_spec),
        "device": args.device,
        "dev_behavior_root": str(dev_behavior_root),
        "discovery_sources_path": str(discovery_sources_path),
        "direction_meta": direction_meta,
        "vector_meta": vector_meta,
        "run_specs": run_specs,
        "requested_mixture_modes": parse_csv(args.mixture_modes),
        "expanded_mixture_modes": mixture_modes,
        "vector_constructions": vector_constructions,
        "gate_cache_reused_across_mixtures": True,
        "gate_cache_count": len(gate_cache),
    }
    return records, meta


def summarize_by_run(records: list[dict[str, Any]], strict_threshold: float) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    run_ids = sorted({row["run_id"] for row in records})
    for run_id in run_ids:
        rows = [row for row in records if row["run_id"] == run_id]
        metrics = PHASE5.summarize_records(rows, strict_threshold)
        first = rows[0]
        output.append(
            {
                "run_id": run_id,
                "vector_construction": first.get("vector_construction"),
                "intervention_construction": first.get("intervention_construction"),
                "mixture_mode": first["mixture_mode"],
                "gate_mode": first["gate_mode"],
                "tau_A_value": first["tau_A_value"],
                "tau_A_scale_from_frozen": first["tau_A_scale_from_frozen"],
                "tau_C": first.get("tau_C"),
                "tau_C_label": first.get("tau_C_label"),
                "intervention_strength": first["intervention_strength"],
                "oracle_gate_used": first["oracle_gate_used"],
                **metrics,
            }
        )
    output.sort(
        key=lambda row: (
            row.get("vector_construction") or "",
            row["mixture_mode"],
            row["gate_mode"],
            row["tau_A_value"] is None,
            row["tau_A_value"] if row["tau_A_value"] is not None else 999999.0,
        )
    )
    return output


def build_audit(
    *,
    args: argparse.Namespace,
    config_issues: list[str],
    behavior_rows: list[dict[str, Any]],
    q_rows: list[dict[str, Any]],
    mixed_views: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    issues = list(config_issues)
    warnings: list[str] = []
    if args.behavior_root.name != "holdout_behavior":
        issues.append("behavior_root_is_not_holdout_behavior")
    if any(row.get("dataset_stage") != "holdout" for row in behavior_rows):
        issues.append("behavior_rows_not_holdout_stage")
    if any(row.get("dataset_stage") != "holdout" for row in q_rows):
        issues.append("q_group_rows_not_holdout_stage")
    if any(row.get("uses_gold_span_fields") for row in records):
        issues.append("records_used_gold_span_fields")
    if any(row.get("span_mode") != "extracted" for row in records):
        issues.append("records_not_extracted_span_mode")
    if any(row.get("uses_correct_answer_for_gate") for row in records):
        issues.append("records_used_correct_answer_for_gate")
    if not records and not args.dry_run:
        issues.append("empty_eval_records")
    if any(row.get("oracle_gate_used") for row in records):
        warnings.append("oracle_q1_gate_is_diagnostic_upper_bound_not_final_method")
    vector_constructions = parse_vector_constructions(args.vector_constructions)
    tau_c_values_are_user_override = bool(parse_floats(args.tau_c_values)) and not bool(
        getattr(args, "frozen_vector_config_tau_c_injected", False)
    )
    diagnostic_variant_eval = is_diagnostic_variant_eval(args)
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": (
            "lace_holdout_audit"
            if bool(getattr(args, "vector_config_mode", False)) and not diagnostic_variant_eval
            else "internal_variant_audit"
        ),
        "status": "fail" if issues else "pass",
        "issues": issues,
        "warnings": warnings,
        "checks": {
            "stage": "holdout",
            "read_holdout_data": True,
            "holdout_used_for_selection": False,
            "holdout_eval_only": True,
            "internal_variant_eval": (
                diagnostic_variant_eval or not bool(getattr(args, "vector_config_mode", False))
            ),
            "final_lace_config_eval": bool(getattr(args, "vector_config_mode", False))
            and not diagnostic_variant_eval,
            "dry_run": bool(args.dry_run),
            "config_path": str(args.config),
            "behavior_root": str(args.behavior_root),
            "mixed_eval_path": str(args.mixed_eval),
            "uses_gold_span_fields": False,
            "span_mode": "extracted",
            "uses_correct_answer_for_gate": False,
            "structured_candidate_allowed_for_locator": True,
            "candidate_extractor": extractor_spec(),
            "vector_constructions": vector_constructions,
            "allow_diagnostic_gate_overrides": bool(getattr(args, "allow_diagnostic_gate_overrides", False)),
            "tau_c_values_effective": parse_floats(args.tau_c_values),
            "tau_c_values_user_override": tau_c_values_are_user_override,
            "frozen_vector_config_tau_c_injected": bool(getattr(args, "frozen_vector_config_tau_c_injected", False)),
            "dev_direction_uses_holdout": False,
            "behavior_row_count": len(behavior_rows),
            "q_group_row_count": len(q_rows),
            "mixed_eval_view_count": len(mixed_views),
            "record_count": len(records),
            "limited_eval": (
                args.dry_run
                or args.max_sources > 0
                or args.max_pairs_per_group > 0
                or args.max_rows_per_condition > 0
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["popqa", "counterfact"], required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--root", type=Path, default=repo_root())
    parser.add_argument("--models-config", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--behavior-root", type=Path)
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument("--mixed-eval", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--sae-path", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mixture-modes", default="selected")
    parser.add_argument("--vector-constructions", default="full_va")
    parser.add_argument("--gate-modes", default="peak_zA_tau,no_gate_false_only,oracle_q1")
    parser.add_argument("--tau-a-scales", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--tau-a-values", default="")
    parser.add_argument("--tau-c-values", default="")
    parser.add_argument("--intervention-strength", type=float, default=3.0)
    parser.add_argument("--max-direction-pairs-per-group", type=int, default=0)
    parser.add_argument("--max-sources", type=int, default=0)
    parser.add_argument("--max-pairs-per-group", type=int, default=0)
    parser.add_argument("--max-rows-per-condition", type=int, default=0)
    parser.add_argument("--strict-damage-threshold", type=float, default=0.5)
    parser.add_argument("--random-seed", type=int, default=13)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--progress-every-direction", type=int, default=40)
    parser.add_argument(
        "--allow-diagnostic-gate-overrides",
        action="store_true",
        help=(
            "Allow supplementary diagnostics to override frozen gate modes/tau values "
            "when evaluating a formal vector-aligned config. This must not be used "
            "for final method selection."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_root = resolve_cli_path(args.output_root, root) or default_output_root(root)
    args.config = resolve_cli_path(args.config, root) or default_config_path(output_root, args.dataset, args.model_tag)
    args.behavior_root = resolve_cli_path(args.behavior_root, root) or default_behavior_root(output_root, args.dataset, args.model_tag)
    args.feature_root = resolve_cli_path(args.feature_root, root) or default_feature_banks_root(output_root, args.dataset, args.model_tag)
    args.output_dir = resolve_cli_path(args.output_dir, root) or default_output_dir(output_root, args.dataset, args.model_tag)
    dataset_stage = load_dataset_stage(root / "configs/datasets.json", args.dataset, "holdout", root)
    args.mixed_eval = resolve_cli_path(args.mixed_eval, root) or dataset_stage.mixed_eval
    models_config = resolve_cli_path(args.models_config, root) or default_models_config(root)
    model_spec = load_model_spec(models_config, args.model_tag, root, allow_placeholder=True)

    config = load_json(args.config)
    vector_config_mode = is_lace_config(config)
    args.vector_config_mode = vector_config_mode
    if vector_config_mode:
        config = apply_vector_config_selection(config, args)
        config_issues = []
    else:
        config_issues = PHASE5.config_guard(config, args.config, args.dataset, args.model_tag)
        if config_issues:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            audit = build_audit(
                args=args,
                config_issues=config_issues,
                behavior_rows=[],
                q_rows=[],
                mixed_views=[],
                records=[],
            )
            write_json(args.output_dir / "audit.json", audit)
            raise SystemExit(f"Frozen config guard failed: {config_issues}")

    behavior_rows = load_jsonl(args.behavior_root / "behavior_records.jsonl")
    q_rows = load_jsonl(args.behavior_root / "q_group_labels.jsonl")
    mixed_views = load_jsonl(args.mixed_eval)
    examples = PHASE5.select_holdout_examples(
        behavior_rows=behavior_rows,
        q_rows=q_rows,
        mixed_views=mixed_views,
        max_sources=args.max_sources,
        max_pairs_per_group=args.max_pairs_per_group,
        max_rows_per_condition=args.max_rows_per_condition,
    )
    if not examples:
        raise ValueError("No holdout examples selected")

    if args.dry_run:
        records: list[dict[str, Any]] = []
        scoring_meta = {
            "dry_run": True,
            "model_loaded": False,
            "example_count": len(examples),
            "run_specs": run_specs_from_args(
                args,
                float(config["selected_config"]["tau_A_value"]),
                float(config["selected_config"].get("tau_C") or 0.0),
            ),
            "note": "Dry-run validates config, feature-bank paths, and holdout example selection only.",
        }
        run_summaries: list[dict[str, Any]] = []
    else:
        records, scoring_meta = evaluate_lace(
            args=args,
            root=root,
            config=config,
            model_spec=model_spec,
            examples=examples,
            feature_root=args.feature_root,
            output_root=output_root,
        )
        run_summaries = summarize_by_run(records, args.strict_damage_threshold)

    audit = build_audit(
        args=args,
        config_issues=[],
        behavior_rows=behavior_rows,
        q_rows=q_rows,
        mixed_views=mixed_views,
        records=records,
    )
    diagnostic_variant_eval = is_diagnostic_variant_eval(args)
    tau_c_values_are_user_override = bool(parse_floats(args.tau_c_values)) and not bool(
        getattr(args, "frozen_vector_config_tau_c_injected", False)
    )
    selected = config["selected_config"]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "lace_holdout_summary" if vector_config_mode else "internal_variant_summary",
        "stage": "holdout",
        "dataset": args.dataset,
        "model_tag": args.model_tag,
        "dataset_name": dataset_stage.name,
        "method": "LACE" if vector_config_mode and not diagnostic_variant_eval else "internal_variant",
        "lace_config_mode": vector_config_mode,
        "dry_run": bool(args.dry_run),
        "config_path": str(args.config),
        "output_dir": str(args.output_dir),
        "config_guard_status": "pass",
        "audit_status": audit["status"],
        "holdout_used_for_selection": False,
        "internal_variant_eval": diagnostic_variant_eval or not vector_config_mode,
        "final_lace_config_eval": bool(vector_config_mode) and not diagnostic_variant_eval,
        "uses_gold_span_fields": False,
        "span_mode": "extracted",
        "uses_correct_answer_for_gate": False,
        "structured_candidate_allowed_for_locator": True,
        "selected_config": {
            "band_name": selected.get("band_name"),
            "layers": selected.get("layers"),
            "top_k_policy": selected.get("top_k_policy"),
            "top_k_A_global": selected.get("top_k_A_global"),
            "top_k_R_global": selected.get("top_k_R_global"),
            "mixture_mode": selected.get("mixture_mode"),
            "tau_A_value": config.get("tau_A_value", selected.get("tau_A_value")),
            "tau_C": config.get("tau_C", selected.get("tau_C")),
        },
        "selected_method_config": config.get("selected_method_config"),
        "method_settings": {
            "mixture_modes": parse_csv(args.mixture_modes),
            "expanded_mixture_modes": expand_mixture_modes(args.mixture_modes, str(selected.get("mixture_mode"))),
            "vector_constructions": parse_vector_constructions(args.vector_constructions),
            "gate_modes": parse_csv(args.gate_modes),
            "tau_a_scales": parse_floats(args.tau_a_scales),
            "tau_a_values": parse_floats(args.tau_a_values),
            "tau_c_values_effective": parse_floats(args.tau_c_values),
            "tau_c_values_user_override": tau_c_values_are_user_override,
            "frozen_vector_config_tau_c_injected": bool(getattr(args, "frozen_vector_config_tau_c_injected", False)),
            "intervention_strength": args.intervention_strength,
            "max_direction_pairs_per_group": args.max_direction_pairs_per_group,
            "random_seed": args.random_seed,
            "allow_diagnostic_gate_overrides": bool(args.allow_diagnostic_gate_overrides),
        },
        "run_summaries": run_summaries,
        "scoring_meta": scoring_meta,
        "limits": {
            "max_sources": args.max_sources,
            "max_pairs_per_group": args.max_pairs_per_group,
            "max_rows_per_condition": args.max_rows_per_condition,
        },
        "outputs": {
            "records": str(args.output_dir / "records.jsonl"),
            "summary": str(args.output_dir / "summary.json"),
            "audit": str(args.output_dir / "audit.json"),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "records.jsonl", records)
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "audit.json", audit)
    print(f"Wrote {args.output_dir / 'records.jsonl'}")
    print(f"Wrote {args.output_dir / 'summary.json'}")
    print(f"Wrote {args.output_dir / 'audit.json'}")
    if audit["status"] != "pass":
        raise SystemExit(f"LACE holdout audit failed: {audit['issues']}")


if __name__ == "__main__":
    main()
