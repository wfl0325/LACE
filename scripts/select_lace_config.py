#!/usr/bin/env python3
"""Select the final frozen LACE configuration on development calibration only.

This script calibrates the final residual-norm-matched intervention:

1. read feature banks and the frozen base config;
2. compute mechanism-aligned A/R vectors using the development discovery split;
3. score mixture/tau/strength candidates only on the development calibration
   split;
4. write the frozen ``lace_config.json`` used by final holdout evaluation.

It must not read holdout data.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.candidate_extractor import extractor_spec  # noqa: E402
from lib.dataset_registry import load_dataset_stage  # noqa: E402
from lib.io_utils import load_json, load_jsonl, repo_root, resolve_path, write_json, write_jsonl  # noqa: E402
from lib.model_registry import load_model_spec, sae_compatibility_metadata  # noqa: E402


SCHEMA_VERSION = "lace_config_selection_v1"
SELECTION_POLICY = "maximize_q1_correction_subject_to_damage_constraints"


def load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VEC = load_module("lace_final_method_runtime", SCRIPT_DIR / "internal" / "lace_runtime.py")
PHASE4 = VEC.PHASE4
PHASE5 = VEC.PHASE5


def default_models_config(root: Path) -> Path:
    local = root / "configs/models.local.json"
    if local.exists():
        return local
    return root / "configs/models.example.json"


def default_output_root(root: Path) -> Path:
    return root / "outputs"


def default_base_config_path(output_root: Path, dataset: str, model_tag: str) -> Path:
    return output_root / dataset / model_tag / "calibration" / "lace_base_config.json"


def default_behavior_root(output_root: Path, dataset: str, model_tag: str) -> Path:
    return output_root / dataset / model_tag / "dev_behavior"


def default_splits_root(output_root: Path, dataset: str, model_tag: str) -> Path:
    return output_root / dataset / model_tag / "dev_splits"


def default_feature_root(output_root: Path, dataset: str, model_tag: str) -> Path:
    return output_root / dataset / model_tag / "feature_banks"


def default_output_dir(output_root: Path, dataset: str, model_tag: str) -> Path:
    return output_root / dataset / model_tag / "lace_config"


def output_config_name(args: argparse.Namespace) -> str:
    if args.dry_run:
        return "lace_config.dry_run.json"
    if args.run_mode == "smoke":
        return "lace_config.smoke.json"
    return "lace_config.json"


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


def default_search_config(root: Path) -> Path:
    return root / "configs/va_search_space.json"


def resolve_search_profile(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    args.search_config = resolve_cli_path(args.search_config, root) or default_search_config(root)
    payload = load_json(args.search_config)
    profiles = payload.get("profiles", {})
    if args.search_profile not in profiles:
        raise ValueError(
            f"Unknown VA search profile {args.search_profile!r}; "
            f"available={sorted(profiles)}"
        )
    profile = dict(profiles[args.search_profile])
    field_map = {
        "mixture_modes": "mixture_modes",
        "vector_constructions": "vector_constructions",
        "gate_modes": "gate_modes",
        "tau_a_scales": "tau_a_scales",
        "tau_a_values": "tau_a_values",
        "tau_c_values": "tau_c_values",
        "intervention_strengths": "intervention_strengths",
    }
    for arg_name, profile_name in field_map.items():
        if getattr(args, arg_name) is None:
            setattr(args, arg_name, str(profile.get(profile_name, "")))
    args.require_positive_tau_a = bool(profile.get("require_positive_tau_a", False))
    args.require_positive_tau_c = bool(profile.get("require_positive_tau_c", False))
    args.require_all_gate_boundary_regimes = bool(
        profile.get("require_all_gate_boundary_regimes", False)
    )
    return profile


def build_search_space_meta(
    args: argparse.Namespace,
    selected: dict[str, Any],
) -> dict[str, Any]:
    mixture_modes = VEC.expand_mixture_modes(
        args.mixture_modes,
        str(selected.get("mixture_mode")),
    )
    vector_constructions = VEC.parse_vector_constructions(args.vector_constructions)
    strengths = parse_floats(args.intervention_strengths)
    tau_c = float(selected.get("tau_C") or 0.0)
    gate_specs = [
        spec
        for spec in VEC.run_specs_from_args(
            args,
            float(selected["tau_A_value"]),
            tau_c,
        )
        if spec["gate_mode"] == "peak_zA_tau"
    ]
    if not strengths:
        raise ValueError("--intervention-strengths must not be empty")
    if not gate_specs:
        raise ValueError("No peak_zA_tau gate candidates were generated")

    effective_tau_a = sorted({float(spec["tau_A_value"]) for spec in gate_specs})
    effective_tau_c = sorted(
        {
            float(spec["tau_C_value"] if spec.get("tau_C_value") is not None else tau_c)
            for spec in gate_specs
        }
    )
    has_zero_tau_a = any(abs(value) <= 1e-12 for value in effective_tau_a)
    has_positive_tau_a = any(value > 0.0 for value in effective_tau_a)
    has_zero_tau_c = any(abs(value) <= 1e-12 for value in effective_tau_c)
    has_positive_tau_c = any(value > 0.0 for value in effective_tau_c)
    gate_boundary_regimes = {
        "zero_A_zero_C": has_zero_tau_a and has_zero_tau_c,
        "positive_A_zero_C": has_positive_tau_a and has_zero_tau_c,
        "zero_A_positive_C": has_zero_tau_a and has_positive_tau_c,
        "positive_A_positive_C": has_positive_tau_a and has_positive_tau_c,
    }
    gate_boundary_regimes_complete = all(gate_boundary_regimes.values())
    if args.require_positive_tau_a and any(value <= 0.0 for value in effective_tau_a):
        raise ValueError(
            f"Search profile {args.search_profile!r} requires tau_A > 0; "
            f"got {effective_tau_a}"
        )
    if args.require_positive_tau_c and any(value <= 0.0 for value in effective_tau_c):
        raise ValueError(
            f"Search profile {args.search_profile!r} requires tau_C > 0; "
            f"got {effective_tau_c}"
        )
    if args.require_all_gate_boundary_regimes and not gate_boundary_regimes_complete:
        missing = sorted(
            name for name, included in gate_boundary_regimes.items() if not included
        )
        raise ValueError(
            f"Search profile {args.search_profile!r} requires all zero/positive "
            f"gate boundary regimes; missing={missing}"
        )

    return {
        "profile_name": args.search_profile,
        "search_config_path": str(args.search_config),
        "selection_policy": SELECTION_POLICY,
        "mixture_modes": mixture_modes,
        "vector_constructions": vector_constructions,
        "gate_modes": parse_csv(args.gate_modes),
        "tau_a_scales": parse_floats(args.tau_a_scales),
        "tau_a_values": parse_floats(args.tau_a_values),
        "tau_c_values": parse_floats(args.tau_c_values),
        "effective_tau_a_values": effective_tau_a,
        "effective_tau_c_values": effective_tau_c,
        "intervention_strengths": strengths,
        "gate_spec_count": len(gate_specs),
        "candidate_count": (
            len(mixture_modes)
            * len(vector_constructions)
            * len(strengths)
            * len(gate_specs)
        ),
        "require_positive_tau_a": bool(args.require_positive_tau_a),
        "require_positive_tau_c": bool(args.require_positive_tau_c),
        "require_all_gate_boundary_regimes": bool(
            args.require_all_gate_boundary_regimes
        ),
        "gate_boundary_regimes": gate_boundary_regimes,
        "gate_boundary_regimes_complete": gate_boundary_regimes_complete,
        "intervention_forward_reused_across_gate_specs": True,
        "random_seed": args.random_seed,
    }


def behavior_by_key(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (row["source_id"], row["template_id"], row["condition"]): row
        for row in rows
    }


def ordered_limited_sources(rows: list[dict[str, Any]], limit: int) -> set[str] | None:
    if limit <= 0:
        return None
    output: list[str] = []
    seen = set()
    for row in rows:
        source_id = row["source_id"]
        if source_id in seen:
            continue
        seen.add(source_id)
        output.append(source_id)
        if len(output) >= limit:
            break
    return set(output)


def select_dev_calibration_examples(
    *,
    q_rows: list[dict[str, Any]],
    behavior_rows: list[dict[str, Any]],
    pressure_by_pair: dict[str, dict[str, Any]],
    side_by_key: dict[tuple[str, str, str], dict[str, Any]],
    max_calibration_sources: int,
    max_pairs_per_group: int,
    max_side_rows_per_condition: int,
) -> list[dict[str, Any]]:
    active_sources = ordered_limited_sources(q_rows, max_calibration_sources)
    behavior_map = behavior_by_key(behavior_rows)
    examples: list[dict[str, Any]] = []
    group_counts: Counter[str] = Counter()

    for row in q_rows:
        source_id = row["source_id"]
        if active_sources is not None and source_id not in active_sources:
            continue
        group = row.get("q_group")
        if group not in {"Q1", "Q2"}:
            continue
        if max_pairs_per_group > 0 and group_counts[group] >= max_pairs_per_group:
            continue
        view = pressure_by_pair.get(row["pair_id"])
        behavior = behavior_map.get((source_id, row["template_id"], "false_suggestion"))
        if not view or not behavior:
            continue
        examples.append(
            {
                "record_type": "false_suggestion",
                "condition": "false_suggestion",
                "q_group": group,
                "source_id": source_id,
                "template_id": row["template_id"],
                "pair_id": row["pair_id"],
                "view": view,
                "neutral_view": pressure_by_pair.get(f"{row['pair_id']}::control"),
                "baseline_margin": float(behavior["margin"]),
                "baseline_logp_correct": float(behavior["logp_correct"]),
                "baseline_logp_wrong": float(behavior["logp_wrong"]),
                "baseline_is_correct": bool(behavior.get("is_correct")),
                "baseline_strict_correct": bool(behavior.get("strict_correct")),
                "relation": behavior.get("relation"),
                "subject": behavior.get("subject"),
            }
        )
        group_counts[group] += 1

    side_counts: Counter[str] = Counter()
    side_condition_map = {"true_suggestion": "true_suggestion", "clean_eval": "clean"}
    for behavior in behavior_rows:
        condition = behavior.get("condition")
        if condition not in side_condition_map:
            continue
        source_id = behavior["source_id"]
        if active_sources is not None and source_id not in active_sources:
            continue
        if max_side_rows_per_condition > 0 and side_counts[condition] >= max_side_rows_per_condition:
            continue
        trigger_condition = side_condition_map[condition]
        view = side_by_key.get((source_id, behavior["template_id"], trigger_condition))
        if not view:
            continue
        examples.append(
            {
                "record_type": "side_effect",
                "condition": condition,
                "q_group": None,
                "source_id": source_id,
                "template_id": behavior["template_id"],
                "pair_id": behavior.get("pair_id"),
                "view": view,
                "neutral_view": side_by_key.get((source_id, behavior["template_id"], "clean")),
                "baseline_margin": float(behavior["margin"]),
                "baseline_logp_correct": float(behavior["logp_correct"]),
                "baseline_logp_wrong": float(behavior["logp_wrong"]),
                "baseline_is_correct": bool(behavior.get("is_correct")),
                "baseline_strict_correct": bool(behavior.get("strict_correct")),
                "relation": behavior.get("relation"),
                "subject": behavior.get("subject"),
            }
        )
        side_counts[condition] += 1
    return examples


def score_candidates(
    *,
    args: argparse.Namespace,
    root: Path,
    base_config: dict[str, Any],
    model_spec: Any,
    examples: list[dict[str, Any]],
    output_root: Path,
    feature_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = dict(base_config["selected_config"])
    selected["tau_A_value"] = base_config.get("tau_A_value", selected.get("tau_A_value"))
    selected["tau_C"] = base_config.get("tau_C", selected.get("tau_C", 0.0))
    selected_layers = [int(layer) for layer in selected["layers"]]
    mixture_modes = VEC.expand_mixture_modes(args.mixture_modes, str(selected.get("mixture_mode")))
    vector_constructions = VEC.parse_vector_constructions(args.vector_constructions)
    strengths = parse_floats(args.intervention_strengths)
    if not strengths:
        raise ValueError("--intervention-strengths must not be empty")

    model_path, sae_path = PHASE4.resolve_model_and_sae_paths(args=args, model_spec=model_spec, root=root)
    model, tokenizer = PHASE4.load_model_and_tokenizer(model_path, args.device)
    sae_by_layer = PHASE4.load_sae_layers(
        sae_path,
        selected_layers,
        args.device,
        model_spec=model_spec,
    )

    direction_records, dev_behavior_root, discovery_sources_path = VEC.load_direction_records(
        output_root=output_root,
        dataset=args.dataset,
        model_tag=args.model_tag,
        max_direction_pairs_per_group=args.max_direction_pairs_per_group,
    )
    robust_directions, direction_meta = VEC.compute_robust_directions(
        model=model,
        tokenizer=tokenizer,
        dataset=args.dataset,
        layers=selected_layers,
        direction_records=direction_records,
        root=root,
        device=args.device,
        progress_every=args.progress_every_direction,
    )
    layer_vectors_by_mode, vector_meta = VEC.build_layer_vectors(
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
    tau_specs = VEC.run_specs_from_args(args, float(selected["tau_A_value"]), tau_c)
    tau_specs = [spec for spec in tau_specs if spec["gate_mode"] == "peak_zA_tau"]

    gate_cache: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for index, example in enumerate(examples, start=1):
        gate_payload = VEC.compute_gate_base_payload(
            model=model,
            tokenizer=tokenizer,
            example=example,
            selected=selected,
            sae_by_layer=sae_by_layer,
            device=args.device,
        )
        gate_cache.append((example, gate_payload))
        if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(examples)):
            print(f"cached dev calibration gate inputs {index}/{len(examples)}", flush=True)

    records: list[dict[str, Any]] = []
    for strength in strengths:
        for vector_run in layer_vectors_by_mode.values():
            construction = vector_run["construction"]
            mixture_mode = vector_run["mixture_mode"]
            layer_vectors = vector_run["layer_vectors"]
            for index, (example, gate_payload) in enumerate(gate_cache, start=1):
                layer_deltas = VEC.layer_deltas_for_positions(
                    layer_vectors=layer_vectors,
                    target_positions=gate_payload["target_positions"],
                    intervention_strength=strength,
                    device=args.device,
                )
                has_delta = bool(gate_payload["target_positions"]) and bool(layer_deltas)
                triggered_by_spec = {
                    spec["run_id"]: VEC.should_trigger(
                        spec=spec,
                        example=example,
                        peak_za=gate_payload["peak_zA"],
                        neutral_preference_score=gate_payload["neutral_preference_score"],
                        tau_c=tau_c,
                        has_delta=has_delta,
                    )
                    for spec in tau_specs
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
                for spec in tau_specs:
                    triggered = triggered_by_spec[spec["run_id"]]
                    effective_tau_c = (
                        float(spec["tau_C_value"])
                        if spec.get("tau_C_value") is not None
                        else tau_c
                    )
                    after = intervened_scores if triggered and intervened_scores is not None else {
                        "logp_correct": example["baseline_logp_correct"],
                        "logp_wrong": example["baseline_logp_wrong"],
                        "margin": example["baseline_margin"],
                    }
                    records.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "record_type": "lace_config_candidate_record",
                            "stage": "dev_calibration",
                            "dataset": args.dataset,
                            "model_tag": args.model_tag,
                            "source_id": example["source_id"],
                            "template_id": example["template_id"],
                            "pair_id": example.get("pair_id"),
                            "condition": example["condition"],
                            "q_group": example.get("q_group"),
                            "relation": example.get("relation"),
                            "subject": example.get("subject"),
                            "base_config_path": str(args.base_config),
                            "selected_band": selected.get("band_name"),
                            "selected_layers": selected.get("layers"),
                            "top_k_policy": selected.get("top_k_policy"),
                            "top_k_A_global": selected.get("top_k_A_global"),
                            "top_k_R_global": selected.get("top_k_R_global"),
                            "vector_construction": construction,
                            "intervention_construction": construction,
                            "mixture_mode": mixture_mode,
                            "gate_mode": "peak_zA_tau",
                            "run_id": f"{construction}__{mixture_mode}__strength{strength:g}__{spec['run_id']}",
                            "tau_A_value": spec["tau_A_value"],
                            "tau_A_scale_from_frozen": spec["tau_A_scale_from_frozen"],
                            "frozen_tau_A_value": selected["tau_A_value"],
                            "tau_C": effective_tau_c,
                            "tau_C_label": spec.get("tau_C_label"),
                            "frozen_tau_C": tau_c,
                            "intervention_strength": strength,
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
                            "oracle_gate_used": False,
                            "uses_gold_span_fields": False,
                            "span_mode": "extracted",
                        }
                    )
                if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(examples)):
                    print(
                        f"scored dev construction={construction} mixture={mixture_mode} strength={strength:g} rows {index}/{len(examples)}",
                        flush=True,
                    )

    del model
    if args.device.startswith("cuda"):
        import torch

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
            "run_specs": tau_specs,
            "strengths": strengths,
            "expanded_vector_runs": sorted(layer_vectors_by_mode),
            "gate_cache_reused_across_mixtures": True,
            "intervention_forward_reused_across_gate_specs": True,
            "gate_cache_count": len(gate_cache),
        }
    return records, meta


def row_passes_constraints(row: dict[str, Any], args: argparse.Namespace) -> bool:
    return (
        row.get("q2_damage") is not None
        and row["q2_damage"] <= args.eps_q2_damage
        and row.get("true_damage") is not None
        and row["true_damage"] <= args.eps_true_damage
        and row.get("clean_damage") is not None
        and row["clean_damage"] <= args.eps_clean_damage
    )


def selection_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row.get("q1_correction_rate") or 0.0),
        -float(row.get("true_damage") or 0.0),
        -float(row.get("q2_damage") or 0.0),
        -float(row.get("clean_damage") or 0.0),
    )


def select_config(run_summaries: list[dict[str, Any]], args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    peak_rows = [row for row in run_summaries if row.get("gate_mode") == "peak_zA_tau"]
    feasible = [row for row in peak_rows if row_passes_constraints(row, args)]
    if feasible:
        return max(feasible, key=selection_key), "selected_feasible"
    return max(peak_rows, key=selection_key), "fallback_no_feasible_config"


def build_audit(
    *,
    args: argparse.Namespace,
    behavior_rows: list[dict[str, Any]],
    q_rows: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    records: list[dict[str, Any]],
    selected_row: dict[str, Any],
    selection_status: str,
    search_space: dict[str, Any],
) -> dict[str, Any]:
    issues: list[str] = []
    warnings: list[str] = []
    if args.behavior_root.name != "dev_behavior":
        issues.append("behavior_root_is_not_dev_behavior")
    if args.splits_root.name != "dev_splits":
        issues.append("splits_root_is_not_dev_splits")
    if "holdout" in str(args.behavior_root).lower() or "holdout" in str(args.splits_root).lower():
        issues.append("holdout_path_detected")
    if any(row.get("dataset_stage") != "dev" for row in behavior_rows):
        issues.append("behavior_rows_not_dev_stage")
    if any(row.get("dataset_stage") != "dev" for row in q_rows):
        issues.append("q_group_rows_not_dev_stage")
    if any(row.get("uses_gold_span_fields") for row in records):
        issues.append("records_used_gold_span_fields")
    if any(row.get("span_mode") != "extracted" for row in records):
        issues.append("records_not_extracted_span_mode")
    if any(row.get("uses_correct_answer_for_gate") for row in records):
        issues.append("records_used_correct_answer_for_gate")
    if selection_status == "fallback_no_feasible_config":
        warnings.append("no_feasible_config_under_constraints")
    if (
        selected_row
        and args.require_positive_tau_a
        and float(selected_row.get("tau_A_value") or 0.0) <= 0.0
    ):
        issues.append("selected_tau_A_is_not_positive")
    if (
        selected_row
        and args.require_positive_tau_c
        and float(selected_row.get("tau_C") or 0.0) <= 0.0
    ):
        issues.append("selected_tau_C_is_not_positive")
    if (
        search_space["require_all_gate_boundary_regimes"]
        and not search_space["gate_boundary_regimes_complete"]
    ):
        issues.append("gate_boundary_regimes_incomplete")
    if not records and not args.dry_run:
        issues.append("empty_calibration_records")
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "lace_config_selection_audit",
        "status": "fail" if issues else "pass",
        "issues": issues,
        "warnings": warnings,
        "checks": {
            "stage": "dev_calibration",
            "run_mode": args.run_mode,
            "read_holdout_data": False,
            "holdout_used_for_selection": False,
            "selection_on_dev_calibration_only": True,
            "uses_gold_span_fields": False,
            "span_mode": "extracted",
            "uses_correct_answer_for_gate": False,
            "structured_candidate_allowed_for_locator": True,
            "candidate_extractor": extractor_spec(),
            "behavior_root": str(args.behavior_root),
            "splits_root": str(args.splits_root),
            "calibration_sources_path": str(args.calibration_sources),
            "base_config_path": str(args.base_config),
            "behavior_row_count": len(behavior_rows),
            "q_group_row_count": len(q_rows),
            "example_count": len(examples),
            "record_count": len(records),
            "selection_status": selection_status,
            "selection_policy": SELECTION_POLICY,
            "search_profile": args.search_profile,
            "search_config_path": str(args.search_config),
            "search_candidate_count": search_space["candidate_count"],
            "effective_tau_A_values": search_space["effective_tau_a_values"],
            "effective_tau_C_values": search_space["effective_tau_c_values"],
            "intervention_strengths": search_space["intervention_strengths"],
            "require_positive_tau_A": search_space["require_positive_tau_a"],
            "require_positive_tau_C": search_space["require_positive_tau_c"],
            "require_all_gate_boundary_regimes": search_space[
                "require_all_gate_boundary_regimes"
            ],
            "gate_boundary_regimes": search_space["gate_boundary_regimes"],
            "gate_boundary_regimes_complete": search_space[
                "gate_boundary_regimes_complete"
            ],
            "selected_tau_A_value": selected_row.get("tau_A_value") if selected_row else None,
            "selected_tau_C_value": selected_row.get("tau_C") if selected_row else None,
            "selected_intervention_strength": (
                selected_row.get("intervention_strength") if selected_row else None
            ),
            "constraints": {
                "eps_q2_damage": args.eps_q2_damage,
                "eps_true_damage": args.eps_true_damage,
                "eps_clean_damage": args.eps_clean_damage,
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["pqpqa", "counterfact"], required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--root", type=Path, default=repo_root())
    parser.add_argument("--models-config", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--behavior-root", type=Path)
    parser.add_argument("--splits-root", type=Path)
    parser.add_argument("--calibration-sources", type=Path)
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--sae-path", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-mode", choices=["formal", "smoke"], default="formal")
    parser.add_argument("--search-config", type=Path)
    parser.add_argument(
        "--search-profile",
        choices=["unified_v1"],
        default="unified_v1",
        help=(
            "Development-only LACE search profile from va_search_space.json."
        ),
    )
    parser.set_defaults(
        mixture_modes=None,
        vector_constructions=None,
        gate_modes=None,
        tau_a_scales=None,
        tau_a_values=None,
        tau_c_values=None,
        intervention_strengths=None,
    )
    parser.add_argument("--eps-q2-damage", type=float, default=0.02)
    parser.add_argument("--eps-true-damage", type=float, default=0.15)
    parser.add_argument("--eps-clean-damage", type=float, default=0.03)
    parser.add_argument("--max-direction-pairs-per-group", type=int, default=0)
    parser.add_argument("--max-calibration-sources", type=int, default=0)
    parser.add_argument("--max-pairs-per-group", type=int, default=0)
    parser.add_argument("--max-side-rows-per-condition", type=int, default=0)
    parser.add_argument("--strict-damage-threshold", type=float, default=0.5)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--progress-every-direction", type=int, default=40)
    parser.add_argument("--random-seed", type=int, default=13)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    search_profile_config = resolve_search_profile(args, root)
    output_root = resolve_cli_path(args.output_root, root) or default_output_root(root)
    args.base_config = resolve_cli_path(args.base_config, root) or default_base_config_path(output_root, args.dataset, args.model_tag)
    args.behavior_root = resolve_cli_path(args.behavior_root, root) or default_behavior_root(output_root, args.dataset, args.model_tag)
    args.splits_root = resolve_cli_path(args.splits_root, root) or default_splits_root(output_root, args.dataset, args.model_tag)
    args.calibration_sources = (
        resolve_cli_path(args.calibration_sources, root)
        or args.splits_root / "calibration_sources.json"
    )
    args.feature_root = resolve_cli_path(args.feature_root, root) or default_feature_root(output_root, args.dataset, args.model_tag)
    args.output_dir = resolve_cli_path(args.output_dir, root) or default_output_dir(output_root, args.dataset, args.model_tag)
    models_config = resolve_cli_path(args.models_config, root) or default_models_config(root)
    model_spec = load_model_spec(models_config, args.model_tag, root, allow_placeholder=True)

    base_config = load_json(args.base_config)
    config_issues = PHASE5.config_guard(base_config, args.base_config, args.dataset, args.model_tag)
    if config_issues:
        raise SystemExit(f"Base config guard failed: {config_issues}")
    selected_base = base_config["selected_config"]
    search_space = build_search_space_meta(args, selected_base)

    dataset_stage = load_dataset_stage(root / "configs/datasets.json", args.dataset, "dev", root)
    calibration_sources = PHASE4.load_calibration_sources(args.calibration_sources)
    q_rows = PHASE4.filter_rows_to_sources(load_jsonl(args.behavior_root / "q_group_labels.jsonl"), calibration_sources)
    behavior_rows = PHASE4.filter_rows_to_sources(load_jsonl(args.behavior_root / "behavior_records.jsonl"), calibration_sources)
    pressure_by_pair, side_by_key = PHASE4.load_views_for_dev(root, args.dataset)
    examples = select_dev_calibration_examples(
        q_rows=q_rows,
        behavior_rows=behavior_rows,
        pressure_by_pair=pressure_by_pair,
        side_by_key=side_by_key,
        max_calibration_sources=args.max_calibration_sources,
        max_pairs_per_group=args.max_pairs_per_group,
        max_side_rows_per_condition=args.max_side_rows_per_condition,
    )
    if not examples:
        raise ValueError("No dev calibration examples selected")

    if args.dry_run:
        records: list[dict[str, Any]] = []
        run_summaries: list[dict[str, Any]] = []
        selected_row: dict[str, Any] = {}
        selection_status = "dry_run_not_selected"
        scoring_meta = {
            "dry_run": True,
            "model_loaded": False,
            "example_count": len(examples),
            "condition_counts": dict(Counter(row["condition"] for row in examples)),
            "q_group_counts": dict(Counter(row.get("q_group") or "none" for row in examples)),
        }
    else:
        records, scoring_meta = score_candidates(
            args=args,
            root=root,
            base_config=base_config,
            model_spec=model_spec,
            examples=examples,
            output_root=output_root,
            feature_root=args.feature_root,
        )
        run_summaries = VEC.summarize_by_run(records, args.strict_damage_threshold)
        selected_row, selection_status = select_config(run_summaries, args)

    audit = build_audit(
        args=args,
        behavior_rows=behavior_rows,
        q_rows=q_rows,
        examples=examples,
        records=records,
        selected_row=selected_row,
        selection_status=selection_status,
        search_space=search_space,
    )
    selected_method_config = None
    if selected_row:
        selected_method_config = {
            "vector_construction": selected_row.get("vector_construction", "full_va"),
            "intervention_construction": selected_row.get("intervention_construction", selected_row.get("vector_construction", "full_va")),
            "mixture_mode": selected_row["mixture_mode"],
            "gate_mode": selected_row["gate_mode"],
            "tau_A_scale_from_frozen": selected_row["tau_A_scale_from_frozen"],
            "tau_A_value": selected_row["tau_A_value"],
            "tau_C": selected_row["tau_C"],
            "intervention_strength": selected_row["intervention_strength"],
            "selection_status": selection_status,
            "selection_metrics": selected_row,
            "selection_constraints": {
                "eps_q2_damage": args.eps_q2_damage,
                "eps_true_damage": args.eps_true_damage,
                "eps_clean_damage": args.eps_clean_damage,
            },
            "selected_on": "development_calibration_split_only",
        }

    summary = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "lace_config_selection_summary",
        "stage": "dev_calibration",
        "dataset": args.dataset,
        "model_tag": args.model_tag,
        "sae_compatibility": sae_compatibility_metadata(model_spec),
        "dataset_name": dataset_stage.name,
        "dry_run": bool(args.dry_run),
        "run_mode": args.run_mode,
        "is_formal_calibration": (
            not args.dry_run
            and args.run_mode == "formal"
            and selection_status == "selected_feasible"
        ),
        "audit_status": audit["status"],
        "selection_status": selection_status,
        "selection_policy": SELECTION_POLICY,
        "search_profile": args.search_profile,
        "search_config_path": str(args.search_config),
        "search_profile_config": search_profile_config,
        "base_config_path": str(args.base_config),
        "output_dir": str(args.output_dir),
        "read_holdout_data": False,
        "holdout_used_for_selection": False,
        "uses_gold_span_fields": False,
        "span_mode": "extracted",
        "uses_correct_answer_for_gate": False,
        "structured_candidate_allowed_for_locator": True,
        "candidate_extractor": extractor_spec(),
        "base_selected_config": {
            "band_name": selected_base.get("band_name"),
            "layers": selected_base.get("layers"),
            "top_k_policy": selected_base.get("top_k_policy"),
            "top_k_A_global": selected_base.get("top_k_A_global"),
            "top_k_R_global": selected_base.get("top_k_R_global"),
            "base_mixture_mode": selected_base.get("mixture_mode"),
            "base_tau_A_value": selected_base.get("tau_A_value"),
        },
        "search_space": search_space,
        "selected_method_config": selected_method_config,
        "run_summaries": run_summaries,
        "scoring_meta": scoring_meta,
        "outputs": {
            "records": str(args.output_dir / "records.jsonl"),
            "calibration_grid": str(args.output_dir / "calibration_grid.jsonl"),
            "summary": str(args.output_dir / "summary.json"),
            "audit": str(args.output_dir / "audit.json"),
            "config": str(args.output_dir / output_config_name(args)),
        },
    }

    is_formal_calibration = (
        not args.dry_run
        and args.run_mode == "formal"
        and selection_status == "selected_feasible"
    )
    vector_config = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "lace_config",
        "run_mode": args.run_mode,
        "is_formal_calibration": is_formal_calibration,
        "frozen_config": is_formal_calibration,
        "selector_mode": "development_calibration",
        "selection_policy": SELECTION_POLICY,
        "search_profile": args.search_profile,
        "search_config_path": str(args.search_config),
        "search_space": search_space,
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
        "vector_constructions": VEC.parse_vector_constructions(args.vector_constructions),
        "random_seed": args.random_seed,
        "base_config_path": str(args.base_config),
        "base_selected_config": selected_base,
        "selected_method_config": selected_method_config,
        "selection_status": selection_status,
        "audit_status": audit["status"],
        "summary_path": str(args.output_dir / "summary.json"),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "records.jsonl", records)
    write_jsonl(args.output_dir / "calibration_grid.jsonl", run_summaries)
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "audit.json", audit)
    config_path = args.output_dir / output_config_name(args)
    write_json(config_path, vector_config)
    print(f"Wrote {args.output_dir / 'records.jsonl'}")
    print(f"Wrote {args.output_dir / 'calibration_grid.jsonl'}")
    print(f"Wrote {args.output_dir / 'summary.json'}")
    print(f"Wrote {args.output_dir / 'audit.json'}")
    print(f"Wrote {config_path}")
    if audit["status"] != "pass":
        raise SystemExit(f"LACE configuration audit failed: {audit['issues']}")
    if not args.dry_run and selection_status != "selected_feasible":
        raise SystemExit("No feasible LACE config selected; see audit/summary")


if __name__ == "__main__":
    main()
