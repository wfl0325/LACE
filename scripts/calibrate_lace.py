#!/usr/bin/env python3
"""Calibrate LACE on the development split with factorized shortlisting.

This is the formal base selector. It replaces brute-force full-grid
calibration with a factorized two-stage protocol:

1. Build a factorized surrogate shortlist from feature-bank primitive
   statistics and development calibration metadata.
2. Exact-score only the shortlisted configurations on the development
   calibration split.
3. Select the frozen config only from exact-scored shortlist rows.

The surrogate stage is never allowed to write ``lace_base_config.json``.
The final config is written only by ``--run-mode exact`` after exact scoring.

The exact intervention scorer is an internal runtime used only after
shortlisting. It is not a second public method.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.candidate_extractor import extractor_spec  # noqa: E402
from lib.io_utils import load_json, load_jsonl, repo_root, resolve_path, write_json, write_jsonl  # noqa: E402
from lib.layer_band_utils import select_band_features_global  # noqa: E402
from lib.model_registry import load_model_spec  # noqa: E402


SCHEMA_VERSION = "lace_lca_sae_factorized_calibration_v1"


def load_scoring_module():
    """Load the exact development-calibration scoring runtime."""
    module_path = SCRIPT_DIR / "internal" / "calibration_runtime.py"
    spec = importlib.util.spec_from_file_location("lace_exact_calibration_runtime", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SCORER = load_scoring_module()


def default_output_root(root: Path) -> Path:
    return root / "outputs"


def default_calibration_root(output_root: Path, dataset: str, model_tag: str) -> Path:
    return output_root / dataset / model_tag / "calibration"


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def mean(values: list[Any]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return sum(cleaned) / len(cleaned) if cleaned else None


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def minmax_norm(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if abs(hi - lo) <= 1e-12:
        return [0.5 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def q_group_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row.get("q_group") or "unknown" for row in rows).items()))


def condition_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row.get("condition") or "unknown" for row in rows).items()))


def feature_stats(
    bank_by_layer: dict[int, list[dict[str, Any]]],
    *,
    layers: list[int],
    top_k: int,
    bank_name: str,
) -> dict[str, Any]:
    selected = select_band_features_global(bank_by_layer, layers=layers, top_k=top_k)
    activation_gap_field = "q1_minus_q2_activation_positive" if bank_name == "A" else "q2_minus_q1_activation"
    return {
        "bank": bank_name,
        "feature_count": len(selected),
        "mean_causal_score": mean([row.get("causal_score") for row in selected]),
        "mean_q1_correction_rate": mean([row.get("q1_correction_rate") for row in selected]),
        "mean_q1_delta_margin": mean([row.get("q1_mean_delta_margin") for row in selected]),
        "mean_q2_side_effect_rate": mean([row.get("q2_side_effect_rate") for row in selected]),
        "mean_q2_delta_margin": mean([row.get("q2_mean_delta_margin") for row in selected]),
        "mean_activation_gap": mean([row.get(activation_gap_field) for row in selected]),
        "feature_uids": [row["feature_uid"] for row in selected],
    }


def row_top_k_a(row: dict[str, Any]) -> int:
    return int(row.get("top_k_A_global", row.get("top_k_A", row.get("top_k_A_per_layer", 0))))


def row_top_k_r(row: dict[str, Any]) -> int:
    return int(row.get("top_k_R_global", row.get("top_k_R", row.get("top_k_R_per_layer", 0))))


def with_surrogate_scores(
    rows: list[dict[str, Any]],
    bank_a_by_layer: dict[int, list[dict[str, Any]]],
    bank_r_by_layer: dict[int, list[dict[str, Any]]],
    constraints: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Attach factorized surrogate scores to formal candidate rows.

    These values are only used for shortlist generation. They are not reported
    as final method metrics and cannot write the frozen config.
    """
    enriched: list[dict[str, Any]] = []
    primitive_rows: list[dict[str, Any]] = []
    a_gaps: list[float] = []
    r_gaps: list[float] = []
    a_deltas: list[float] = []
    r_deltas: list[float] = []

    stats_cache: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        layers = [int(layer) for layer in row["layers"]]
        band = row["band_name"]
        key_a = (band, "A", row_top_k_a(row))
        key_r = (band, "R", row_top_k_r(row))
        if key_a not in stats_cache:
            stats_cache[key_a] = feature_stats(
                bank_a_by_layer,
                layers=layers,
                top_k=row_top_k_a(row),
                bank_name="A",
            )
        if key_r not in stats_cache:
            stats_cache[key_r] = feature_stats(
                bank_r_by_layer,
                layers=layers,
                top_k=row_top_k_r(row),
                bank_name="R",
            )
        a_stats = stats_cache[key_a]
        r_stats = stats_cache[key_r]
        a_gaps.append(safe_float(a_stats["mean_activation_gap"]))
        r_gaps.append(safe_float(r_stats["mean_activation_gap"]))
        a_deltas.append(safe_float(a_stats["mean_q1_delta_margin"]))
        r_deltas.append(safe_float(r_stats["mean_q1_delta_margin"]))

    a_gap_norms = minmax_norm(a_gaps)
    r_gap_norms = minmax_norm(r_gaps)
    a_delta_norms = minmax_norm(a_deltas)
    r_delta_norms = minmax_norm(r_deltas)
    max_alpha_a = max([safe_float(row.get("alpha_A")) for row in rows] or [1.0]) or 1.0
    max_alpha_r = max([safe_float(row.get("alpha_R")) for row in rows] or [1.0]) or 1.0

    for index, row in enumerate(rows):
        candidate = copy.deepcopy(row)
        layers = [int(layer) for layer in candidate["layers"]]
        band = candidate["band_name"]
        a_stats = stats_cache[(band, "A", row_top_k_a(candidate))]
        r_stats = stats_cache[(band, "R", row_top_k_r(candidate))]
        alpha_a_weight = safe_float(candidate.get("alpha_A")) / max_alpha_a
        alpha_r_weight = safe_float(candidate.get("alpha_R")) / max_alpha_r
        tau_a_quantile = safe_float(candidate.get("tau_A_quantile"), 0.5)
        tau_c = safe_float(candidate.get("tau_C"))

        a_gap = a_gap_norms[index]
        r_gap = r_gap_norms[index]
        neutral_factor = 1.0 / (1.0 + max(0.0, tau_c))
        base_trigger = clamp(1.0 - tau_a_quantile)
        gate_q1_trigger = clamp(base_trigger * neutral_factor * (0.65 + 0.25 * a_gap + 0.10 * r_gap))
        gate_q2_trigger = clamp(base_trigger * neutral_factor * (0.35 + 0.35 * (1.0 - a_gap)))
        side_trigger = clamp(0.65 * gate_q2_trigger + 0.10 * gate_q1_trigger)

        a_corr = safe_float(a_stats["mean_q1_correction_rate"])
        r_corr = safe_float(r_stats["mean_q1_correction_rate"])
        a_delta = safe_float(a_stats["mean_q1_delta_margin"])
        r_delta = safe_float(r_stats["mean_q1_delta_margin"])
        a_side = safe_float(a_stats["mean_q2_side_effect_rate"])
        r_side = safe_float(r_stats["mean_q2_side_effect_rate"])

        combo_bonus = 0.04 if candidate["alpha_A"] > 0 and candidate["alpha_R"] > 0 else 0.0
        q1_corr_proxy = clamp(gate_q1_trigger * (alpha_a_weight * a_corr + alpha_r_weight * r_corr + combo_bonus))
        q1_delta_proxy = gate_q1_trigger * (alpha_a_weight * a_delta + alpha_r_weight * r_delta)
        q2_damage_proxy = clamp(gate_q2_trigger * (alpha_a_weight * a_side + alpha_r_weight * r_side + 0.04 * side_trigger))
        true_damage_proxy = clamp(0.5 * q2_damage_proxy + 0.02 * side_trigger)
        clean_damage_proxy = clamp(0.35 * q2_damage_proxy + 0.015 * side_trigger)
        surrogate_feasible = (
            q2_damage_proxy <= constraints["eps_q2_damage"] * 2.0
            and true_damage_proxy <= constraints["eps_true_damage"] * 2.0
            and clean_damage_proxy <= constraints["eps_clean_damage"] * 2.0
        )
        score = q1_corr_proxy + 0.02 * q1_delta_proxy - q2_damage_proxy - true_damage_proxy - clean_damage_proxy

        primitive = {
            "schema_version": SCHEMA_VERSION,
            "record_type": "primitive_response_summary",
            "dataset": candidate.get("dataset"),
            "model_tag": candidate.get("model_tag"),
            "band_name": band,
            "layers": layers,
            "top_k_policy": candidate.get("top_k_policy", "band_global"),
            "top_k_A_global": row_top_k_a(candidate),
            "top_k_R_global": row_top_k_r(candidate),
            "A": {key: value for key, value in a_stats.items() if key != "feature_uids"},
            "R": {key: value for key, value in r_stats.items() if key != "feature_uids"},
            "source": "feature_bank_discovery_statistics",
        }
        primitive_rows.append(primitive)
        candidate.update(
            {
                "selector_mode": "factorized",
                "surrogate_used_for_final_selection": False,
                "surrogate_role": "shortlist_generation_only",
                "exact_scored": False,
                "surrogate_q1_trigger_rate": gate_q1_trigger,
                "surrogate_q2_trigger_rate": gate_q2_trigger,
                "surrogate_side_trigger_rate": side_trigger,
                "surrogate_q1_correction_rate": q1_corr_proxy,
                "surrogate_q1_delta_margin": q1_delta_proxy,
                "surrogate_q2_damage": q2_damage_proxy,
                "surrogate_true_damage": true_damage_proxy,
                "surrogate_clean_damage": clean_damage_proxy,
                "surrogate_feasible": surrogate_feasible,
                "surrogate_selection_score": score,
                "surrogate_A_activation_gap_norm": a_gap,
                "surrogate_R_activation_gap_norm": r_gap,
                "surrogate_A_delta_norm": a_delta_norms[index],
                "surrogate_R_delta_norm": r_delta_norms[index],
            }
        )
        enriched.append(candidate)
    return enriched, primitive_rows, gate_pareto_rows(enriched)


def gate_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["band_name"],
        tuple(row["layers"]),
        row_top_k_a(row),
        row["tau_A_quantile"],
        row["tau_C"],
    )


def gate_pareto_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = gate_key(row)
        payload = grouped.setdefault(
            key,
            {
                "schema_version": SCHEMA_VERSION,
                "record_type": "gate_pareto_candidate",
                "band_name": row["band_name"],
                "layers": row["layers"],
                "top_k_policy": row.get("top_k_policy", "band_global"),
                "top_k_A_global": row_top_k_a(row),
                "tau_A_quantile": row["tau_A_quantile"],
                "tau_C": row["tau_C"],
                "surrogate_q1_trigger_rate": 0.0,
                "surrogate_q2_trigger_rate": 1.0,
                "surrogate_side_trigger_rate": 1.0,
                "source": "feature_bank_activation_surrogate",
                "exact_peak_zA_cached": False,
                "exact_neutral_preference_cached": False,
            },
        )
        payload["surrogate_q1_trigger_rate"] = max(
            payload["surrogate_q1_trigger_rate"], row["surrogate_q1_trigger_rate"]
        )
        payload["surrogate_q2_trigger_rate"] = min(
            payload["surrogate_q2_trigger_rate"], row["surrogate_q2_trigger_rate"]
        )
        payload["surrogate_side_trigger_rate"] = min(
            payload["surrogate_side_trigger_rate"], row["surrogate_side_trigger_rate"]
        )

    candidates = list(grouped.values())
    for row in candidates:
        dominated = False
        for other in candidates:
            if row is other:
                continue
            better_or_equal = (
                other["surrogate_q1_trigger_rate"] >= row["surrogate_q1_trigger_rate"]
                and other["surrogate_q2_trigger_rate"] <= row["surrogate_q2_trigger_rate"]
                and other["surrogate_side_trigger_rate"] <= row["surrogate_side_trigger_rate"]
            )
            strictly_better = (
                other["surrogate_q1_trigger_rate"] > row["surrogate_q1_trigger_rate"]
                or other["surrogate_q2_trigger_rate"] < row["surrogate_q2_trigger_rate"]
                or other["surrogate_side_trigger_rate"] < row["surrogate_side_trigger_rate"]
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        row["gate_pareto"] = not dominated
    candidates.sort(
        key=lambda row: (
            row["gate_pareto"],
            row["surrogate_q1_trigger_rate"] - row["surrogate_q2_trigger_rate"] - row["surrogate_side_trigger_rate"],
        ),
        reverse=True,
    )
    for rank, row in enumerate(candidates, start=1):
        row["gate_pareto_rank"] = rank
    return candidates


def candidate_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["band_name"],
        tuple(row["layers"]),
        row_top_k_a(row),
        row_top_k_r(row),
        row["mixture_mode"],
        row["alpha_A"],
        row["alpha_R"],
        row["tau_A_quantile"],
        row["tau_C"],
    )


def build_shortlist(
    rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
    shortlist_size: int,
    min_per_band: int,
) -> list[dict[str, Any]]:
    pareto_keys = {
        (
            row["band_name"],
            tuple(row["layers"]),
            row_top_k_a(row),
            row["tau_A_quantile"],
            row["tau_C"],
        )
        for row in gate_rows
        if row["gate_pareto"]
    }

    def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            row.get("surrogate_feasible", False),
            row.get("surrogate_selection_score", -10**9),
            row.get("surrogate_q1_correction_rate", -10**9),
            -row.get("surrogate_q2_damage", 10**9),
            row.get("surrogate_q1_delta_margin", -10**9),
        )

    prioritized = [
        row
        for row in rows
        if gate_key(row) in pareto_keys
    ]
    prioritized.sort(key=sort_key, reverse=True)
    by_band: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prioritized:
        by_band[row["band_name"]].append(row)
    all_by_band: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        all_by_band[row["band_name"]].append(row)
    for band_rows in all_by_band.values():
        band_rows.sort(key=sort_key, reverse=True)

    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[Any, ...]] = set()
    band_names = sorted(all_by_band)
    if min_per_band > 0 and shortlist_size > 0:
        effective_min = min(min_per_band, max(1, shortlist_size // max(1, len(band_names))))
        for band in band_names:
            candidates = by_band.get(band, [])
            if len(candidates) < effective_min:
                existing = {candidate_key(row) for row in candidates}
                candidates = candidates + [row for row in all_by_band[band] if candidate_key(row) not in existing]
            for row in candidates[:effective_min]:
                key = candidate_key(row)
                if key in selected_keys:
                    continue
                selected.append(row)
                selected_keys.add(key)

    if len(prioritized) < shortlist_size:
        seen = {candidate_key(row) for row in prioritized}
        filler = [row for row in rows if candidate_key(row) not in seen]
        filler.sort(key=sort_key, reverse=True)
        prioritized.extend(filler)
    for row in prioritized:
        if len(selected) >= shortlist_size:
            break
        key = candidate_key(row)
        if key in selected_keys:
            continue
        selected.append(row)
        selected_keys.add(key)

    shortlisted = [copy.deepcopy(row) for row in selected[:shortlist_size]]
    for rank, row in enumerate(shortlisted, start=1):
        row["shortlist_rank"] = rank
        row["shortlist_source"] = "band_diverse_gate_pareto_then_surrogate_score"
        row["exact_scoring_required_for_final_selection"] = True
    return shortlisted


def exact_selection_sort_key(row: dict[str, Any], policy: str) -> tuple[float, ...]:
    if policy == "q1_max":
        return (
            safe_float(row.get("q1_correction_rate"), 0.0),
            safe_float(row.get("q1_delta_margin"), 0.0),
            -safe_float(row.get("q2_damage"), 0.0),
            -safe_float(row.get("clean_damage"), 0.0),
            -safe_float(row.get("clean_strict_damage"), 0.0),
            -safe_float(row.get("true_damage"), 0.0),
            -safe_float(row.get("true_strict_damage"), 0.0),
        )
    return (
        safe_float(row.get("selection_score"), -10**9),
        safe_float(row.get("q1_delta_margin"), -10**9),
    )


def select_exact_config(rows: list[dict[str, Any]], policy: str = "constrained_score") -> dict[str, Any] | None:
    feasible = [row for row in rows if row.get("feasible")]
    feasible.sort(key=lambda row: exact_selection_sort_key(row, policy), reverse=True)
    return feasible[0] if feasible else None


def resolve_inputs(args: argparse.Namespace, root: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    output_root = resolve_path(args.output_root or default_output_root(root), root)
    behavior_root = resolve_path(
        args.behavior_root or SCORER.default_behavior_root(output_root, args.dataset, args.model_tag),
        root,
    )
    splits_root = resolve_path(
        args.splits_root or SCORER.default_splits_root(output_root, args.dataset, args.model_tag),
        root,
    )
    feature_banks_root = resolve_path(
        args.feature_banks_root or SCORER.default_feature_banks_root(output_root, args.dataset, args.model_tag),
        root,
    )
    calibration_root = resolve_path(
        args.calibration_root or default_calibration_root(output_root, args.dataset, args.model_tag),
        root,
    )
    search_config_path = resolve_path(
        args.search_config or root / "configs/lace_search_space.json",
        root,
    )
    constraints_config_path = resolve_path(
        args.constraints_config or root / "configs/side_effect_constraints.json",
        root,
    )
    models_config = resolve_path(args.models_config or SCORER.default_models_config(root), root)
    input_paths = {
        "behavior_root": behavior_root,
        "calibration_sources": splits_root / "calibration_sources.json",
        "q_group_labels": behavior_root / "q_group_labels.jsonl",
        "behavior_records": behavior_root / "behavior_records.jsonl",
        "feature_bank_A": feature_banks_root / "feature_bank_A.jsonl",
        "feature_bank_R": feature_banks_root / "feature_bank_R.jsonl",
        "search_config": search_config_path,
        "constraints_config": constraints_config_path,
        "models_config": models_config,
        "calibration_root": calibration_root,
    }
    for label, path in input_paths.items():
        if label == "calibration_root":
            continue
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    payload = {
        "output_root": output_root,
        "behavior_root": behavior_root,
        "splits_root": splits_root,
        "feature_banks_root": feature_banks_root,
        "calibration_root": calibration_root,
    }
    return input_paths, payload


def build_candidate_rows(args: argparse.Namespace, input_paths: dict[str, Path]) -> dict[str, Any]:
    search_config = load_json(input_paths["search_config"])
    constraints_payload = load_json(input_paths["constraints_config"])
    constraints = constraints_payload["default"]
    calibration_sources = SCORER.load_calibration_sources(input_paths["calibration_sources"])
    q_rows = SCORER.filter_rows_to_sources(load_jsonl(input_paths["q_group_labels"]), calibration_sources)
    behavior_rows = SCORER.filter_rows_to_sources(load_jsonl(input_paths["behavior_records"]), calibration_sources)
    bank_a_by_layer = SCORER.load_feature_bank(input_paths["feature_bank_A"], "A", args.dataset, args.model_tag)
    bank_r_by_layer = SCORER.load_feature_bank(input_paths["feature_bank_R"], "R", args.dataset, args.model_tag)
    grid_rows, grid_warnings = SCORER.build_grid(
        search_config=search_config,
        constraints=constraints,
        bank_a_by_layer=bank_a_by_layer,
        bank_r_by_layer=bank_r_by_layer,
    )
    filter_args = argparse.Namespace(**vars(args))
    filter_args.run_mode = "formal"
    grid_rows, grid_filter_report = SCORER.filter_grid_for_run_mode(grid_rows, filter_args)
    enriched, primitive_rows, gate_rows = with_surrogate_scores(
        grid_rows,
        bank_a_by_layer,
        bank_r_by_layer,
        constraints,
    )
    shortlist = build_shortlist(enriched, gate_rows, args.shortlist_size, args.min_shortlist_per_band)
    return {
        "search_config": search_config,
        "constraints": constraints,
        "calibration_sources": calibration_sources,
        "q_rows": q_rows,
        "behavior_rows": behavior_rows,
        "bank_a_by_layer": bank_a_by_layer,
        "bank_r_by_layer": bank_r_by_layer,
        "grid_warnings": grid_warnings,
        "grid_filter_report": grid_filter_report,
        "surrogate_rows": enriched,
        "primitive_rows": primitive_rows,
        "gate_rows": gate_rows,
        "shortlist": shortlist,
    }


def build_audit(
    *,
    args: argparse.Namespace,
    input_paths: dict[str, Path],
    payload: dict[str, Any],
    exact_rows: list[dict[str, Any]] | None = None,
    final_config_written: bool = False,
) -> dict[str, Any]:
    issues: list[str] = []
    warnings = list(payload["grid_warnings"])
    if input_paths["behavior_root"].name != "dev_behavior":
        issues.append("behavior_root_is_not_dev_behavior")
    if any("holdout" in str(path).lower() for path in input_paths.values()):
        issues.append("holdout_path_detected_in_base_calibration_inputs")
    if not payload["shortlist"]:
        issues.append("empty_exact_shortlist")
    if args.run_mode == "shortlist" and final_config_written:
        issues.append("surrogate_shortlist_wrote_formal_config")
    if exact_rows is not None and any(not row.get("scored") for row in exact_rows):
        issues.append("exact_mode_contains_unscored_rows")
    if exact_rows is not None and any(row.get("uses_correct_answer_for_gate") for row in exact_rows):
        issues.append("exact_rows_used_correct_answer_for_gate")

    observed_conditions = condition_counts(payload["behavior_rows"])
    cue_status = SCORER.cue_only_status(observed_conditions)
    if cue_status["cue_only_unavailable_in_dev"]:
        warnings.append("cue_only_unavailable_in_dev:cue_damage_holdout_report_only")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "fail" if issues else "pass",
        "issues": issues,
        "warnings": warnings,
        "checks": {
            "selector_mode": "factorized",
            "run_mode": args.run_mode,
            "read_holdout_data": False,
            "holdout_used_for_selection": False,
            "holdout_path_recorded": False,
            "uses_gold_span_fields": False,
            "span_mode": "extracted",
            "uses_correct_answer_for_gate": False,
            "structured_candidate_allowed_for_locator": True,
            "candidate_extractor": extractor_spec(),
            "surrogate_composition_role": "shortlist_generation_only",
            "surrogate_used_for_final_selection": False,
            "final_selection_source": "exact_scored_shortlist" if args.run_mode == "exact" else None,
            "selection_policy": args.selection_policy,
            "final_config_written": final_config_written,
            "full_grid_used_for_selection": False,
            "exact_scoring_runtime": "internal/calibration_runtime.py",
            "formal_bands_policy": payload["grid_filter_report"].get("formal_band_policy"),
            "raw_grid_rows": payload["grid_filter_report"].get("raw_grid_rows"),
            "formal_candidate_rows": len(payload["surrogate_rows"]),
            "gate_candidate_rows": len(payload["gate_rows"]),
            "gate_pareto_rows": sum(1 for row in payload["gate_rows"] if row.get("gate_pareto")),
            "shortlist_size_requested": args.shortlist_size,
            "min_shortlist_per_band": args.min_shortlist_per_band,
            "shortlist_size_actual": len(payload["shortlist"]),
            "exact_scored_rows": len(exact_rows or []),
            "calibration_source_count": len(payload["calibration_sources"]),
            "q_group_counts": q_group_counts(payload["q_rows"]),
            "condition_counts": observed_conditions,
            **cue_status,
        },
    }


def build_summary(
    *,
    args: argparse.Namespace,
    paths: dict[str, Path],
    payload: dict[str, Any],
    exact_rows: list[dict[str, Any]] | None = None,
    exact_records: list[dict[str, Any]] | None = None,
    final_config_written: bool = False,
) -> dict[str, Any]:
    selected = select_exact_config(exact_rows or [], args.selection_policy) if exact_rows else None
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "factorized_calibration_summary",
        "dataset": args.dataset,
        "model_tag": args.model_tag,
        "selector_mode": "factorized",
        "run_mode": args.run_mode,
        "stage": "dev_calibration",
        "selected_on": "development_calibration_split_only",
        "holdout_used_for_selection": False,
        "uses_gold_span_fields": False,
        "span_mode": "extracted",
        "uses_correct_answer_for_gate": False,
        "structured_candidate_allowed_for_locator": True,
        "candidate_extractor": extractor_spec(),
        "surrogate_composition_role": "shortlist_generation_only",
        "surrogate_used_for_final_selection": False,
        "final_selection_source": "exact_scored_shortlist" if args.run_mode == "exact" else None,
        "final_config_written": final_config_written,
        "selection_status": (
            f"exact_selected_feasible_config_{args.selection_policy}"
            if selected
            else ("shortlist_ready_exact_scoring_required" if args.run_mode == "shortlist" else "no_feasible_exact_config")
        ),
        "selection_policy": args.selection_policy,
        "selected_config_preview": selected,
        "calibration_source_count": len(payload["calibration_sources"]),
        "q_group_counts": q_group_counts(payload["q_rows"]),
        "condition_counts": condition_counts(payload["behavior_rows"]),
        "raw_grid_rows": payload["grid_filter_report"].get("raw_grid_rows"),
        "formal_candidate_rows": len(payload["surrogate_rows"]),
        "gate_candidate_rows": len(payload["gate_rows"]),
        "gate_pareto_rows": sum(1 for row in payload["gate_rows"] if row.get("gate_pareto")),
        "shortlist_size_requested": args.shortlist_size,
        "min_shortlist_per_band": args.min_shortlist_per_band,
        "shortlist_size_actual": len(payload["shortlist"]),
        "exact_scored_rows": len(exact_rows or []),
        "exact_record_count": len(exact_records or []),
        "constraints": payload["constraints"],
        "grid_filter_report": payload["grid_filter_report"],
        "outputs": {
            "cache_manifest": str(paths["cache_manifest"]),
            "gate_pareto_grid": str(paths["gate_pareto_grid"]),
            "primitive_response_records": str(paths["primitive_response_records"]),
            "surrogate_grid": str(paths["surrogate_grid"]),
            "exact_shortlist_grid": str(paths["exact_shortlist_grid"]),
            "exact_gate_signal_records": str(paths["exact_gate_signal_records"]),
            "factorized_summary": str(paths["factorized_summary"]),
            "factorized_audit": str(paths["factorized_audit"]),
            "formal_config": str(paths["formal_config"]),
            "calibration_grid": str(paths["calibration_grid"]),
            "calibration_records": str(paths["calibration_records"]),
        },
    }


def output_paths(calibration_root: Path) -> dict[str, Path]:
    cache_root = calibration_root / "calibration_cache"
    return {
        "cache_root": cache_root,
        "cache_manifest": cache_root / "cache_manifest.json",
        "gate_pareto_grid": calibration_root / "gate_pareto_grid.jsonl",
        "primitive_response_records": calibration_root / "primitive_response_records.jsonl",
        "surrogate_grid": calibration_root / "surrogate_grid.jsonl",
        "exact_shortlist_grid": calibration_root / "exact_shortlist_grid.jsonl",
        "exact_gate_signal_records": cache_root / "exact_gate_signal_records.jsonl",
        "factorized_summary": calibration_root / "calibration_summary.factorized.json",
        "factorized_audit": calibration_root / "calibration_audit.factorized.json",
        "calibration_grid": calibration_root / "calibration_grid.jsonl",
        "calibration_records": calibration_root / "calibration_records.jsonl",
        "calibration_summary": calibration_root / "calibration_summary.json",
        "calibration_audit": calibration_root / "calibration_audit.json",
        "formal_config": calibration_root / "lace_base_config.json",
    }


def write_shortlist_outputs(args: argparse.Namespace, paths: dict[str, Path], input_paths: dict[str, Path], payload: dict[str, Any]) -> None:
    paths["cache_root"].mkdir(parents=True, exist_ok=True)
    paths["gate_pareto_grid"].parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "factorized_cache_manifest",
        "dataset": args.dataset,
        "model_tag": args.model_tag,
        "stage": "dev_calibration",
        "selector_mode": "factorized",
        "cache_contents": [
            "development_calibration_base_margins",
            "feature_bank_primitive_statistics",
            "surrogate_gate_pareto_grid",
            "surrogate_exact_shortlist",
        ],
        "exact_peak_zA_cached": False,
        "exact_neutral_preference_cached": False,
        "exact_candidate_positions_cached": False,
        "exact_cache_note": (
            "Shortlist mode is CPU-only. Exact peak_zA and neutral preference are computed "
            "during --run-mode exact and the final config can only come from exact-scored rows."
        ),
        "input_paths": {key: str(value) for key, value in input_paths.items() if key != "calibration_root"},
        "calibration_source_count": len(payload["calibration_sources"]),
        "q_group_counts": q_group_counts(payload["q_rows"]),
        "condition_counts": condition_counts(payload["behavior_rows"]),
    }
    write_json(paths["cache_manifest"], manifest)
    write_jsonl(paths["gate_pareto_grid"], payload["gate_rows"])
    write_jsonl(paths["primitive_response_records"], payload["primitive_rows"])
    write_jsonl(paths["surrogate_grid"], payload["surrogate_rows"])
    write_jsonl(paths["exact_shortlist_grid"], payload["shortlist"])


def write_exact_gate_cache(paths: dict[str, Path], records: list[dict[str, Any]]) -> None:
    gate_rows = []
    for record in records:
        gate_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "record_type": "exact_gate_signal_record",
                "grid_row_index": record.get("grid_row_index"),
                "dataset": record.get("dataset"),
                "model_tag": record.get("model_tag"),
                "source_id": record.get("source_id"),
                "template_id": record.get("template_id"),
                "pair_id": record.get("pair_id"),
                "condition": record.get("condition"),
                "q_group": record.get("q_group"),
                "peak_zA": record.get("peak_zA"),
                "tau_A_value": record.get("tau_A_value"),
                "tau_A_quantile": record.get("tau_A_quantile"),
                "neutral_preference_score": record.get("neutral_preference_score"),
                "self_answer_conflict_score": record.get("self_answer_conflict_score"),
                "gate_signal_name": record.get("gate_signal_name"),
                "gate_score_status": record.get("gate_score_status"),
                "candidate_text_source": record.get("candidate_text_source"),
                "uses_correct_answer_for_gate": record.get("uses_correct_answer_for_gate", False),
                "structured_candidate_allowed_for_locator": record.get("structured_candidate_allowed_for_locator"),
                "tau_C": record.get("tau_C"),
                "baseline_margin": record.get("baseline_margin"),
                "target_positions": record.get("target_positions"),
                "target_token_count": record.get("target_token_count"),
                "span_mode": record.get("span_mode"),
                "uses_gold_span_fields": record.get("uses_gold_span_fields"),
            }
        )
    write_jsonl(paths["exact_gate_signal_records"], gate_rows)
    manifest = load_json(paths["cache_manifest"])
    manifest["exact_peak_zA_cached"] = True
    manifest["exact_neutral_preference_cached"] = True
    manifest["exact_candidate_positions_cached"] = True
    manifest["exact_gate_signal_records"] = str(paths["exact_gate_signal_records"])
    if "exact_gate_signal_records" not in manifest["cache_contents"]:
        manifest["cache_contents"].append("exact_gate_signal_records")
    write_json(paths["cache_manifest"], manifest)


def run_exact_scoring(
    *,
    args: argparse.Namespace,
    root: Path,
    input_paths: dict[str, Path],
    paths: dict[str, Path],
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    model_spec = load_model_spec(input_paths["models_config"], args.model_tag, root, allow_placeholder=True)
    scoring_args = argparse.Namespace(**vars(args))
    scoring_args.run_mode = "formal"
    shortlist = payload["shortlist"]
    if args.shortlist_path:
        shortlist = load_jsonl(resolve_path(args.shortlist_path, root))
    if not shortlist:
        raise ValueError("Exact scoring requires a non-empty shortlist.")

    pressure_by_pair, side_by_key = SCORER.load_views_for_dev(root, args.dataset)
    estimated_examples = len(
        SCORER.select_smoke_examples(
            q_rows=payload["q_rows"],
            behavior_rows=payload["behavior_rows"],
            pressure_by_pair=pressure_by_pair,
            side_by_key=side_by_key,
            args=scoring_args,
        )
    )
    estimated_work = estimated_examples * len(shortlist)
    if estimated_work > args.exact_work_threshold and not args.allow_large_exact:
        raise SystemExit(
            "Factorized exact scoring is large "
            f"({len(shortlist)} shortlist rows * {estimated_examples} examples = {estimated_work}). "
            "Re-run with --allow-large-exact or a smaller --shortlist-size."
        )

    scored_rows, records, scoring_meta = SCORER.score_smoke_grid(
        args=scoring_args,
        root=root,
        model_spec=model_spec,
        calibration_sources=payload["calibration_sources"],
        q_rows=payload["q_rows"],
        behavior_rows=payload["behavior_rows"],
        grid_rows=shortlist,
        cue_status=SCORER.cue_only_status(condition_counts(payload["behavior_rows"])),
    )
    scoring_meta.update(
        {
            "selector_mode": "factorized",
            "surrogate_composition_role": "shortlist_generation_only",
            "final_selection_source": "exact_scored_shortlist",
            "estimated_scoring_examples": estimated_examples,
            "estimated_grid_example_products": estimated_work,
            "shortlist_path": str(args.shortlist_path or paths["exact_shortlist_grid"]),
        }
    )
    return scored_rows, records, scoring_meta


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
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--sae-path", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-mode", choices=["shortlist", "exact"], default="shortlist")
    parser.add_argument("--shortlist-size", type=int, default=64)
    parser.add_argument("--min-shortlist-per-band", type=int, default=8)
    parser.add_argument("--shortlist-path", type=Path, help="Optional exact shortlist override for --run-mode exact.")
    parser.add_argument("--include-multilayer-formal", action="store_true")
    parser.add_argument("--max-calibration-sources", type=int, default=0)
    parser.add_argument("--max-pairs-per-group", type=int, default=0)
    parser.add_argument("--max-side-rows-per-condition", type=int, default=0)
    parser.add_argument("--strict-damage-threshold", type=float, default=0.5)
    parser.add_argument("--allow-large-exact", action="store_true")
    parser.add_argument("--exact-work-threshold", type=int, default=250000)
    parser.add_argument("--progress-every-grid", type=int, default=0)
    parser.add_argument(
        "--selection-policy",
        choices=["constrained_score", "q1_max"],
        default="constrained_score",
        help="How to select the final row among feasible exact-scored shortlist configs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    input_paths, resolved = resolve_inputs(args, root)
    calibration_root = resolved["calibration_root"]
    paths = output_paths(calibration_root)
    payload = build_candidate_rows(args, input_paths)
    write_shortlist_outputs(args, paths, input_paths, payload)

    exact_rows: list[dict[str, Any]] | None = None
    exact_records: list[dict[str, Any]] | None = None
    scoring_meta: dict[str, Any] = {}
    final_config_written = False

    if args.run_mode == "exact":
        exact_rows, exact_records, scoring_meta = run_exact_scoring(
            args=args,
            root=root,
            input_paths=input_paths,
            paths=paths,
            payload=payload,
        )
        selected = select_exact_config(exact_rows, args.selection_policy)
        write_jsonl(paths["exact_shortlist_grid"], exact_rows)
        write_jsonl(paths["calibration_grid"], exact_rows)
        write_jsonl(paths["calibration_records"], exact_records)
        write_exact_gate_cache(paths, exact_records)
        if selected is None:
            audit = build_audit(args=args, input_paths=input_paths, payload=payload, exact_rows=exact_rows)
            summary = build_summary(
                args=args,
                paths=paths,
                payload=payload,
                exact_rows=exact_rows,
                exact_records=exact_records,
                final_config_written=False,
            )
            write_json(paths["calibration_summary"], summary)
            write_json(paths["calibration_audit"], audit)
            write_json(paths["factorized_summary"], summary)
            write_json(paths["factorized_audit"], audit)
            raise SystemExit("No feasible exact-scored shortlist config; refused to write lace_base_config.json")

        config_args = argparse.Namespace(**vars(args))
        config_args.run_mode = "formal"
        config = SCORER.build_formal_config(
            args=config_args,
            input_paths=input_paths,
            search_config=payload["search_config"],
            constraints=payload["constraints"],
            grid_rows=exact_rows,
            cue_status=SCORER.cue_only_status(condition_counts(payload["behavior_rows"])),
            scoring_meta=scoring_meta,
            grid_filter_report={
                **payload["grid_filter_report"],
                "selector_mode": "factorized",
                "surrogate_composition_role": "shortlist_generation_only",
                "final_selection_source": "exact_scored_shortlist",
                "shortlist_size": len(exact_rows),
            },
        )
        config.update(
            {
                "record_type": "lace_base_config",
                "selector_mode": "factorized",
                "surrogate_composition_role": "shortlist_generation_only",
                "surrogate_used_for_final_selection": False,
                "final_selection_source": "exact_scored_shortlist",
                "full_grid_used_for_selection": False,
                "exact_scoring_runtime": "internal/calibration_runtime.py",
                "factorized_shortlist_size": len(exact_rows),
                "factorized_shortlist_path": str(paths["exact_shortlist_grid"]),
                "uses_correct_answer_for_gate": False,
                "structured_candidate_allowed_for_locator": True,
            }
        )
        write_json(paths["formal_config"], config)
        final_config_written = True

    audit = build_audit(
        args=args,
        input_paths=input_paths,
        payload=payload,
        exact_rows=exact_rows,
        final_config_written=final_config_written,
    )
    summary = build_summary(
        args=args,
        paths=paths,
        payload=payload,
        exact_rows=exact_rows,
        exact_records=exact_records,
        final_config_written=final_config_written,
    )
    if scoring_meta:
        summary["scoring_meta"] = scoring_meta
    write_json(paths["factorized_summary"], summary)
    write_json(paths["factorized_audit"], audit)
    if args.run_mode == "exact":
        write_json(paths["calibration_summary"], summary)
        write_json(paths["calibration_audit"], audit)
    if audit["status"] != "pass":
        raise SystemExit(f"Factorized calibration audit failed: {audit['issues']}")

    print(f"Wrote {paths['cache_manifest']}")
    print(f"Wrote {paths['gate_pareto_grid']}")
    print(f"Wrote {paths['primitive_response_records']}")
    print(f"Wrote {paths['surrogate_grid']}")
    print(f"Wrote {paths['exact_shortlist_grid']}")
    if args.run_mode == "exact":
        print(f"Wrote {paths['exact_gate_signal_records']}")
    print(f"Wrote {paths['factorized_summary']}")
    print(f"Wrote {paths['factorized_audit']}")
    if final_config_written:
        print(f"Wrote {paths['formal_config']}")


if __name__ == "__main__":
    main()
