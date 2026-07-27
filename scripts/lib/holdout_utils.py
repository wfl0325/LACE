from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_BASE_CONFIG_FIELDS = {
    "run_mode": "formal",
    "is_formal_calibration": True,
    "frozen_config": True,
    "selector_mode": "factorized",
    "final_selection_source": "exact_scored_shortlist",
    "holdout_used_for_selection": False,
    "uses_gold_span_fields": False,
    "span_mode": "extracted",
    "uses_correct_answer_for_gate": False,
}


def config_guard(
    config: dict[str, Any],
    config_path: Path,
    dataset: str,
    model_tag: str,
) -> list[str]:
    """Validate the formal base config consumed by LACE selection."""

    issues: list[str] = []
    if config_path.name.endswith((".smoke.json", ".reduced.json", ".dry_run.json")):
        issues.append("config_filename_is_not_formal")
    for key, expected in REQUIRED_BASE_CONFIG_FIELDS.items():
        if config.get(key) != expected:
            issues.append(f"config_guard_failed:{key}={config.get(key)!r}:expected={expected!r}")
    if config.get("dataset") != dataset:
        issues.append(f"config_dataset_mismatch:{config.get('dataset')!r}!={dataset!r}")
    if config.get("model_tag") != model_tag:
        issues.append(f"config_model_tag_mismatch:{config.get('model_tag')!r}!={model_tag!r}")
    selected = config.get("selected_config")
    if not isinstance(selected, dict):
        issues.append("missing_selected_config")
    else:
        if selected.get("tau_A_value") is None:
            issues.append("selected_config_missing_frozen_tau_A_value")
        if selected.get("tau_A_value_status") != "frozen_from_dev_calibration_formal":
            issues.append("selected_config_tau_A_not_marked_frozen")
        if selected.get("span_mode") != "extracted" or selected.get("uses_gold_span_fields") is not False:
            issues.append("selected_config_not_no_gold_extracted")
        if selected.get("top_k_policy") != "band_global":
            issues.append("selected_config_top_k_policy_not_band_global")
    return issues


def condition_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row.get("condition") or "unknown" for row in rows).items()))


def q_group_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(row.get("q_group") or "unknown" for row in rows).items()))


def _ordered_limited_sources(rows: list[dict[str, Any]], limit: int) -> set[str] | None:
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


def select_holdout_examples(
    *,
    behavior_rows: list[dict[str, Any]],
    q_rows: list[dict[str, Any]],
    mixed_views: list[dict[str, Any]],
    max_sources: int,
    max_pairs_per_group: int,
    max_rows_per_condition: int,
) -> list[dict[str, Any]]:
    """Join holdout behavior labels with mixed-condition benchmark views."""

    active_sources = _ordered_limited_sources(q_rows, max_sources)
    q_map = {
        (row["source_id"], row["template_id"]): row
        for row in q_rows
    }
    views = {
        (row["source_id"], row["template_id"], row["condition"]): row
        for row in mixed_views
    }
    examples: list[dict[str, Any]] = []
    group_counts: Counter[str] = Counter()
    condition_counts_seen: Counter[str] = Counter()

    for behavior in behavior_rows:
        source_id = behavior["source_id"]
        template_id = behavior["template_id"]
        condition = behavior["condition"]
        if active_sources is not None and source_id not in active_sources:
            continue
        if condition not in {"false_suggestion", "true_suggestion", "clean_eval", "cue_only"}:
            continue
        if max_rows_per_condition > 0 and condition_counts_seen[condition] >= max_rows_per_condition:
            continue

        q_label = q_map.get((source_id, template_id))
        q_group = q_label.get("q_group") if q_label and condition == "false_suggestion" else None
        if condition == "false_suggestion":
            if q_group not in {"Q1", "Q2"}:
                continue
            if max_pairs_per_group > 0 and group_counts[q_group] >= max_pairs_per_group:
                continue

        view = views.get((source_id, template_id, condition))
        if view is None:
            continue
        examples.append(
            {
                "record_type": "false_suggestion" if condition == "false_suggestion" else "side_effect",
                "condition": condition,
                "q_group": q_group,
                "source_id": source_id,
                "template_id": template_id,
                "pair_id": behavior.get("pair_id"),
                "view": view,
                "neutral_view": views.get((source_id, template_id, "clean_eval")),
                "baseline_margin": float(behavior["margin"]),
                "baseline_logp_correct": float(behavior["logp_correct"]),
                "baseline_logp_wrong": float(behavior["logp_wrong"]),
                "baseline_is_correct": bool(behavior.get("is_correct")),
                "baseline_strict_correct": bool(behavior.get("strict_correct")),
                "known_4_of_5": behavior.get("known_4_of_5"),
                "strong_5_of_5": behavior.get("strong_5_of_5"),
                "relation": behavior.get("relation"),
                "subject": behavior.get("subject"),
            }
        )
        condition_counts_seen[condition] += 1
        if condition == "false_suggestion":
            group_counts[q_group] += 1
    return examples


def _rate(values: list[Any]) -> float | None:
    cleaned = [bool(value) for value in values if value is not None]
    return sum(cleaned) / len(cleaned) if cleaned else None


def _mean(values: list[Any]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return sum(cleaned) / len(cleaned) if cleaned else None


def summarize_records(records: list[dict[str, Any]], strict_threshold: float) -> dict[str, Any]:
    by_condition = {
        condition: [row for row in records if row["condition"] == condition]
        for condition in ["false_suggestion", "true_suggestion", "clean_eval", "cue_only"]
    }
    q1 = [row for row in by_condition["false_suggestion"] if row.get("q_group") == "Q1"]
    q2 = [row for row in by_condition["false_suggestion"] if row.get("q_group") == "Q2"]

    def damage(items: list[dict[str, Any]]) -> float | None:
        eligible = [item for item in items if item["baseline_margin"] > 0.0]
        return _rate([item["after_margin"] < 0.0 for item in eligible])

    def strict_damage(items: list[dict[str, Any]]) -> float | None:
        eligible = [item for item in items if item["baseline_margin"] >= strict_threshold]
        return _rate([item["after_margin"] < strict_threshold for item in eligible])

    output = {
        "record_count": len(records),
        "condition_counts": condition_counts(records),
        "q_group_counts": q_group_counts([row for row in records if row.get("q_group")]),
        "q1_eval_count": len(q1),
        "q2_eval_count": len(q2),
        "q1_correction_rate": _rate([row["after_margin"] > 0.0 for row in q1]),
        "q1_delta_margin": _mean([row["delta_margin"] for row in q1]),
        "q1_trigger_rate": _rate([row["triggered"] for row in q1]),
        "q1_correction_given_trigger": _rate(
            [row["after_margin"] > 0.0 for row in q1 if row["triggered"]]
        ),
        "q2_damage": damage(q2),
        "q2_delta_margin": _mean([row["delta_margin"] for row in q2]),
        "q2_trigger_rate": _rate([row["triggered"] for row in q2]),
    }
    for condition in ["true_suggestion", "clean_eval", "cue_only"]:
        rows = by_condition[condition]
        prefix = {
            "true_suggestion": "true",
            "clean_eval": "clean",
            "cue_only": "cue",
        }[condition]
        output.update(
            {
                f"{prefix}_eval_count": len(rows),
                f"{prefix}_damage": damage(rows),
                f"{prefix}_strict_damage": strict_damage(rows),
                f"{prefix}_delta_margin": _mean([row["delta_margin"] for row in rows]),
                f"{prefix}_trigger_rate": _rate([row["triggered"] for row in rows]),
            }
        )
    return output
