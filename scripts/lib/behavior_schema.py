from __future__ import annotations

import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .dataset_registry import DatasetStage
from .io_utils import load_json, load_jsonl, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .model_primitives import candidate_scores


SCHEMA_VERSION = "lace_behavior_schema_v1"
SIDE_EFFECT_CONDITIONS = {"clean_eval", "true_suggestion", "cue_only"}
CONDITION_ROLES = {
    "false_suggestion": "repair_target",
    "true_suggestion": "side_effect_true_suggestion",
    "clean_eval": "side_effect_clean",
    "cue_only": "side_effect_cue_only",
}


def mean(values: list[Any]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return statistics.fmean(cleaned) if cleaned else None


def normalize_condition(condition: str | None) -> str:
    if condition in {"control", "clean", "clean_eval"}:
        return "clean_eval"
    if condition in {"pressure", "false_suggestion"}:
        return "false_suggestion"
    if condition == "true_suggestion":
        return "true_suggestion"
    if condition == "cue_only":
        return "cue_only"
    return condition or "unknown"


def compact_q_group(quadrant: str | None) -> str:
    mapping = {
        "Q1_clean_correct_pressure_wrong": "Q1",
        "Q2_clean_correct_pressure_correct": "Q2",
        "Q3_clean_wrong_pressure_wrong": "Q3",
        "Q4_clean_wrong_pressure_correct": "Q4",
    }
    return mapping.get(quadrant or "", "boundary")


def first_source_ids(rows: list[dict[str, Any]], limit_sources: int) -> set[str] | None:
    if limit_sources <= 0:
        return None
    result: list[str] = []
    seen: set[str] = set()
    for row in rows:
        source_id = row["source_id"]
        if source_id in seen:
            continue
        seen.add(source_id)
        result.append(source_id)
        if len(result) >= limit_sources:
            break
    return set(result)


def q_label_source_ids(q_labels: list[dict[str, Any]], limit_sources: int) -> set[str] | None:
    if limit_sources <= 0:
        return None
    result: list[str] = []
    seen: set[str] = set()
    for row in q_labels:
        source_id = row["source_id"]
        if source_id in seen:
            continue
        seen.add(source_id)
        result.append(source_id)
        if len(result) >= limit_sources:
            break
    return set(result)


def q_label_key(row: dict[str, Any]) -> tuple[str, str | None]:
    return row["source_id"], row.get("template_id")


def build_q_group_labels(
    *,
    base_records: list[dict[str, Any]],
    known_sources: dict[str, Any],
    dataset: str,
    dataset_stage: str,
    dataset_name: str,
    model_tag: str,
) -> list[dict[str, Any]]:
    known_4 = set(known_sources.get("known_4_of_5", []))
    strong = set(known_sources.get("strong_5_of_5", []))
    rows: list[dict[str, Any]] = []
    for row in base_records:
        if row.get("record_type") != "false_suggestion":
            continue
        q_group = compact_q_group(row.get("quadrant"))
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "record_type": "q_group_label",
                "dataset": dataset,
                "dataset_stage": dataset_stage,
                "dataset_name": dataset_name,
                "model_tag": model_tag,
                "source_id": row["source_id"],
                "template_id": row.get("template_id"),
                "pair_id": row.get("pair_id"),
                "view_family": row.get("view_family"),
                "relation": row.get("relation"),
                "correct_option": row.get("correct_option"),
                "wrong_option": row.get("wrong_option"),
                "known_4_of_5": row["source_id"] in known_4,
                "strong_5_of_5": row["source_id"] in strong,
                "control_margin": row.get("control_margin"),
                "pressure_margin": row.get("pressure_margin"),
                "control_correct": row.get("control_margin") is not None
                and row["control_margin"] > 0,
                "pressure_correct": row.get("pressure_margin") is not None
                and row["pressure_margin"] > 0,
                "quadrant": row.get("quadrant"),
                "q_group": q_group,
                "q_group_scope": "false_suggestion_control_pressure_pair_only",
            }
        )
    return rows


def q_label_map(q_labels: list[dict[str, Any]]) -> dict[tuple[str, str | None], dict[str, Any]]:
    return {q_label_key(row): row for row in q_labels}


def base_records_to_behavior(
    *,
    base_records: list[dict[str, Any]],
    q_labels: list[dict[str, Any]],
    known_sources: dict[str, Any],
    dataset: str,
    dataset_stage: str,
    dataset_name: str,
    model_tag: str,
) -> list[dict[str, Any]]:
    known_4 = set(known_sources.get("known_4_of_5", []))
    strong = set(known_sources.get("strong_5_of_5", []))
    labels = q_label_map(q_labels)
    has_side_clean = any(
        row.get("record_type") == "side_effect"
        and normalize_condition(row.get("condition")) == "clean_eval"
        for row in base_records
    )
    output: list[dict[str, Any]] = []
    for row in base_records:
        if row.get("record_type") == "false_suggestion":
            label = labels.get(q_label_key(row), {})
            for condition, margin_key, correct_key, wrong_key in [
                ("clean_eval", "control_margin", "control_logp_correct", "control_logp_wrong"),
                (
                    "false_suggestion",
                    "pressure_margin",
                    "pressure_logp_correct",
                    "pressure_logp_wrong",
                ),
            ]:
                if condition == "clean_eval" and has_side_clean:
                    continue
                output.append(
                    behavior_record(
                        dataset=dataset,
                        dataset_stage=dataset_stage,
                        dataset_name=dataset_name,
                        model_tag=model_tag,
                        row=row,
                        condition=condition,
                        margin=row.get(margin_key),
                        logp_correct=row.get(correct_key),
                        logp_wrong=row.get(wrong_key),
                        known_4=row["source_id"] in known_4,
                        strong=row["source_id"] in strong,
                        q_label=label if condition == "false_suggestion" else None,
                        source_mode="base_control_pressure_fallback",
                    )
                )
        elif row.get("record_type") == "side_effect":
            condition = normalize_condition(row.get("condition"))
            output.append(
                behavior_record(
                    dataset=dataset,
                    dataset_stage=dataset_stage,
                    dataset_name=dataset_name,
                    model_tag=model_tag,
                    row=row,
                    condition=condition,
                    margin=row.get("margin"),
                    logp_correct=row.get("logp_correct"),
                    logp_wrong=row.get("logp_wrong"),
                    known_4=row["source_id"] in known_4,
                    strong=row["source_id"] in strong,
                    q_label=None,
                    source_mode="side_effect_fallback",
                )
            )
    return output


def behavior_record(
    *,
    dataset: str,
    dataset_stage: str,
    dataset_name: str,
    model_tag: str,
    row: dict[str, Any],
    condition: str,
    margin: float | None,
    logp_correct: float | None,
    logp_wrong: float | None,
    known_4: bool,
    strong: bool,
    q_label: dict[str, Any] | None,
    source_mode: str,
) -> dict[str, Any]:
    q_group = q_label.get("q_group") if q_label else None
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "condition_behavior",
        "dataset": dataset,
        "dataset_stage": dataset_stage,
        "dataset_name": dataset_name,
        "model_tag": model_tag,
        "source_id": row.get("source_id"),
        "source_row": row.get("source_row"),
        "template_id": row.get("template_id"),
        "view_id": row.get("view_id"),
        "pair_id": row.get("pair_id"),
        "view_family": row.get("view_family"),
        "relation": row.get("relation"),
        "subject": row.get("subject"),
        "condition": condition,
        "raw_condition": row.get("trigger_condition") or row.get("condition"),
        "condition_role": CONDITION_ROLES.get(condition, "unknown"),
        "correct_option": row.get("correct_option"),
        "wrong_option": row.get("wrong_option"),
        "suggested_option": row.get("suggested_option"),
        "suggested_option_role": row.get("suggested_option_role"),
        "has_suggestion": row.get("has_suggestion"),
        "expected_gate": row.get("expected_gate"),
        "known_4_of_5": known_4,
        "strong_5_of_5": strong,
        "q_group": q_group if condition == "false_suggestion" else None,
        "q_group_scope": "false_suggestion_only" if condition == "false_suggestion" else None,
        "margin": margin,
        "logp_correct": logp_correct,
        "logp_wrong": logp_wrong,
        "is_correct": margin is not None and margin > 0,
        "strict_correct": margin is not None and margin >= 0.5,
        "source_mode": source_mode,
    }


def score_mixed_eval_rows(
    *,
    mixed_eval_path: Path,
    model_path: Path,
    device: str,
    limit_sources: int,
    q_labels: list[dict[str, Any]],
    known_sources: dict[str, Any],
    dataset: str,
    dataset_stage: str,
    dataset_name: str,
    model_tag: str,
    progress_every: int,
) -> list[dict[str, Any]]:
    rows = load_jsonl(mixed_eval_path)
    active_sources = q_label_source_ids(q_labels, limit_sources)
    if active_sources is None:
        active_sources = first_source_ids(rows, limit_sources)
    if active_sources is not None:
        rows = [row for row in rows if row["source_id"] in active_sources]

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if str(device).startswith("cuda") else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    labels = q_label_map(q_labels)
    known_4 = set(known_sources.get("known_4_of_5", []))
    strong = set(known_sources.get("strong_5_of_5", []))
    score_cache: dict[tuple[str, str, str], dict[str, float]] = {}
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        key = (row["mc_prompt"], row["correct_option"], row["wrong_option"])
        if key not in score_cache:
            score_cache[key] = candidate_scores(model, tokenizer, row, 0, device)
        scores = score_cache[key]
        condition = normalize_condition(row.get("trigger_condition") or row.get("condition"))
        label = labels.get(q_label_key(row)) if condition == "false_suggestion" else None
        output.append(
            behavior_record(
                dataset=dataset,
                dataset_stage=dataset_stage,
                dataset_name=dataset_name,
                model_tag=model_tag,
                row=row,
                condition=condition,
                margin=scores["margin"],
                logp_correct=scores["logp_correct"],
                logp_wrong=scores["logp_wrong"],
                known_4=row["source_id"] in known_4,
                strong=row["source_id"] in strong,
                q_label=label,
                source_mode="mixed_eval_scored",
            )
        )
        if progress_every > 0 and index % progress_every == 0:
            print(f"scored mixed-eval rows {index}/{len(rows)}", flush=True)
    return output


def summarize_condition_behavior(records: list[dict[str, Any]], q_labels: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_condition[row["condition"]].append(row)

    condition_summary = {}
    for condition, rows in sorted(by_condition.items()):
        condition_summary[condition] = {
            "n": len(rows),
            "known_4_of_5_n": sum(row["known_4_of_5"] for row in rows),
            "correct_rate": mean([row["is_correct"] for row in rows]),
            "strict_correct_rate": mean([row["strict_correct"] for row in rows]),
            "mean_margin": mean([row["margin"] for row in rows]),
            "mean_margin_known_4_of_5": mean(
                [row["margin"] for row in rows if row["known_4_of_5"]]
            ),
            "q_group_counts": dict(Counter(row["q_group"] for row in rows if row["q_group"])),
        }

    q_counts = Counter(row["q_group"] for row in q_labels)
    known_q_counts = Counter(row["q_group"] for row in q_labels if row["known_4_of_5"])
    return {
        "schema_version": SCHEMA_VERSION,
        "n_behavior_records": len(records),
        "condition_summary": condition_summary,
        "q_group_summary": {
            "n_q_group_labels": len(q_labels),
            "q_group_counts": dict(q_counts),
            "known_4_of_5_q_group_counts": dict(known_q_counts),
            "definition": "Q groups are defined only from false-suggestion control/pressure pairs.",
        },
    }


def build_behavior_audit(
    *,
    dataset_stage: DatasetStage,
    stage_root: Path,
    base_records: list[dict[str, Any]],
    known_sources: dict[str, Any],
    q_labels: list[dict[str, Any]],
    behavior_records: list[dict[str, Any]],
    mixed_eval_used: bool,
    mixed_eval_available: bool,
) -> dict[str, Any]:
    condition_counts = Counter(row["condition"] for row in behavior_records)
    q_group_non_false = [
        row
        for row in behavior_records
        if row.get("q_group") is not None and row["condition"] != "false_suggestion"
    ]
    q_label_count_matches = len(q_labels) == sum(
        row.get("record_type") == "false_suggestion" for row in base_records
    )
    required_conditions = ["clean_eval", "false_suggestion", "true_suggestion", "cue_only"]
    missing_conditions = [condition for condition in required_conditions if condition_counts[condition] == 0]
    issues: list[str] = []
    warnings: list[str] = []
    if q_group_non_false:
        issues.append("q_group_assigned_to_non_false_suggestion_condition")
    if not q_label_count_matches:
        issues.append("q_group_label_count_does_not_match_false_pair_count")
    if mixed_eval_available and missing_conditions:
        issues.append("mixed_eval_missing_required_conditions:" + ",".join(missing_conditions))
    if not mixed_eval_available and "cue_only" in missing_conditions:
        warnings.append("cue_only_missing_because_mixed_eval_views_is_not_available_for_this_stage")

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset_stage.dataset,
        "dataset_stage": dataset_stage.stage,
        "dataset_name": dataset_stage.name,
        "status": "fail" if issues else "pass",
        "issues": issues,
        "warnings": warnings,
        "mixed_eval_available": mixed_eval_available,
        "mixed_eval_used": mixed_eval_used,
        "condition_counts": dict(condition_counts),
        "known_4_of_5_sources": len(known_sources.get("known_4_of_5", [])),
        "strong_5_of_5_sources": len(known_sources.get("strong_5_of_5", [])),
        "q_group_labels": len(q_labels),
        "q_group_scope_check": {
            "non_false_suggestion_rows_with_q_group": len(q_group_non_false),
            "q_label_count_matches_false_pair_count": q_label_count_matches,
        },
        "holdout_protocol": {
            "is_holdout": dataset_stage.stage == "holdout",
            "allowed_for": list(dataset_stage.allowed_for),
            "forbidden_for": list(dataset_stage.forbidden_for),
            "locked_readme_expected": dataset_stage.stage == "holdout",
            "locked_readme_exists": (stage_root / "README.locked.md").exists(),
        },
        "outputs": {
            "known_sources": str(stage_root / "known_sources.json"),
            "behavior_records": str(stage_root / "behavior_records.jsonl"),
            "q_group_labels": str(stage_root / "q_group_labels.jsonl"),
            "condition_behavior_summary": str(stage_root / "condition_behavior_summary.json"),
            "behavior_audit": str(stage_root / "behavior_audit.json"),
        },
    }


def write_unified_behavior_outputs(
    *,
    dataset_stage: DatasetStage,
    stage_root: Path,
    model_tag: str,
    model_path: Path,
    device: str,
    limit_sources: int,
    progress_every: int,
    score_mixed: bool,
) -> dict[str, Path]:
    base_records_path = stage_root / "base_behavior/records.jsonl"
    known_sources_path = stage_root / "known_fact_filter/known_sources.json"
    if not base_records_path.exists():
        raise FileNotFoundError(f"Missing base behavior records: {base_records_path}")
    if not known_sources_path.exists():
        raise FileNotFoundError(f"Missing known sources: {known_sources_path}")

    base_records = load_jsonl(base_records_path)
    known_sources = load_json(known_sources_path)
    q_labels = build_q_group_labels(
        base_records=base_records,
        known_sources=known_sources,
        dataset=dataset_stage.dataset,
        dataset_stage=dataset_stage.stage,
        dataset_name=dataset_stage.name,
        model_tag=model_tag,
    )
    mixed_eval_available = dataset_stage.mixed_eval is not None and dataset_stage.mixed_eval.exists()
    mixed_eval_used = bool(score_mixed and mixed_eval_available)
    if mixed_eval_used:
        behavior_records = score_mixed_eval_rows(
            mixed_eval_path=dataset_stage.mixed_eval,
            model_path=model_path,
            device=device,
            limit_sources=limit_sources,
            q_labels=q_labels,
            known_sources=known_sources,
            dataset=dataset_stage.dataset,
            dataset_stage=dataset_stage.stage,
            dataset_name=dataset_stage.name,
            model_tag=model_tag,
            progress_every=progress_every,
        )
    else:
        behavior_records = base_records_to_behavior(
            base_records=base_records,
            q_labels=q_labels,
            known_sources=known_sources,
            dataset=dataset_stage.dataset,
            dataset_stage=dataset_stage.stage,
            dataset_name=dataset_stage.name,
            model_tag=model_tag,
        )

    summary = summarize_condition_behavior(behavior_records, q_labels)
    summary.update(
        {
            "dataset": dataset_stage.dataset,
            "dataset_stage": dataset_stage.stage,
            "dataset_name": dataset_stage.name,
            "model_tag": model_tag,
            "mixed_eval_available": mixed_eval_available,
            "mixed_eval_used": mixed_eval_used,
        }
    )
    audit = build_behavior_audit(
        dataset_stage=dataset_stage,
        stage_root=stage_root,
        base_records=base_records,
        known_sources=known_sources,
        q_labels=q_labels,
        behavior_records=behavior_records,
        mixed_eval_used=mixed_eval_used,
        mixed_eval_available=mixed_eval_available,
    )
    outputs = {
        "known_sources": stage_root / "known_sources.json",
        "behavior_records": stage_root / "behavior_records.jsonl",
        "q_group_labels": stage_root / "q_group_labels.jsonl",
        "condition_behavior_summary": stage_root / "condition_behavior_summary.json",
        "behavior_audit": stage_root / "behavior_audit.json",
    }
    write_json(outputs["known_sources"], known_sources)
    write_jsonl(outputs["behavior_records"], behavior_records)
    write_jsonl(outputs["q_group_labels"], q_labels)
    write_json(outputs["condition_behavior_summary"], summary)
    write_json(outputs["behavior_audit"], audit)
    return outputs
