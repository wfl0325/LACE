#!/usr/bin/env python3
"""Validate LACE dataset files without loading a model."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lib.dataset_registry import load_dataset_stage, validate_dataset_stage  # noqa: E402
from lib.io_utils import load_json, load_jsonl, repo_root, resolve_path, write_json  # noqa: E402


COMMON_REQUIRED = {
    "source_id",
    "pair_id",
    "template_id",
    "mc_prompt",
    "question_text",
    "correct_option",
    "wrong_option",
    "suggested_option",
    "relation",
}
FALSE_REQUIRED = COMMON_REQUIRED | {"trigger_condition"}
SIDE_REQUIRED = COMMON_REQUIRED | {"trigger_condition"}
MIXED_REQUIRED = COMMON_REQUIRED | {"condition"}
KNOWN_REQUIRED = COMMON_REQUIRED | {"trigger_condition"}
FALSE_CONDITIONS = {"control", "pressure"}
SIDE_CONDITIONS = {"clean", "true_suggestion"}
MIXED_CONDITIONS = {"clean_eval", "false_suggestion", "true_suggestion", "cue_only"}


def source_ids(rows: Iterable[dict]) -> set[str]:
    return {str(row["source_id"]) for row in rows if "source_id" in row}


def add_missing_field_errors(errors: list[str], label: str, rows: list[dict], required: set[str]) -> None:
    for index, row in enumerate(rows, start=1):
        missing = sorted(required - set(row))
        if missing:
            errors.append(f"{label}: row {index} missing fields {missing}")


def validate_conditions(
    errors: list[str],
    label: str,
    rows: list[dict],
    field: str,
    allowed: set[str],
) -> Counter:
    counts = Counter(str(row.get(field, "")) for row in rows)
    unknown = sorted(condition for condition in counts if condition not in allowed)
    if unknown:
        errors.append(f"{label}: unknown {field} values {unknown}; allowed={sorted(allowed)}")
    return counts


def validate_false_pairs(errors: list[str], rows: list[dict]) -> None:
    by_pair: dict[str, set[str]] = defaultdict(set)
    source_by_pair: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_pair[str(row.get("pair_id"))].add(str(row.get("trigger_condition")))
        source_by_pair[str(row.get("pair_id"))].add(str(row.get("source_id")))
    for pair_id, conditions in sorted(by_pair.items()):
        if conditions != FALSE_CONDITIONS:
            errors.append(
                f"false_suggestion: pair_id={pair_id} has conditions={sorted(conditions)}, "
                "expected ['control', 'pressure']"
            )
        if len(source_by_pair[pair_id]) != 1:
            errors.append(f"false_suggestion: pair_id={pair_id} spans multiple source_id values")


def validate_stage(root: Path, datasets_config: Path, dataset: str, stage_name: str) -> dict:
    stage = load_dataset_stage(datasets_config, dataset, stage_name, root)
    errors: list[str] = []
    missing = validate_dataset_stage(stage)
    if missing:
        errors.extend(f"missing path: {item}" for item in missing)
        return {
            "dataset": dataset,
            "stage": stage_name,
            "status": "fail",
            "errors": errors,
        }

    known_rows = load_jsonl(stage.known_fact_clean)
    false_rows = load_jsonl(stage.false_suggestion)
    side_rows = load_jsonl(stage.side_effect)
    construction = load_json(stage.construction_summary)
    mixed_rows = load_jsonl(stage.mixed_eval) if stage.mixed_eval else []

    add_missing_field_errors(errors, "known_fact_clean", known_rows, KNOWN_REQUIRED)
    add_missing_field_errors(errors, "false_suggestion", false_rows, FALSE_REQUIRED)
    add_missing_field_errors(errors, "side_effect", side_rows, SIDE_REQUIRED)
    if stage.mixed_eval:
        add_missing_field_errors(errors, "mixed_eval", mixed_rows, MIXED_REQUIRED)

    known_conditions = validate_conditions(
        errors, "known_fact_clean", known_rows, "trigger_condition", {"clean"}
    )
    false_conditions = validate_conditions(
        errors, "false_suggestion", false_rows, "trigger_condition", FALSE_CONDITIONS
    )
    side_conditions = validate_conditions(
        errors, "side_effect", side_rows, "trigger_condition", SIDE_CONDITIONS
    )
    mixed_conditions = Counter()
    if stage.mixed_eval:
        mixed_conditions = validate_conditions(
            errors, "mixed_eval", mixed_rows, "condition", MIXED_CONDITIONS
        )
        missing_mixed = sorted(MIXED_CONDITIONS - set(mixed_conditions))
        if missing_mixed:
            errors.append(f"mixed_eval: missing conditions {missing_mixed}")

    validate_false_pairs(errors, false_rows)

    all_sources = source_ids(known_rows) | source_ids(false_rows) | source_ids(side_rows) | source_ids(mixed_rows)
    if len(all_sources) != stage.source_count:
        errors.append(
            f"source_count mismatch: registry={stage.source_count}, observed={len(all_sources)}"
        )
    summary_source_count = construction.get("source_count")
    if summary_source_count is not None and int(summary_source_count) != stage.source_count:
        errors.append(
            f"construction_summary source_count={summary_source_count}, registry={stage.source_count}"
        )

    return {
        "dataset": dataset,
        "stage": stage_name,
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "paths": {
            "known_fact_clean": str(stage.known_fact_clean),
            "false_suggestion": str(stage.false_suggestion),
            "side_effect": str(stage.side_effect),
            "mixed_eval": str(stage.mixed_eval) if stage.mixed_eval else None,
            "construction_summary": str(stage.construction_summary),
        },
        "row_counts": {
            "known_fact_clean": len(known_rows),
            "false_suggestion": len(false_rows),
            "side_effect": len(side_rows),
            "mixed_eval": len(mixed_rows),
        },
        "source_count": len(all_sources),
        "condition_counts": {
            "known_fact_clean": dict(known_conditions),
            "false_suggestion": dict(false_conditions),
            "side_effect": dict(side_conditions),
            "mixed_eval": dict(mixed_conditions),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--stage", choices=["dev", "holdout", "all"], default="all")
    parser.add_argument("--root", default=str(repo_root()))
    parser.add_argument("--datasets-config", default="configs/datasets.json")
    parser.add_argument("--audit-output", default=None)
    args = parser.parse_args()

    root = resolve_path(args.root)
    datasets_config = resolve_path(args.datasets_config, root)
    stages = ["dev", "holdout"] if args.stage == "all" else [args.stage]
    audits = [validate_stage(root, datasets_config, args.dataset, stage) for stage in stages]

    if args.stage == "all":
        dev = load_dataset_stage(datasets_config, args.dataset, "dev", root)
        holdout = load_dataset_stage(datasets_config, args.dataset, "holdout", root)
        dev_sources = source_ids(load_jsonl(dev.false_suggestion))
        holdout_sources = source_ids(load_jsonl(holdout.false_suggestion))
        overlap = sorted(dev_sources & holdout_sources)
        audits.append(
            {
                "dataset": args.dataset,
                "stage": "source_disjointness",
                "status": "pass" if not overlap else "fail",
                "errors": [] if not overlap else [f"dev/holdout overlap: {overlap[:10]}"],
                "overlap_count": len(overlap),
            }
        )

    status = "pass" if all(audit["status"] == "pass" for audit in audits) else "fail"
    payload = {
        "schema_version": "lace_data_validation_v1",
        "dataset": args.dataset,
        "stage": args.stage,
        "status": status,
        "audits": audits,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.audit_output:
        write_json(resolve_path(args.audit_output, root), payload)
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
