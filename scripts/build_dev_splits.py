#!/usr/bin/env python3
"""Build source-level discovery/calibration splits for development data only.

This entry point consumes the model-dependent outputs from
``prepare_behavior.py --stage dev`` and creates source-disjoint development
splits. It never reads holdout data and never performs feature discovery,
calibration, or method selection.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lib.dataset_registry import load_dataset_stage, validate_dataset_stage  # noqa: E402
from lib.io_utils import load_json, load_jsonl, repo_root, resolve_path, write_json  # noqa: E402


SCHEMA_VERSION = "lace_dev_split_v1"


def default_behavior_root(output_root: Path, dataset: str, model_tag: str) -> Path:
    return output_root / dataset / model_tag / "dev_behavior"


def known_sources_path(behavior_root: Path) -> Path:
    root_level = behavior_root / "known_sources.json"
    if root_level.exists():
        return root_level
    nested = behavior_root / "known_fact_filter/known_sources.json"
    if nested.exists():
        return nested
    return root_level


def ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def source_profile(q_counts: Counter[str]) -> str:
    has_q1 = q_counts.get("Q1", 0) > 0
    has_q2 = q_counts.get("Q2", 0) > 0
    if has_q1 and has_q2:
        return "mixed_q1_q2"
    if has_q1:
        return "has_q1"
    if has_q2:
        return "has_q2"
    return "no_q1_q2_boundary"


def build_source_rows(
    *,
    q_labels: list[dict[str, Any]],
    known_sources: dict[str, Any],
    source_scope: str,
) -> list[dict[str, Any]]:
    known_4 = set(known_sources.get("known_4_of_5", []))
    strong = set(known_sources.get("strong_5_of_5", []))
    source_order = ordered_unique([row["source_id"] for row in q_labels])
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in q_labels:
        by_source[row["source_id"]].append(row)

    rows: list[dict[str, Any]] = []
    for source_id in source_order:
        if source_scope == "known_4_of_5" and source_id not in known_4:
            continue
        source_labels = by_source[source_id]
        q_counts = Counter(row.get("q_group") or "unknown" for row in source_labels)
        first = source_labels[0]
        rows.append(
            {
                "source_id": source_id,
                "source_scope": source_scope,
                "known_4_of_5": source_id in known_4,
                "strong_5_of_5": source_id in strong,
                "q_label_count": len(source_labels),
                "q_group_counts": dict(sorted(q_counts.items())),
                "q_profile": source_profile(q_counts),
                "relation": first.get("relation"),
                "view_family": first.get("view_family"),
            }
        )
    return rows


def stratified_split(
    sources: list[dict[str, Any]],
    *,
    discovery_fraction: float,
    split_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < discovery_fraction < 1.0:
        raise ValueError("--discovery-fraction must be between 0 and 1")
    if len(sources) < 2:
        raise ValueError("Need at least two eligible development sources to split")

    rng = random.Random(split_seed)
    by_profile: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sources:
        by_profile[row["q_profile"]].append(row)

    discovery: list[dict[str, Any]] = []
    calibration: list[dict[str, Any]] = []
    for profile in sorted(by_profile):
        bucket = list(by_profile[profile])
        rng.shuffle(bucket)
        if len(bucket) == 1:
            target = 1 if len(discovery) <= len(calibration) else 0
        else:
            target = round(len(bucket) * discovery_fraction)
            target = min(max(target, 1), len(bucket) - 1)
        discovery.extend(bucket[:target])
        calibration.extend(bucket[target:])

    discovery.sort(key=lambda row: row["source_id"])
    calibration.sort(key=lambda row: row["source_id"])
    if not discovery or not calibration:
        raise ValueError("Split produced an empty discovery or calibration set")
    return discovery, calibration


def summarize_sources(rows: list[dict[str, Any]]) -> dict[str, Any]:
    q_profile_counts = Counter(row["q_profile"] for row in rows)
    q_group_totals: Counter[str] = Counter()
    relation_counts = Counter(row.get("relation") or "unknown" for row in rows)
    known_count = 0
    strong_count = 0
    for row in rows:
        known_count += int(row["known_4_of_5"])
        strong_count += int(row["strong_5_of_5"])
        q_group_totals.update(row["q_group_counts"])
    return {
        "n_sources": len(rows),
        "known_4_of_5_sources": known_count,
        "strong_5_of_5_sources": strong_count,
        "q_profile_counts": dict(sorted(q_profile_counts.items())),
        "q_group_label_totals": dict(sorted(q_group_totals.items())),
        "relation_counts": dict(sorted(relation_counts.items())),
    }


def relation_distribution_audit(
    discovery: list[dict[str, Any]],
    calibration: list[dict[str, Any]],
    *,
    max_relation_proportion_diff: float,
) -> dict[str, Any]:
    discovery_counts = Counter(row.get("relation") or "unknown" for row in discovery)
    calibration_counts = Counter(row.get("relation") or "unknown" for row in calibration)
    relations = sorted(set(discovery_counts) | set(calibration_counts))
    discovery_total = max(1, len(discovery))
    calibration_total = max(1, len(calibration))
    rows = []
    max_abs_diff = 0.0
    for relation in relations:
        discovery_rate = discovery_counts[relation] / discovery_total
        calibration_rate = calibration_counts[relation] / calibration_total
        abs_diff = abs(discovery_rate - calibration_rate)
        max_abs_diff = max(max_abs_diff, abs_diff)
        rows.append(
            {
                "relation": relation,
                "discovery_count": discovery_counts[relation],
                "calibration_count": calibration_counts[relation],
                "discovery_fraction": discovery_rate,
                "calibration_fraction": calibration_rate,
                "abs_fraction_diff": abs_diff,
            }
        )
    rows.sort(key=lambda row: (-row["abs_fraction_diff"], row["relation"]))
    return {
        "max_abs_fraction_diff": max_abs_diff,
        "max_allowed_abs_fraction_diff": max_relation_proportion_diff,
        "severe_imbalance": max_abs_diff > max_relation_proportion_diff,
        "top_differences": rows[:20],
        "discovery_relation_counts": dict(sorted(discovery_counts.items())),
        "calibration_relation_counts": dict(sorted(calibration_counts.items())),
    }


def build_audit(
    *,
    dataset_stage_name: str,
    behavior_root: Path,
    q_labels_path: Path,
    known_path: Path,
    discovery: list[dict[str, Any]],
    calibration: list[dict[str, Any]],
    all_sources: list[dict[str, Any]],
    min_q1: int,
    min_q2: int,
    max_relation_proportion_diff: float,
) -> dict[str, Any]:
    discovery_ids = {row["source_id"] for row in discovery}
    calibration_ids = {row["source_id"] for row in calibration}
    all_ids = {row["source_id"] for row in all_sources}
    overlap = sorted(discovery_ids & calibration_ids)
    union = discovery_ids | calibration_ids
    issues: list[str] = []
    warnings: list[str] = []
    discovery_summary = summarize_sources(discovery)
    calibration_summary = summarize_sources(calibration)
    discovery_q_groups = discovery_summary["q_group_label_totals"]
    calibration_q_groups = calibration_summary["q_group_label_totals"]
    relation_audit = relation_distribution_audit(
        discovery,
        calibration,
        max_relation_proportion_diff=max_relation_proportion_diff,
    )

    if "holdout" in str(behavior_root):
        issues.append("behavior_root_appears_to_be_holdout")
    if behavior_root.name != "dev_behavior":
        issues.append("behavior_root_name_is_not_dev_behavior")
    if overlap:
        issues.append("discovery_calibration_source_overlap")
    if union != all_ids:
        issues.append("split_union_does_not_match_eligible_sources")
    if not discovery or not calibration:
        issues.append("empty_split")
    if discovery_q_groups.get("Q1", 0) < min_q1:
        issues.append("discovery_q1_below_minimum")
    if calibration_q_groups.get("Q1", 0) < min_q1:
        issues.append("calibration_q1_below_minimum")
    if discovery_q_groups.get("Q2", 0) < min_q2:
        issues.append("discovery_q2_below_minimum")
    if calibration_q_groups.get("Q2", 0) < min_q2:
        issues.append("calibration_q2_below_minimum")
    if relation_audit["severe_imbalance"]:
        warnings.append("relation_distribution_severe_imbalance")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "fail" if issues else "pass",
        "issues": issues,
        "warnings": warnings,
        "dataset_stage_name": dataset_stage_name,
        "behavior_root": str(behavior_root),
        "inputs": {
            "q_group_labels": str(q_labels_path),
            "known_sources": str(known_path),
        },
        "checks": {
            "read_holdout_data": False,
            "source_overlap_count": len(overlap),
            "source_overlap_examples": overlap[:10],
            "eligible_sources": len(all_ids),
            "split_union_sources": len(union),
            "discovery_sources": len(discovery_ids),
            "calibration_sources": len(calibration_ids),
            "min_q1": min_q1,
            "min_q2": min_q2,
            "discovery_q1": discovery_q_groups.get("Q1", 0),
            "discovery_q2": discovery_q_groups.get("Q2", 0),
            "calibration_q1": calibration_q_groups.get("Q1", 0),
            "calibration_q2": calibration_q_groups.get("Q2", 0),
        },
        "relation_distribution": relation_audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["pqpqa", "counterfact"], required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--root", default=str(repo_root()))
    parser.add_argument("--datasets-config", default="configs/datasets.json")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--behavior-root", default=None)
    parser.add_argument("--source-scope", choices=["known_4_of_5", "all_q_labeled"], default="known_4_of_5")
    parser.add_argument("--discovery-fraction", type=float, default=0.5)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--min-q1", type=int, default=1)
    parser.add_argument("--min-q2", type=int, default=1)
    parser.add_argument("--max-relation-proportion-diff", type=float, default=0.25)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = resolve_path(args.root)
    datasets_config = resolve_path(args.datasets_config, root)
    output_root = resolve_path(args.output_root, root)
    dataset_stage = load_dataset_stage(datasets_config, args.dataset, "dev", root)
    missing = validate_dataset_stage(dataset_stage)
    if missing:
        raise FileNotFoundError("Dataset registry has missing development paths:\n" + "\n".join(missing))

    behavior_root = (
        resolve_path(args.behavior_root, root)
        if args.behavior_root
        else default_behavior_root(output_root, args.dataset, args.model_tag)
    )
    if behavior_root.name != "dev_behavior":
        raise ValueError(f"--behavior-root must point to a dev_behavior directory, got: {behavior_root}")
    q_labels_path = behavior_root / "q_group_labels.jsonl"
    known_path = known_sources_path(behavior_root)
    split_root = output_root / args.dataset / args.model_tag / "dev_splits"

    if args.dry_run:
        print(f"Planned dev split output root: {split_root}", flush=True)
        print(f"Will read q labels: {q_labels_path}", flush=True)
        print(f"Will read known sources: {known_path}", flush=True)
        print("Holdout paths will not be read.", flush=True)
        return

    missing_inputs = [str(path) for path in [q_labels_path, known_path] if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError(
            "Missing development behavior outputs. Run prepare_behavior.py --stage dev first:\n"
            + "\n".join(missing_inputs)
        )

    q_labels = load_jsonl(q_labels_path)
    known_sources = load_json(known_path)
    source_rows = build_source_rows(
        q_labels=q_labels,
        known_sources=known_sources,
        source_scope=args.source_scope,
    )
    discovery, calibration = stratified_split(
        source_rows,
        discovery_fraction=args.discovery_fraction,
        split_seed=args.split_seed,
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset": args.dataset,
        "dataset_stage": "dev",
        "dataset_name": dataset_stage.name,
        "model_tag": args.model_tag,
        "source_scope": args.source_scope,
        "split_seed": args.split_seed,
        "discovery_fraction": args.discovery_fraction,
        "min_q1": args.min_q1,
        "min_q2": args.min_q2,
        "max_relation_proportion_diff": args.max_relation_proportion_diff,
        "input_behavior_root": str(behavior_root),
        "all_eligible": summarize_sources(source_rows),
        "discovery": summarize_sources(discovery),
        "calibration": summarize_sources(calibration),
    }
    audit = build_audit(
        dataset_stage_name=dataset_stage.name,
        behavior_root=behavior_root,
        q_labels_path=q_labels_path,
        known_path=known_path,
        discovery=discovery,
        calibration=calibration,
        all_sources=source_rows,
        min_q1=args.min_q1,
        min_q2=args.min_q2,
        max_relation_proportion_diff=args.max_relation_proportion_diff,
    )
    splits = {
        "schema_version": SCHEMA_VERSION,
        "dataset": args.dataset,
        "dataset_stage": "dev",
        "dataset_name": dataset_stage.name,
        "model_tag": args.model_tag,
        "source_scope": args.source_scope,
        "split_seed": args.split_seed,
        "discovery_fraction": args.discovery_fraction,
        "discovery_source_ids": [row["source_id"] for row in discovery],
        "calibration_source_ids": [row["source_id"] for row in calibration],
        "source_metadata": {
            row["source_id"]: row | {"split": "discovery"}
            for row in discovery
        }
        | {
            row["source_id"]: row | {"split": "calibration"}
            for row in calibration
        },
    }

    write_json(split_root / "splits.json", splits)
    write_json(
        split_root / "discovery_sources.json",
        {
            "schema_version": SCHEMA_VERSION,
            "split": "discovery",
            "dataset": args.dataset,
            "model_tag": args.model_tag,
            "sources": discovery,
        },
    )
    write_json(
        split_root / "calibration_sources.json",
        {
            "schema_version": SCHEMA_VERSION,
            "split": "calibration",
            "dataset": args.dataset,
            "model_tag": args.model_tag,
            "sources": calibration,
        },
    )
    write_json(split_root / "split_summary.json", summary)
    write_json(split_root / "split_audit.json", audit)

    print(f"Wrote {split_root / 'splits.json'}", flush=True)
    print(f"Wrote {split_root / 'discovery_sources.json'}", flush=True)
    print(f"Wrote {split_root / 'calibration_sources.json'}", flush=True)
    print(f"Wrote {split_root / 'split_summary.json'}", flush=True)
    print(f"Wrote {split_root / 'split_audit.json'}", flush=True)
    if audit["status"] != "pass":
        raise SystemExit("Split audit failed; see split_audit.json")


if __name__ == "__main__":
    main()
