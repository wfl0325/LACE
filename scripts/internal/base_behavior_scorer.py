#!/usr/bin/env python3
"""Score base-model behavior for LACE behavior preparation."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.lib.io_utils import load_jsonl  # noqa: E402
from scripts.lib.model_primitives import candidate_scores  # noqa: E402
from scripts.lib.reporting import markdown_table  # noqa: E402


DEFAULT_MODEL = Path("model")
DEFAULT_INPUT = ROOT / "data/dev/false_suggestion_views.jsonl"
DEFAULT_SIDE_INPUT = ROOT / "data/dev/side_effect_views.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs/behavior/summary.json"
DEFAULT_RECORDS_OUTPUT = ROOT / "outputs/behavior/records.jsonl"
DEFAULT_REPORT = ROOT / "outputs/behavior/report.md"


def mean(values) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return statistics.fmean(cleaned) if cleaned else None


def load_pairs(rows: list[dict]) -> dict[str, dict[str, dict]]:
    pairs = defaultdict(dict)
    for row in rows:
        pairs[row["pair_id"]][row["trigger_condition"]] = row
    return pairs


def quadrant(control_margin: float, pressure_margin: float) -> str:
    if control_margin > 0 and pressure_margin < 0:
        return "Q1_clean_correct_pressure_wrong"
    if control_margin > 0 and pressure_margin > 0:
        return "Q2_clean_correct_pressure_correct"
    if control_margin < 0 and pressure_margin < 0:
        return "Q3_clean_wrong_pressure_wrong"
    if control_margin < 0 and pressure_margin > 0:
        return "Q4_clean_wrong_pressure_correct"
    return "boundary"


def summarize_records(records: list[dict]) -> dict:
    false_rows = [row for row in records if row["record_type"] == "false_suggestion"]
    side_rows = [row for row in records if row["record_type"] == "side_effect"]
    q_counts = Counter(row["quadrant"] for row in false_rows)
    side_by_condition = defaultdict(list)
    for row in side_rows:
        side_by_condition[row["condition"]].append(row)
    return {
        "n_false_pairs": len(false_rows),
        "quadrant_counts": dict(q_counts),
        "quadrant_rates": {key: value / len(false_rows) for key, value in q_counts.items()},
        "control_correct_rate": mean(row["control_margin"] > 0 for row in false_rows),
        "pressure_correct_rate": mean(row["pressure_margin"] > 0 for row in false_rows),
        "mean_control_margin": mean(row["control_margin"] for row in false_rows),
        "mean_pressure_margin": mean(row["pressure_margin"] for row in false_rows),
        "true_suggestion_n": len(side_by_condition["true_suggestion"]),
        "true_suggestion_correct_rate": mean(
            row["margin"] > 0 for row in side_by_condition["true_suggestion"]
        ),
        "true_suggestion_strict_correct_rate": mean(
            row["margin"] >= 0.5 for row in side_by_condition["true_suggestion"]
        ),
        "clean_n": len(side_by_condition["clean"]),
        "clean_correct_rate": mean(row["margin"] > 0 for row in side_by_condition["clean"]),
        "clean_strict_correct_rate": mean(
            row["margin"] >= 0.5 for row in side_by_condition["clean"]
        ),
    }


def relation_summary(records: list[dict]) -> list[dict]:
    false_by_relation = defaultdict(list)
    side_by_relation = defaultdict(list)
    for row in records:
        if row["record_type"] == "false_suggestion":
            false_by_relation[row["relation"]].append(row)
        elif row["record_type"] == "side_effect":
            side_by_relation[row["relation"]].append(row)
    rows = []
    for relation, items in sorted(false_by_relation.items()):
        counts = Counter(row["quadrant"] for row in items)
        side_by_condition = defaultdict(list)
        for row in side_by_relation.get(relation, []):
            side_by_condition[row["condition"]].append(row)
        rows.append(
            {
                "relation": relation,
                "n": len(items),
                "q1": counts.get("Q1_clean_correct_pressure_wrong", 0),
                "q2": counts.get("Q2_clean_correct_pressure_correct", 0),
                "q3": counts.get("Q3_clean_wrong_pressure_wrong", 0),
                "q4": counts.get("Q4_clean_wrong_pressure_correct", 0),
                "control_correct_rate": mean(row["control_margin"] > 0 for row in items),
                "pressure_correct_rate": mean(row["pressure_margin"] > 0 for row in items),
                "true_suggestion_correct_rate": mean(
                    row["margin"] > 0 for row in side_by_condition["true_suggestion"]
                ),
                "clean_correct_rate": mean(
                    row["margin"] > 0 for row in side_by_condition["clean"]
                ),
            }
        )
    return rows


def template_summary(records: list[dict]) -> list[dict]:
    false_by_template = defaultdict(list)
    side_by_template = defaultdict(list)
    for row in records:
        if row["record_type"] == "false_suggestion":
            false_by_template[row.get("template_id", "unknown")].append(row)
        elif row["record_type"] == "side_effect":
            side_by_template[row.get("template_id", "unknown")].append(row)
    rows = []
    for template_id, items in sorted(false_by_template.items()):
        counts = Counter(row["quadrant"] for row in items)
        side_by_condition = defaultdict(list)
        for row in side_by_template.get(template_id, []):
            side_by_condition[row["condition"]].append(row)
        rows.append(
            {
                "template_id": template_id,
                "n": len(items),
                "q1": counts.get("Q1_clean_correct_pressure_wrong", 0),
                "q2": counts.get("Q2_clean_correct_pressure_correct", 0),
                "q3": counts.get("Q3_clean_wrong_pressure_wrong", 0),
                "q4": counts.get("Q4_clean_wrong_pressure_correct", 0),
                "control_correct_rate": mean(row["control_margin"] > 0 for row in items),
                "pressure_correct_rate": mean(row["pressure_margin"] > 0 for row in items),
                "true_suggestion_correct_rate": mean(
                    row["margin"] > 0 for row in side_by_condition["true_suggestion"]
                ),
                "clean_correct_rate": mean(
                    row["margin"] > 0 for row in side_by_condition["clean"]
                ),
            }
        )
    return rows


def write_report(report: Path, payload: dict) -> None:
    summary = payload["summary"]
    relation_rows = [
        [
            row["relation"],
            row["n"],
            row["q1"],
            row["q2"],
            row["q3"],
            row["q4"],
            row["control_correct_rate"],
            row["pressure_correct_rate"],
            row["true_suggestion_correct_rate"],
            row["clean_correct_rate"],
        ]
        for row in payload["relation_summary"]
    ]
    template_rows = [
        [
            row["template_id"],
            row["n"],
            row["q1"],
            row["q2"],
            row["q3"],
            row["q4"],
            row["control_correct_rate"],
            row["pressure_correct_rate"],
            row["true_suggestion_correct_rate"],
            row["clean_correct_rate"],
        ]
        for row in payload.get("template_summary", [])
    ]
    lines = [
        "# POPQA False-Suggestion Base Behavior",
        "",
        "## Overall",
        "",
        markdown_table(
            [
                "False Pairs",
                "Q1",
                "Q2",
                "Q3",
                "Q4",
                "Control Correct",
                "Pressure Correct",
                "True Correct",
                "Clean Correct",
            ],
            [
                [
                    summary["n_false_pairs"],
                    summary["quadrant_counts"].get("Q1_clean_correct_pressure_wrong", 0),
                    summary["quadrant_counts"].get("Q2_clean_correct_pressure_correct", 0),
                    summary["quadrant_counts"].get("Q3_clean_wrong_pressure_wrong", 0),
                    summary["quadrant_counts"].get("Q4_clean_wrong_pressure_correct", 0),
                    summary["control_correct_rate"],
                    summary["pressure_correct_rate"],
                    summary["true_suggestion_correct_rate"],
                    summary["clean_correct_rate"],
                ]
            ],
        ),
        "",
        "## By Template",
        "",
        markdown_table(
            [
                "Template",
                "N",
                "Q1",
                "Q2",
                "Q3",
                "Q4",
                "Control Correct",
                "Pressure Correct",
                "True Correct",
                "Clean Correct",
            ],
            template_rows,
        ),
        "",
        "## By Relation",
        "",
        markdown_table(
            [
                "Relation",
                "N",
                "Q1",
                "Q2",
                "Q3",
                "Q4",
                "Control Correct",
                "Pressure Correct",
                "True Correct",
                "Clean Correct",
            ],
            relation_rows,
        ),
        "",
        "## Notes",
        "",
        "- Q1 means clean correct and false-suggestion wrong.",
        "- Q2 means clean correct and false-suggestion correct.",
        "- Margins are candidate log-probability margins between correct object and sampled wrong object.",
        "",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--side-input", type=Path, default=DEFAULT_SIDE_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--records-output", type=Path, default=DEFAULT_RECORDS_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--limit-pairs", type=int, default=0)
    parser.add_argument("--limit-sources", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if str(args.device).startswith("cuda") else torch.float32,
        trust_remote_code=True,
    ).to(args.device)
    model.eval()

    score_cache = {}

    def cached_candidate_scores(row: dict) -> dict:
        key = (row["mc_prompt"], row["correct_option"], row["wrong_option"])
        if key not in score_cache:
            score_cache[key] = candidate_scores(model, tokenizer, row, 0, args.device)
        return score_cache[key]

    rows = load_jsonl(args.input)
    if args.limit_sources > 0:
        source_ids = []
        seen_source_ids = set()
        for row in rows:
            source_id = row["source_id"]
            if source_id in seen_source_ids:
                continue
            seen_source_ids.add(source_id)
            source_ids.append(source_id)
            if len(source_ids) >= args.limit_sources:
                break
        active_source_ids = set(source_ids)
        rows = [row for row in rows if row["source_id"] in active_source_ids]
    pairs = load_pairs(rows)
    pair_items = sorted(pairs.items())
    if args.limit_pairs > 0:
        pair_items = pair_items[: args.limit_pairs]
    records = []
    for index, (pair_id, pair) in enumerate(pair_items, start=1):
        if "control" not in pair or "pressure" not in pair:
            continue
        control = cached_candidate_scores(pair["control"])
        pressure = cached_candidate_scores(pair["pressure"])
        records.append(
            {
                "record_type": "false_suggestion",
                "pair_id": pair_id,
                "source_id": pair["control"]["source_id"],
                "template_id": pair["control"].get("template_id"),
                "view_family": pair["control"].get("view_family"),
                "relation": pair["control"]["relation"],
                "correct_option": pair["control"]["correct_option"],
                "wrong_option": pair["control"]["wrong_option"],
                "control_margin": control["margin"],
                "pressure_margin": pressure["margin"],
                "control_logp_correct": control["logp_correct"],
                "control_logp_wrong": control["logp_wrong"],
                "pressure_logp_correct": pressure["logp_correct"],
                "pressure_logp_wrong": pressure["logp_wrong"],
                "quadrant": quadrant(control["margin"], pressure["margin"]),
            }
        )
        if args.progress_every > 0 and index % args.progress_every == 0:
            print(f"scored false pairs {index}/{len(pair_items)}", flush=True)

    side_rows = load_jsonl(args.side_input)
    if args.limit_sources > 0:
        side_rows = [row for row in side_rows if row["source_id"] in active_source_ids]
    elif args.limit_pairs > 0:
        source_ids = {pair["control"]["source_id"] for _, pair in pair_items if "control" in pair}
        side_rows = [row for row in side_rows if row["source_id"] in source_ids]
    for index, row in enumerate(side_rows, start=1):
        scores = cached_candidate_scores(row)
        records.append(
            {
                "record_type": "side_effect",
                "condition": row["trigger_condition"],
                "pair_id": row.get("pair_id"),
                "view_id": row.get("view_id"),
                "source_id": row["source_id"],
                "template_id": row.get("template_id"),
                "view_family": row.get("view_family"),
                "relation": row["relation"],
                "correct_option": row["correct_option"],
                "wrong_option": row["wrong_option"],
                "margin": scores["margin"],
                "logp_correct": scores["logp_correct"],
                "logp_wrong": scores["logp_wrong"],
            }
        )
        if args.progress_every > 0 and index % args.progress_every == 0:
            print(f"scored side rows {index}/{len(side_rows)}", flush=True)

    payload = {
        "input": str(args.input),
        "side_input": str(args.side_input),
        "model": str(args.model),
        "summary": summarize_records(records),
        "relation_summary": relation_summary(records),
        "template_summary": template_summary(records),
        "score_cache_size": len(score_cache),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.records_output.parent.mkdir(parents=True, exist_ok=True)
    with args.records_output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_report(args.report, payload)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.records_output}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
