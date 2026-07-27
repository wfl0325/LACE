#!/usr/bin/env python3
"""Run model-dependent known-fact filtering for LACE."""

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
DEFAULT_INPUT = ROOT / "data/dev/known_fact_clean_views.jsonl"
DEFAULT_FALSE_RECORDS = ROOT / "outputs/behavior/base_behavior/records.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs/behavior/known_fact_filter/summary.json"
DEFAULT_RECORDS_OUTPUT = ROOT / "outputs/behavior/known_fact_filter/records.jsonl"
DEFAULT_KNOWN_SOURCES_OUTPUT = ROOT / "outputs/behavior/known_fact_filter/known_sources.json"
DEFAULT_REPORT = ROOT / "outputs/behavior/known_fact_filter/report.md"


def mean(values) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return statistics.fmean(cleaned) if cleaned else None


def median(values) -> float | None:
    cleaned = sorted(float(value) for value in values if value is not None)
    return statistics.median(cleaned) if cleaned else None


def group_by_source(rows: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["source_id"]].append(row)
    return grouped


def summarize_source(rows: list[dict], margins: list[dict], known_min_correct: int, strong_mean_margin: float) -> dict:
    pass_count = sum(item["margin"] > 0 for item in margins)
    strict_count = sum(item["margin"] >= 0.5 for item in margins)
    first = rows[0]
    mean_margin = mean(item["margin"] for item in margins)
    min_margin = min(item["margin"] for item in margins)
    return {
        "source_id": first["source_id"],
        "source_row": first.get("source_row"),
        "relation": first["relation"],
        "subject": first.get("subject"),
        "question_text": first["question_text"],
        "correct_option": first["correct_option"],
        "wrong_option": first["wrong_option"],
        "n_paraphrases": len(margins),
        "pass_count": pass_count,
        "strict_count": strict_count,
        "pass_rate": pass_count / len(margins) if margins else None,
        "mean_margin": mean_margin,
        "median_margin": median(item["margin"] for item in margins),
        "min_margin": min_margin,
        "max_margin": max(item["margin"] for item in margins),
        "known_4_of_5": pass_count >= known_min_correct,
        "strong_5_of_5": pass_count == len(margins) and mean_margin is not None and mean_margin >= strong_mean_margin,
        "paraphrase_margins": margins,
    }


def false_subset_summary(false_records: list[dict], source_ids: set[str]) -> dict:
    rows = [
        row
        for row in false_records
        if row.get("record_type") == "false_suggestion" and row["source_id"] in source_ids
    ]
    q_counts = Counter(row["quadrant"] for row in rows)
    n = len(rows)
    q1 = q_counts.get("Q1_clean_correct_pressure_wrong", 0)
    q2 = q_counts.get("Q2_clean_correct_pressure_correct", 0)
    q3 = q_counts.get("Q3_clean_wrong_pressure_wrong", 0)
    q4 = q_counts.get("Q4_clean_wrong_pressure_correct", 0)
    return {
        "n_false_pairs": n,
        "n_sources": len(source_ids),
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "q4": q4,
        "q1_rate": q1 / n if n else None,
        "q2_rate": q2 / n if n else None,
        "control_correct_rate": mean(row["control_margin"] > 0 for row in rows),
        "pressure_correct_rate": mean(row["pressure_margin"] > 0 for row in rows),
        "mean_control_margin": mean(row["control_margin"] for row in rows),
        "mean_pressure_margin": mean(row["pressure_margin"] for row in rows),
        "mean_delta_margin": mean(row["pressure_margin"] - row["control_margin"] for row in rows),
    }


def relation_known_summary(source_records: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in source_records:
        grouped[row["relation"]].append(row)
    result = []
    for relation, rows in sorted(grouped.items()):
        result.append(
            {
                "relation": relation,
                "sources": len(rows),
                "known_4_of_5": sum(row["known_4_of_5"] for row in rows),
                "strong_5_of_5": sum(row["strong_5_of_5"] for row in rows),
                "mean_pass_rate": mean(row["pass_rate"] for row in rows),
                "mean_margin": mean(row["mean_margin"] for row in rows),
            }
        )
    return result


def margin_bin_summary(source_records: list[dict], false_records: list[dict]) -> list[dict]:
    known = [row for row in source_records if row["known_4_of_5"]]
    known = sorted(known, key=lambda row: row["mean_margin"])
    if not known:
        return []
    bins = [
        ("low", known[: len(known) // 3]),
        ("medium", known[len(known) // 3 : (2 * len(known)) // 3]),
        ("high", known[(2 * len(known)) // 3 :]),
    ]
    result = []
    for label, rows in bins:
        source_ids = {row["source_id"] for row in rows}
        subset = false_subset_summary(false_records, source_ids)
        result.append(
            {
                "bin": label,
                "sources": len(rows),
                "mean_known_margin": mean(row["mean_margin"] for row in rows),
                **subset,
            }
        )
    return result


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(report: Path, payload: dict) -> None:
    summary = payload["summary"]
    subset_name = {
        "evaluated_sources": "全部评估样本",
        "known_4_of_5": "Known 4/5",
        "strong_5_of_5": "Strong 5/5",
    }
    bin_name = {
        "low": "低 clean margin",
        "medium": "中 clean margin",
        "high": "高 clean margin",
    }
    subset_rows = [
        [
            subset_name.get(item["subset"], item["subset"]),
            item["n_sources"],
            item["n_false_pairs"],
            item["control_correct_rate"],
            item["pressure_correct_rate"],
            item["q1"],
            item["q2"],
            item["q3"],
            item["q1_rate"],
            item["mean_delta_margin"],
        ]
        for item in payload["subset_phenomenon"]
    ]
    relation_rows = [
        [
            row["relation"],
            row["sources"],
            row["known_4_of_5"],
            row["strong_5_of_5"],
            row["mean_pass_rate"],
            row["mean_margin"],
        ]
        for row in payload["relation_summary"]
    ]
    bin_rows = [
        [
            bin_name.get(row["bin"], row["bin"]),
            row["sources"],
            row["mean_known_margin"],
            row["pressure_correct_rate"],
            row["q1_rate"],
            row["mean_delta_margin"],
        ]
        for row in payload["margin_bin_summary"]
    ]
    lines = [
        "# PQPQA Known-Fact 过滤报告",
        "",
        "## 总览",
        "",
        markdown_table(
            [
                "事实源数",
                "Clean 改写数",
                "Known 4/5",
                "Strong 5/5",
                "平均通过率",
                "平均 Margin",
            ],
            [
                [
                    summary["n_sources"],
                    summary["n_paraphrases"],
                    summary["known_4_of_5"],
                    summary["strong_5_of_5"],
                    summary["mean_pass_rate"],
                    summary["mean_margin"],
                ]
            ],
        ),
        "",
        "## 子集上的 False-Suggestion 现象",
        "",
        markdown_table(
            [
                "子集",
                "事实源数",
                "False Pairs",
                "Clean Acc",
                "False Acc",
                "Q1",
                "Q2",
                "Q3",
                "Q1 Rate",
                "Mean Delta Margin",
            ],
            subset_rows,
        ),
        "",
        "## 按关系类型统计 Known-Fact",
        "",
        markdown_table(
            [
                "关系",
                "事实源数",
                "Known 4/5",
                "Strong 5/5",
                "平均通过率",
                "平均 Margin",
            ],
            relation_rows,
        ),
        "",
        "## Clean Margin 分桶",
        "",
        markdown_table(
            [
                "分桶",
                "事实源数",
                "Known 平均 Margin",
                "False Acc",
                "Q1 Rate",
                "Mean Delta Margin",
            ],
            bin_rows,
        ),
        "",
        "## 备注",
        "",
        f"- Known Fact Set 定义为：5 个 clean paraphrases 中至少 `{payload['config']['known_min_correct']}` 个具有正 candidate margin。",
        f"- Strong Known Fact Set 定义为：5/5 全部为正，且平均 clean margin >= `{payload['config']['strong_mean_margin']}`。",
        "- 正确性按 PQPQA 正确 object 与固定 same-relation wrong object 的 candidate margin 判断。",
        "- 子集现象指标复用已有 PQPQA multitemplate false-suggestion base-behavior records。",
        "",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--false-records", type=Path, default=DEFAULT_FALSE_RECORDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--records-output", type=Path, default=DEFAULT_RECORDS_OUTPUT)
    parser.add_argument("--known-sources-output", type=Path, default=DEFAULT_KNOWN_SOURCES_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--known-min-correct", type=int, default=4)
    parser.add_argument("--strong-mean-margin", type=float, default=1.0)
    parser.add_argument("--limit-sources", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    grouped = group_by_source(rows)
    source_items = sorted(grouped.items())
    if args.limit_sources > 0:
        source_items = source_items[: args.limit_sources]
    active_source_ids = {source_id for source_id, _ in source_items}
    rows = [row for row in rows if row["source_id"] in active_source_ids]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if str(args.device).startswith("cuda") else torch.float32,
        trust_remote_code=True,
    ).to(args.device)
    model.eval()

    score_records = []
    source_records = []
    for source_index, (source_id, source_rows) in enumerate(source_items, start=1):
        margins = []
        for row in sorted(source_rows, key=lambda item: item["clean_paraphrase_index"]):
            scores = candidate_scores(model, tokenizer, row, 0, args.device)
            record = {
                "record_type": "known_fact_paraphrase",
                "source_id": source_id,
                "source_row": row.get("source_row"),
                "relation": row["relation"],
                "template_id": row["template_id"],
                "clean_paraphrase_index": row["clean_paraphrase_index"],
                "question_text": row["question_text"],
                "correct_option": row["correct_option"],
                "wrong_option": row["wrong_option"],
                "margin": scores["margin"],
                "logp_correct": scores["logp_correct"],
                "logp_wrong": scores["logp_wrong"],
                "is_correct": scores["margin"] > 0,
            }
            score_records.append(record)
            margins.append(record)
        source_records.append(
            summarize_source(
                source_rows,
                margins,
                args.known_min_correct,
                args.strong_mean_margin,
            )
        )
        if args.progress_every > 0 and source_index % args.progress_every == 0:
            print(f"scored known-fact sources {source_index}/{len(source_items)}", flush=True)

    false_records = load_jsonl(args.false_records)
    all_eval_source_ids = {row["source_id"] for row in source_records}
    known_source_ids = {row["source_id"] for row in source_records if row["known_4_of_5"]}
    strong_source_ids = {row["source_id"] for row in source_records if row["strong_5_of_5"]}

    subset_phenomenon = []
    for label, source_ids in [
        ("evaluated_sources", all_eval_source_ids),
        ("known_4_of_5", known_source_ids),
        ("strong_5_of_5", strong_source_ids),
    ]:
        subset = false_subset_summary(false_records, source_ids)
        subset_phenomenon.append({"subset": label, **subset})

    summary = {
        "n_sources": len(source_records),
        "n_paraphrases": len(score_records),
        "known_4_of_5": len(known_source_ids),
        "strong_5_of_5": len(strong_source_ids),
        "mean_pass_rate": mean(row["pass_rate"] for row in source_records),
        "mean_margin": mean(row["mean_margin"] for row in source_records),
        "median_margin": median(row["mean_margin"] for row in source_records),
    }
    payload = {
        "input": str(args.input),
        "false_records": str(args.false_records),
        "model": str(args.model),
        "config": {
            "known_min_correct": args.known_min_correct,
            "strong_mean_margin": args.strong_mean_margin,
            "limit_sources": args.limit_sources,
        },
        "summary": summary,
        "subset_phenomenon": subset_phenomenon,
        "relation_summary": relation_known_summary(source_records),
        "margin_bin_summary": margin_bin_summary(source_records, false_records),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_jsonl(args.records_output, score_records)
    args.known_sources_output.parent.mkdir(parents=True, exist_ok=True)
    args.known_sources_output.write_text(
        json.dumps(
            {
                "known_4_of_5": sorted(known_source_ids),
                "strong_5_of_5": sorted(strong_source_ids),
                "source_records": source_records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    write_report(args.report, payload)

    print(f"Wrote {args.output}")
    print(f"Wrote {args.records_output}")
    print(f"Wrote {args.known_sources_output}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
