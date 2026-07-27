#!/usr/bin/env python3
"""Prepare model-dependent behavior records for LACE.

This entry point prepares model-dependent behavior labels with config-driven
paths and separate development/holdout output folders.

Current scope:
- known-fact filtering from clean paraphrase views
- base behavior from control/pressure compatibility views
- true-suggestion and clean side-effect behavior from compatibility views
- unified behavior schema generation
- mixed_eval scoring for clean_eval / false_suggestion / true_suggestion / cue_only when available
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lib.behavior_schema import write_unified_behavior_outputs  # noqa: E402
from lib.dataset_registry import load_dataset_stage, validate_dataset_stage  # noqa: E402
from lib.io_utils import repo_root, resolve_path, write_json, write_text  # noqa: E402
from lib.model_registry import load_model_spec  # noqa: E402


def default_models_config(root: Path) -> Path:
    local = root / "configs/models.local.json"
    if local.exists():
        return local
    return root / "configs/models.example.json"


def locked_readme() -> str:
    return """# Locked Holdout Behavior

This directory contains holdout known-fact filtering and base-behavior labels.

These files are allowed for final evaluation labels only:

- Known 4/5 source selection for this model
- Q1/Q2/Q3/Q4 grouping for this model
- before-intervention margins for final metrics

They must not be used for:

- feature discovery
- layer selection
- top-k selection
- threshold tuning
- strength selection
- mixture-mode selection

If any method choice depends on this directory, the holdout protocol is invalid.
"""


def command_to_text(command: list[str]) -> str:
    return " ".join(json.dumps(part) if " " in part else part for part in command)


def run_command(command: list[str], dry_run: bool) -> None:
    print(command_to_text(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def build_commands(
    args: argparse.Namespace,
) -> tuple[list[str], list[str], Path, Path, object, Path, str]:
    root = resolve_path(args.root)
    datasets_config = resolve_path(args.datasets_config, root)
    models_config = resolve_path(args.models_config, root)
    output_root = resolve_path(args.output_root, root)

    dataset_stage = load_dataset_stage(datasets_config, args.dataset, args.stage, root)
    missing = validate_dataset_stage(dataset_stage)
    if missing:
        raise FileNotFoundError("Dataset registry has missing paths:\n" + "\n".join(missing))

    model = load_model_spec(
        models_config,
        args.model_tag,
        root,
        allow_placeholder=args.dry_run or bool(args.model_path),
    )
    model_path = resolve_path(args.model_path, root) if args.model_path else model.model_path
    if not args.dry_run and not model_path.exists():
        raise FileNotFoundError(f"model_path not found: {model_path}")
    resolved_device = args.device or model.default_device
    stage_dir_name = "dev_behavior" if args.stage == "dev" else "holdout_behavior"
    stage_root = output_root / args.dataset / args.model_tag / stage_dir_name
    base_root = stage_root / "base_behavior"
    known_root = stage_root / "known_fact_filter"
    reports_root = stage_root / "reports"

    base_scorer = root / "scripts/internal/base_behavior_scorer.py"
    known_filter = root / "scripts/internal/known_fact_filter.py"

    base_command = [
        sys.executable,
        str(base_scorer),
        "--model",
        str(model_path),
        "--input",
        str(dataset_stage.false_suggestion),
        "--side-input",
        str(dataset_stage.side_effect),
        "--output",
        str(base_root / "summary.json"),
        "--records-output",
        str(base_root / "records.jsonl"),
        "--report",
        str(reports_root / "base_behavior_report.md"),
        "--device",
        resolved_device,
        "--progress-every",
        str(args.progress_every),
    ]
    known_command = [
        sys.executable,
        str(known_filter),
        "--model",
        str(model_path),
        "--input",
        str(dataset_stage.known_fact_clean),
        "--false-records",
        str(base_root / "records.jsonl"),
        "--output",
        str(known_root / "summary.json"),
        "--records-output",
        str(known_root / "records.jsonl"),
        "--known-sources-output",
        str(known_root / "known_sources.json"),
        "--report",
        str(reports_root / "known_fact_filter_report.md"),
        "--known-min-correct",
        str(args.known_min_correct),
        "--strong-mean-margin",
        str(args.strong_mean_margin),
        "--device",
        resolved_device,
        "--progress-every",
        str(args.progress_every),
    ]
    if args.limit_sources > 0:
        base_command.extend(["--limit-sources", str(args.limit_sources)])
        known_command.extend(["--limit-sources", str(args.limit_sources)])

    run_config = {
        "script": "prepare_behavior.py",
        "stage": "behavior_preparation",
        "dataset": args.dataset,
        "dataset_stage": args.stage,
        "dataset_name": dataset_stage.name,
        "model_tag": args.model_tag,
        "model_path": str(model_path),
        "model_path_source": "cli_override" if args.model_path else "model_registry",
        "output_root": str(stage_root),
        "known_fact_clean": str(dataset_stage.known_fact_clean),
        "false_suggestion": str(dataset_stage.false_suggestion),
        "side_effect": str(dataset_stage.side_effect),
        "mixed_eval": str(dataset_stage.mixed_eval) if dataset_stage.mixed_eval else None,
        "known_min_correct": args.known_min_correct,
        "strong_mean_margin": args.strong_mean_margin,
        "limit_sources": args.limit_sources,
        "progress_every": args.progress_every,
        "device": resolved_device,
        "allowed_uses": dataset_stage.allowed_for,
        "forbidden_uses": dataset_stage.forbidden_for,
        "notes": [
            "Holdout behavior may be used only for evaluation labels, not method selection.",
            "Q groups are defined only from false-suggestion control/pressure pairs.",
            "Behavior preparation writes unified behavior schema files at the stage root.",
        ],
        "unified_outputs": {
            "known_sources": str(stage_root / "known_sources.json"),
            "behavior_records": str(stage_root / "behavior_records.jsonl"),
            "q_group_labels": str(stage_root / "q_group_labels.jsonl"),
            "condition_behavior_summary": str(stage_root / "condition_behavior_summary.json"),
            "behavior_audit": str(stage_root / "behavior_audit.json"),
        },
        "mixed_eval_scoring": {
            "available": dataset_stage.mixed_eval is not None and dataset_stage.mixed_eval.exists(),
            "enabled_by_default": True,
        },
        "commands": {
            "base_behavior": base_command,
            "known_fact_filter": known_command,
        },
    }
    write_json(stage_root / "run_config.json", run_config)
    if args.stage == "holdout":
        write_text(stage_root / "README.locked.md", locked_readme())
    return base_command, known_command, stage_root, known_root, dataset_stage, model_path, resolved_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["popqa", "counterfact"], required=True)
    parser.add_argument("--stage", choices=["dev", "holdout"], required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--root", default=str(repo_root()))
    parser.add_argument("--datasets-config", default="configs/datasets.json")
    parser.add_argument("--models-config", default=None)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-sources", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--known-min-correct", type=int, default=4)
    parser.add_argument("--strong-mean-margin", type=float, default=1.0)
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--skip-known", action="store_true")
    parser.add_argument("--skip-unified", action="store_true")
    parser.add_argument("--skip-mixed-scoring", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = resolve_path(args.root)
    if args.models_config is None:
        args.models_config = str(default_models_config(root))

    base_command, known_command, stage_root, _, dataset_stage, model_path, resolved_device = build_commands(args)
    print(f"Prepared behavior output root: {stage_root}", flush=True)

    if not args.skip_base:
        run_command(base_command, args.dry_run)
    if not args.skip_known:
        if args.skip_base and not (stage_root / "base_behavior/records.jsonl").exists() and not args.dry_run:
            raise FileNotFoundError(
                "--skip-base was set, but base_behavior/records.jsonl does not exist. "
                "Run base behavior first or remove --skip-base."
            )
        run_command(known_command, args.dry_run)

    if not args.skip_unified:
        if args.dry_run:
            print("Planned unified behavior outputs:", flush=True)
            for name in [
                "known_sources.json",
                "behavior_records.jsonl",
                "q_group_labels.jsonl",
                "condition_behavior_summary.json",
                "behavior_audit.json",
            ]:
                print(stage_root / name, flush=True)
        else:
            outputs = write_unified_behavior_outputs(
                dataset_stage=dataset_stage,
                stage_root=stage_root,
                model_tag=args.model_tag,
                model_path=model_path,
                device=resolved_device,
                limit_sources=args.limit_sources,
                progress_every=args.progress_every,
                score_mixed=not args.skip_mixed_scoring,
            )
            for name, path in outputs.items():
                print(f"Wrote {name}: {path}", flush=True)


if __name__ == "__main__":
    main()
