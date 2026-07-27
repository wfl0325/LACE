#!/usr/bin/env python3
"""Evaluate one frozen LACE configuration on the holdout split.

This is the only public final-evaluation entry point. Method choices are read
from the frozen development configuration; the command line cannot override
the feature mixture, vector construction, intervention strength, or gate
thresholds.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.io_utils import repo_root, resolve_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["pqpqa", "counterfact"], required=True)
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
    parser.add_argument("--strict-damage-threshold", type=float, default=0.5)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--progress-every-direction", type=int, default=40)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the frozen config and holdout inputs without loading model weights.",
    )
    return parser.parse_args()


def append_path(argv: list[str], flag: str, value: Path | None) -> None:
    if value is not None:
        argv.extend([flag, str(value)])


def load_runtime():
    module_path = SCRIPT_DIR / "internal" / "lace_runtime.py"
    spec = importlib.util.spec_from_file_location("lace_final_holdout_runtime", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_root = resolve_path(args.output_root or root / "outputs", root)
    config = resolve_path(
        args.config
        or output_root
        / args.dataset
        / args.model_tag
        / "lace_config"
        / "lace_config.json",
        root,
    )
    output_dir = resolve_path(
        args.output_dir
        or output_root / args.dataset / args.model_tag / "holdout_lace_eval",
        root,
    )

    runtime_argv = [
        str(SCRIPT_DIR / "internal" / "lace_runtime.py"),
        "--dataset",
        args.dataset,
        "--model-tag",
        args.model_tag,
        "--root",
        str(root),
        "--output-root",
        str(output_root),
        "--config",
        str(config),
        "--output-dir",
        str(output_dir),
        "--device",
        args.device,
        "--mixture-modes",
        "selected",
        "--vector-constructions",
        "full_va",
        "--gate-modes",
        "peak_zA_tau",
        "--tau-a-scales",
        "",
        "--tau-a-values",
        "",
        "--tau-c-values",
        "",
        "--strict-damage-threshold",
        str(args.strict_damage_threshold),
        "--progress-every",
        str(args.progress_every),
        "--progress-every-direction",
        str(args.progress_every_direction),
    ]
    append_path(runtime_argv, "--models-config", args.models_config)
    append_path(runtime_argv, "--behavior-root", args.behavior_root)
    append_path(runtime_argv, "--feature-root", args.feature_root)
    append_path(runtime_argv, "--mixed-eval", args.mixed_eval)
    append_path(runtime_argv, "--model-path", args.model_path)
    append_path(runtime_argv, "--sae-path", args.sae_path)
    if args.dry_run:
        runtime_argv.append("--dry-run")

    runtime = load_runtime()
    previous_argv = sys.argv
    try:
        sys.argv = runtime_argv
        runtime.main()
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
