#!/usr/bin/env python3
"""Discover candidate LACE feature banks from the development discovery split.

This entry point only discovers candidate feature banks:

- F_A: wrong-attractor suppression candidates
- F_R: robust-state enhancement candidates

It reads the development discovery split produced by ``build_dev_splits.py``.
It does not read holdout data and does not perform final calibration, layer
selection, threshold selection, mixture selection, or side-effect-constrained
method selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in [SCRIPT_DIR, REPO_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lib.dataset_registry import load_dataset_stage, validate_dataset_stage  # noqa: E402
from lib.io_utils import load_json, load_jsonl, repo_root, resolve_path, write_json, write_jsonl  # noqa: E402
from lib.model_registry import (  # noqa: E402
    load_model_spec,
    require_residual_sae_compatibility,
    sae_compatibility_metadata,
)
from lib.feature_primitives import (  # noqa: E402
    Q1,
    Q2,
    direction_projection,
    grouped_subset,
    load_sae,
    pair_lookup,
    pooled_claim_hidden,
    sae_activation,
    screen_features,
    select_quadrants,
    suppression_projection,
)
from lib.model_primitives import capture_hidden  # noqa: E402
from lib.span_utils import locate_regions  # noqa: E402


SCHEMA_VERSION = "lace_feature_bank_candidate_v1"


def default_models_config(root: Path) -> Path:
    local = root / "configs/models.local.json"
    if local.exists():
        return local
    return root / "configs/models.example.json"


def parse_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def default_layers(search_config_path: Path) -> str:
    config = load_json(search_config_path)
    return ",".join(str(layer) for layer in config.get("candidate_layers", [10, 11, 12, 13, 14]))


def count_groups(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {"Q1": 0, "Q2": 0, "other": 0}
    for row in rows:
        if row.get("quadrant") == Q1:
            counts["Q1"] += 1
        elif row.get("quadrant") == Q2:
            counts["Q2"] += 1
        else:
            counts["other"] += 1
    return dict(counts)


def load_discovery_sources(discovery_sources_path: Path) -> set[str]:
    payload = load_json(discovery_sources_path)
    if payload.get("split") == "discovery":
        source_ids = {row["source_id"] for row in payload.get("sources", [])}
    else:
        if payload.get("dataset_stage") != "dev":
            raise ValueError(
                f"Expected dev discovery sources, got dataset_stage={payload.get('dataset_stage')!r}"
            )
        source_ids = set(payload.get("discovery_source_ids", []))
    if not source_ids:
        raise ValueError(f"No discovery source ids found in {discovery_sources_path}")
    return source_ids


def feature_metric(row: dict[str, Any], key: str, default: float | None = None) -> float | None:
    value = row.get(key, default)
    return None if value is None else float(value)


def annotate_bank_rows(
    *,
    rows: list[dict[str, Any]],
    bank: str,
    dataset: str,
    dataset_name: str,
    model_tag: str,
    layer: int,
    projection_scores: torch.Tensor,
    decoder_alignment: torch.Tensor,
    z_q1: torch.Tensor,
    z_q2: torch.Tensor,
    extra_projection: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    output = []
    q2_q1_gap = z_q2 - z_q1
    for row in rows:
        fid = int(row["feature_id"])
        payload = {
            "schema_version": SCHEMA_VERSION,
            "record_type": "feature_bank_candidate",
            "dataset": dataset,
            "dataset_stage": "dev",
            "dataset_name": dataset_name,
            "model_tag": model_tag,
            "split": "discovery",
            "bank": bank,
            "bank_name": "wrong_attractor_features" if bank == "A" else "robust_state_features",
            "layer": layer,
            "feature_id": fid,
            "feature_uid": f"L{layer}:F{fid}",
            "rank_by_causal_score": row.get("rank_by_causal_score"),
            "candidate_rank": row.get("candidate_rank"),
            "causal_score": feature_metric(row, "causal_score"),
            "screen_sign": feature_metric(row, "screen_sign"),
            "screen_strength": feature_metric(row, "screen_strength"),
            "decoder_norm": feature_metric(row, "decoder_norm"),
            "projection_score": float(projection_scores[fid].item()),
            "decoder_alignment": float(decoder_alignment[fid].item()),
            "q1_activation_mean": float(z_q1[fid].item()),
            "q2_activation_mean": float(z_q2[fid].item()),
            "q2_minus_q1_activation": float(q2_q1_gap[fid].item()),
            "q1_correction_rate": feature_metric(row, "q1_correction_rate"),
            "q1_mean_delta_margin": feature_metric(row, "q1_mean_delta_margin"),
            "q2_side_effect_rate": feature_metric(row, "q2_side_effect_rate"),
            "q2_mean_delta_margin": feature_metric(row, "q2_mean_delta_margin"),
            "q1_improved_fraction": feature_metric(row, "q1_improved_fraction"),
            "q2_improved_fraction": feature_metric(row, "q2_improved_fraction"),
        }
        for key, tensor in extra_projection.items():
            payload[key] = float(tensor[fid].item())
        output.append(payload)
    return output


def build_audit(
    *,
    behavior_root: Path,
    discovery_sources_path: Path,
    discovery_sources: set[str],
    selected: list[dict[str, Any]],
    screen_records: list[dict[str, Any]],
    zero_target_count: int,
    max_zero_target_count: int,
    layers: list[int],
    bank_a: list[dict[str, Any]],
    bank_r: list[dict[str, Any]],
    enforce_nonempty_banks: bool,
) -> dict[str, Any]:
    issues = []
    if behavior_root.name != "dev_behavior":
        issues.append("behavior_root_name_is_not_dev_behavior")
    if "holdout" in str(behavior_root) or "holdout" in str(discovery_sources_path):
        issues.append("holdout_path_detected")
    selected_sources = {row["source_id"] for row in selected}
    outside_discovery = sorted(selected_sources - discovery_sources)
    selected_counts = count_groups(selected)
    screen_counts = count_groups(screen_records)
    if outside_discovery:
        issues.append("selected_records_outside_discovery_sources")
    if selected_counts.get("other", 0) != 0:
        issues.append("selected_records_have_non_q1_q2_quadrants")
    if screen_counts.get("other", 0) != 0:
        issues.append("screen_records_have_non_q1_q2_quadrants")
    if zero_target_count > max_zero_target_count:
        issues.append("zero_target_count_exceeds_maximum")
    if selected_counts.get("Q1", 0) == 0:
        issues.append("no_q1_discovery_records")
    if selected_counts.get("Q2", 0) == 0:
        issues.append("no_q2_discovery_records")
    if enforce_nonempty_banks and not bank_a:
        issues.append("empty_feature_bank_A")
    if enforce_nonempty_banks and not bank_r:
        issues.append("empty_feature_bank_R")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "fail" if issues else "pass",
        "issues": issues,
        "checks": {
            "read_holdout_data": False,
            "behavior_root": str(behavior_root),
            "discovery_sources_path": str(discovery_sources_path),
            "discovery_source_count": len(discovery_sources),
            "selected_source_count": len(selected_sources),
            "selected_counts": selected_counts,
            "screen_counts": screen_counts,
            "zero_target_count": zero_target_count,
            "max_zero_target_count": max_zero_target_count,
            "layers": layers,
            "feature_bank_A_count": len(bank_a),
            "feature_bank_R_count": len(bank_r),
            "enforce_nonempty_banks": enforce_nonempty_banks,
            "outside_discovery_examples": outside_discovery[:10],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["popqa", "counterfact"], required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--sae-path", default=None)
    parser.add_argument("--root", default=str(repo_root()))
    parser.add_argument("--datasets-config", default="configs/datasets.json")
    parser.add_argument("--models-config", default=None)
    parser.add_argument("--search-config", default="configs/lace_search_space.json")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--behavior-root", default=None)
    parser.add_argument("--splits-root", default=None)
    parser.add_argument("--discovery-sources", default=None)
    parser.add_argument("--layers", default=None)
    parser.add_argument("--candidate-k", type=int, default=200)
    parser.add_argument("--screen-strength", type=float, default=2.0)
    parser.add_argument("--screen-max-train-per-group", type=int, default=0)
    parser.add_argument("--screen-feature-scale", choices=["raw", "unit", "alignment"], default="raw")
    parser.add_argument("--q2-side-beta", type=float, default=5.0)
    parser.add_argument("--q2-delta-beta", type=float, default=1.0)
    parser.add_argument("--max-pairs-per-group", type=int, default=0)
    parser.add_argument("--max-zero-target-count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    root = resolve_path(args.root)
    datasets_config = resolve_path(args.datasets_config, root)
    search_config = resolve_path(args.search_config, root)
    output_root = resolve_path(args.output_root, root)
    if args.models_config is None:
        args.models_config = str(default_models_config(root))
    models_config = resolve_path(args.models_config, root)

    dataset_stage = load_dataset_stage(datasets_config, args.dataset, "dev", root)
    missing = validate_dataset_stage(dataset_stage)
    if missing:
        raise FileNotFoundError("Dataset registry has missing development paths:\n" + "\n".join(missing))

    model = load_model_spec(
        models_config,
        args.model_tag,
        root,
        allow_placeholder=args.dry_run or bool(args.model_path),
    )
    model_path = resolve_path(args.model_path, root) if args.model_path else model.model_path
    sae_path = resolve_path(args.sae_path, root) if args.sae_path else model.sae_path
    require_residual_sae_compatibility(model)
    if sae_path is None and not args.dry_run:
        raise ValueError(f"sae_path for {args.model_tag!r} is required for feature discovery")

    behavior_root = (
        resolve_path(args.behavior_root, root)
        if args.behavior_root
        else output_root / args.dataset / args.model_tag / "dev_behavior"
    )
    if behavior_root.name != "dev_behavior":
        raise ValueError(f"--behavior-root must point to a dev_behavior directory, got: {behavior_root}")
    splits_root = (
        resolve_path(args.splits_root, root)
        if args.splits_root
        else output_root / args.dataset / args.model_tag / "dev_splits"
    )
    discovery_sources_path = (
        resolve_path(args.discovery_sources, root)
        if args.discovery_sources
        else splits_root / "discovery_sources.json"
    )
    feature_root = output_root / args.dataset / args.model_tag / "feature_banks"
    layers = parse_ints(args.layers or default_layers(search_config))
    resolved_device = args.device or model.default_device

    if args.dry_run:
        print(f"Planned feature bank output root: {feature_root}", flush=True)
        print(f"Will read discovery sources: {discovery_sources_path}", flush=True)
        print(f"Will read dev behavior root: {behavior_root}", flush=True)
        print(f"Will read dev false-suggestion views: {dataset_stage.false_suggestion}", flush=True)
        print(f"Layers: {layers}", flush=True)
        print("Holdout paths will not be read.", flush=True)
        return

    base_records_path = behavior_root / "base_behavior/records.jsonl"
    missing_inputs = [str(path) for path in [discovery_sources_path, base_records_path] if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError(
            "Missing development behavior or split outputs:\n" + "\n".join(missing_inputs)
        )

    discovery_sources = load_discovery_sources(discovery_sources_path)
    views = load_jsonl(dataset_stage.false_suggestion)
    base_records = load_jsonl(base_records_path)
    selected = select_quadrants(
        base_records,
        allowed_sources=discovery_sources,
        max_pairs_per_group=args.max_pairs_per_group,
        seed=args.seed,
    )
    if not selected:
        raise ValueError("No Q1/Q2 records selected from discovery split")
    screen_records = grouped_subset(selected, args.screen_max_train_per_group, args.seed + 17)
    by_pair = pair_lookup(views)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if not tokenizer.is_fast:
        raise ValueError("Feature discovery requires a fast tokenizer")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    needed_pair_ids = {row["pair_id"] for row in selected}
    for pair_id in sorted(needed_pair_ids):
        by_pair[pair_id]["pressure"]["regions"] = locate_regions(tokenizer, by_pair[pair_id]["pressure"])
    zero_target_count = sum(
        len(by_pair[row["pair_id"]]["pressure"]["regions"]["pressure_claim"]) == 0
        for row in selected
    )
    if args.audit_only:
        audit = build_audit(
            behavior_root=behavior_root,
            discovery_sources_path=discovery_sources_path,
            discovery_sources=discovery_sources,
            selected=selected,
            screen_records=screen_records,
            zero_target_count=zero_target_count,
            max_zero_target_count=args.max_zero_target_count,
            layers=layers,
            bank_a=[],
            bank_r=[],
            enforce_nonempty_banks=False,
        )
        audit["mode"] = "audit_only"
        write_json(feature_root / "feature_bank_audit.json", audit)
        print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)
        return

    early_audit = build_audit(
        behavior_root=behavior_root,
        discovery_sources_path=discovery_sources_path,
        discovery_sources=discovery_sources,
        selected=selected,
        screen_records=screen_records,
        zero_target_count=zero_target_count,
        max_zero_target_count=args.max_zero_target_count,
        layers=layers,
        bank_a=[],
        bank_r=[],
        enforce_nonempty_banks=False,
    )
    if early_audit["status"] != "pass":
        write_json(feature_root / "feature_bank_audit.json", early_audit)
        raise SystemExit("Feature bank pre-audit failed; see feature_bank_audit.json")

    model_obj = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if str(resolved_device).startswith("cuda") else torch.float32,
        trust_remote_code=True,
    ).to(resolved_device)
    model_obj.eval()

    bank_a: list[dict[str, Any]] = []
    bank_r: list[dict[str, Any]] = []
    layer_summaries = []
    for layer in layers:
        states = {"Q1": [], "Q2": []}
        for index, qrec in enumerate(selected):
            pressure = by_pair[qrec["pair_id"]]["pressure"]
            target = pressure["regions"]["pressure_claim"]
            if not target:
                continue
            input_ids = torch.tensor([pressure["regions"]["input_ids"]], device=resolved_device)
            hidden = capture_hidden(model_obj, input_ids, layer)
            pooled = pooled_claim_hidden(hidden, target)
            if pooled is None:
                continue
            states["Q1" if qrec["quadrant"] == Q1 else "Q2"].append(pooled.detach())
            if index == 0 or index + 1 == len(selected) or (index + 1) % 40 == 0:
                print(f"layer={layer} discovery capture {index + 1}/{len(selected)}", flush=True)
        if not states["Q1"] or not states["Q2"]:
            continue

        q1_hidden = torch.stack(states["Q1"]).to(resolved_device)
        q2_hidden = torch.stack(states["Q2"]).to(resolved_device)
        robust_direction = q2_hidden.mean(dim=0) - q1_hidden.mean(dim=0)
        sae = load_sae(
            sae_path,
            layer,
            resolved_device,
            sae_variant=model.sae_variant,
            sae_variants_by_layer=model.sae_variants_by_layer,
            sae_activation_fn=model.sae_activation_fn,
            sae_apply_b_dec_to_input=model.sae_apply_b_dec_to_input,
        )
        z_q1 = sae_activation(q1_hidden, sae, resolved_device).mean(dim=0)
        z_q2 = sae_activation(q2_hidden, sae, resolved_device).mean(dim=0)
        robust_projection = direction_projection(
            robust_direction,
            sae["W_dec"],
            z_q1,
            z_q2,
            top_k=args.candidate_k,
        )
        suppression = suppression_projection(
            robust_direction,
            sae["W_dec"],
            z_q1,
            z_q2,
            top_k=args.candidate_k,
        )
        robust_screened = screen_features(
            model_obj,
            tokenizer,
            by_pair,
            screen_records,
            layer,
            sae,
            robust_projection["weighted_ids"][: args.candidate_k],
            robust_direction,
            robust_projection["positive_alignment"],
            +1.0,
            args.screen_strength,
            args.screen_feature_scale,
            args.q2_side_beta,
            args.q2_delta_beta,
            resolved_device,
        )
        suppression_screened = screen_features(
            model_obj,
            tokenizer,
            by_pair,
            screen_records,
            layer,
            sae,
            suppression["ids"][: args.candidate_k],
            robust_direction,
            torch.clamp(-suppression["decoder_alignment"], min=0),
            -1.0,
            args.screen_strength,
            args.screen_feature_scale,
            args.q2_side_beta,
            args.q2_delta_beta,
            resolved_device,
        )
        layer_bank_r = annotate_bank_rows(
            rows=robust_screened,
            bank="R",
            dataset=args.dataset,
            dataset_name=dataset_stage.name,
            model_tag=args.model_tag,
            layer=layer,
            projection_scores=robust_projection["activation_weighted_score"],
            decoder_alignment=robust_projection["decoder_alignment"],
            z_q1=z_q1,
            z_q2=z_q2,
            extra_projection={
                "positive_decoder_alignment": robust_projection["positive_alignment"],
            },
        )
        layer_bank_a = annotate_bank_rows(
            rows=suppression_screened,
            bank="A",
            dataset=args.dataset,
            dataset_name=dataset_stage.name,
            model_tag=args.model_tag,
            layer=layer,
            projection_scores=suppression["score"],
            decoder_alignment=suppression["decoder_alignment"],
            z_q1=z_q1,
            z_q2=z_q2,
            extra_projection={
                "opposite_decoder_alignment": suppression["opposite_alignment"],
                "q1_minus_q2_activation_positive": suppression["q1_enrichment"],
            },
        )
        bank_r.extend(layer_bank_r)
        bank_a.extend(layer_bank_a)
        layer_summaries.append(
            {
                "layer": layer,
                "q1_hidden_count": len(states["Q1"]),
                "q2_hidden_count": len(states["Q2"]),
                "feature_bank_A_count": len(layer_bank_a),
                "feature_bank_R_count": len(layer_bank_r),
                "best_A_causal_score": layer_bank_a[0]["causal_score"] if layer_bank_a else None,
                "best_R_causal_score": layer_bank_r[0]["causal_score"] if layer_bank_r else None,
            }
        )
        del sae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    bank_a.sort(key=lambda row: (row["layer"], row["rank_by_causal_score"] or 10**9))
    bank_r.sort(key=lambda row: (row["layer"], row["rank_by_causal_score"] or 10**9))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset": args.dataset,
        "dataset_stage": "dev",
        "dataset_name": dataset_stage.name,
        "model_tag": args.model_tag,
        "model_path": str(model_path),
        "sae_path": str(sae_path),
        "sae_compatibility": sae_compatibility_metadata(model),
        "split": "discovery",
        "discovery_sources_path": str(discovery_sources_path),
        "behavior_root": str(behavior_root),
        "feature_bank_definitions": {
            "F_A": (
                "Anti-robust / wrong-attractor suppression causal candidates. "
                "They are selected from features with opposite decoder alignment to the Q2-Q1 robust direction "
                "and Q1-enriched activation, then ranked by single-feature causal screening on discovery Q1/Q2 pressure records. "
                "They are not merely top-activated wrong-entity features."
            ),
            "F_R": (
                "Robust-state enhancement causal candidates. "
                "They are selected from activation-weighted features aligned with the Q2-Q1 robust direction "
                "and ranked by single-feature causal screening on discovery Q1/Q2 pressure records."
            ),
        },
        "span_policy": {
            "development_feature_discovery": (
                "May use pressure_claim regions derived from dataset construction and tokenizer span localization "
                "to localize the discovery intervention site."
            ),
            "final_inference": (
                "Must not use gold suggestion spans or dataset-provided pressure_claim regions; "
                "final inference must use candidate extraction / non-gold localization."
            ),
        },
        "layers": layers,
        "candidate_k": args.candidate_k,
        "screen_strength": args.screen_strength,
        "screen_feature_scale": args.screen_feature_scale,
        "screen_max_train_per_group": args.screen_max_train_per_group,
        "q2_side_beta": args.q2_side_beta,
        "q2_delta_beta": args.q2_delta_beta,
        "max_zero_target_count": args.max_zero_target_count,
        "seed": args.seed,
        "selected_counts": count_groups(selected),
        "screen_counts": count_groups(screen_records),
        "zero_target_count": zero_target_count,
        "feature_bank_A_count": len(bank_a),
        "feature_bank_R_count": len(bank_r),
        "layer_summaries": layer_summaries,
        "scope_note": "Candidate discovery only; no calibration or holdout evaluation was performed.",
    }
    audit = build_audit(
        behavior_root=behavior_root,
        discovery_sources_path=discovery_sources_path,
        discovery_sources=discovery_sources,
        selected=selected,
        screen_records=screen_records,
        zero_target_count=zero_target_count,
        max_zero_target_count=args.max_zero_target_count,
        layers=layers,
        bank_a=bank_a,
        bank_r=bank_r,
        enforce_nonempty_banks=True,
    )
    outputs = {
        "feature_bank_A": feature_root / "feature_bank_A.jsonl",
        "feature_bank_R": feature_root / "feature_bank_R.jsonl",
        "feature_bank_summary": feature_root / "feature_bank_summary.json",
        "feature_bank_audit": feature_root / "feature_bank_audit.json",
    }
    write_jsonl(outputs["feature_bank_A"], bank_a)
    write_jsonl(outputs["feature_bank_R"], bank_r)
    write_json(outputs["feature_bank_summary"], summary)
    write_json(outputs["feature_bank_audit"], audit)
    for name, path in outputs.items():
        print(f"Wrote {name}: {path}", flush=True)
    if audit["status"] != "pass":
        raise SystemExit("Feature bank audit failed; see feature_bank_audit.json")


if __name__ == "__main__":
    main()
