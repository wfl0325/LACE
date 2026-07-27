from __future__ import annotations

import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model_primitives import candidate_scores


Q1 = "Q1_clean_correct_pressure_wrong"
Q2 = "Q2_clean_correct_pressure_correct"


def pair_lookup(rows: list[dict]) -> dict[str, dict[str, dict]]:
    output: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        output[row["pair_id"]][row["trigger_condition"]] = row
    return output


def select_quadrants(
    records: list[dict],
    allowed_sources: set[str] | None,
    max_pairs_per_group: int,
    seed: int,
) -> list[dict]:
    grouped = {"Q1": [], "Q2": []}
    for row in records:
        if row.get("record_type") != "false_suggestion":
            continue
        if row.get("quadrant") not in {Q1, Q2}:
            continue
        if allowed_sources is not None and row["source_id"] not in allowed_sources:
            continue
        grouped["Q1" if row["quadrant"] == Q1 else "Q2"].append(row)

    rng = random.Random(seed)
    selected: list[dict] = []
    for group in ["Q1", "Q2"]:
        rows = sorted(grouped[group], key=lambda item: item["pair_id"])
        if max_pairs_per_group > 0:
            rng.shuffle(rows)
            rows = sorted(rows[:max_pairs_per_group], key=lambda item: item["pair_id"])
        selected.extend(rows)
    return selected


def grouped_subset(records: list[dict], max_per_group: int, seed: int) -> list[dict]:
    if max_per_group <= 0:
        return list(records)
    grouped = {"Q1": [], "Q2": []}
    for row in records:
        grouped["Q1" if row["quadrant"] == Q1 else "Q2"].append(row)
    rng = random.Random(seed)
    output: list[dict] = []
    for group in ["Q1", "Q2"]:
        rows = list(grouped[group])
        rng.shuffle(rows)
        output.extend(rows[:max_per_group])
    return output


def pooled_claim_hidden(hidden: torch.Tensor, positions: list[int]) -> torch.Tensor | None:
    if not positions:
        return None
    return hidden[positions].float().mean(dim=0)


def _average_l0(path: Path) -> float:
    match = re.search(r"average_l0_([0-9.]+)", str(path))
    return float(match.group(1)) if match else float("inf")


def _resolve_sae_file(
    sae_dir: Path,
    layer: int,
    sae_variant: str | None = None,
    sae_variants_by_layer: dict[str, str] | None = None,
) -> Path:
    qwen_path = sae_dir / f"layer{layer}.sae.pt"
    if qwen_path.exists():
        return qwen_path
    direct_npz = sae_dir / "params.npz"
    if direct_npz.exists():
        return direct_npz
    configured_variant = (sae_variants_by_layer or {}).get(str(layer)) or sae_variant
    if configured_variant:
        if configured_variant.startswith(f"layer_{layer}/"):
            configured = sae_dir / configured_variant / "params.npz"
        else:
            configured = sae_dir / f"layer_{layer}" / configured_variant / "params.npz"
        if configured.exists():
            return configured
        raise FileNotFoundError(
            f"Configured SAE variant {configured_variant!r} was not found for layer {layer}: "
            f"{configured}"
        )
    candidates = sorted(
        (sae_dir / f"layer_{layer}").glob("width_*/average_l0_*/params.npz"),
        key=lambda path: (_average_l0(path), str(path)),
    )
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple SAE variants found for layer {layer} under {sae_dir}. "
            "Set sae_variant in configs/models.local.json."
        )
    raise FileNotFoundError(f"No SAE file found for layer {layer} under {sae_dir}")


def load_sae(
    sae_dir: Path,
    layer: int,
    device: str,
    sae_variant: str | None = None,
    sae_variants_by_layer: dict[str, str] | None = None,
    sae_activation_fn: str = "relu_legacy_lace",
    sae_apply_b_dec_to_input: bool = False,
) -> dict[str, Any]:
    sae_file = _resolve_sae_file(sae_dir, layer, sae_variant, sae_variants_by_layer)
    if "pt-mlp" in str(sae_file).lower().replace("_", "-"):
        raise ValueError(f"Refusing to load an MLP SAE for LACE's residual_post hook: {sae_file}")
    if sae_file.suffix == ".npz":
        state = np.load(sae_file)
        W_dec = torch.as_tensor(state["W_dec"], dtype=torch.float32, device=device)
        W_enc = torch.as_tensor(state["W_enc"], dtype=torch.float32, device=device)
        b_enc = torch.as_tensor(state["b_enc"], dtype=torch.float32, device=device)
        b_dec = (
            torch.as_tensor(state["b_dec"], dtype=torch.float32, device=device)
            if "b_dec" in state.files
            else None
        )
        threshold = (
            torch.as_tensor(state["threshold"], dtype=torch.float32, device=device)
            if "threshold" in state.files
            else None
        )
        if W_dec.shape[0] == W_enc.shape[1]:
            W_dec = W_dec.T.contiguous()
            W_enc = W_enc.T.contiguous()
        return {
            "W_enc": W_enc,
            "W_dec": W_dec,
            "b_enc": b_enc,
            "b_dec": b_dec,
            "threshold": threshold,
            "sae_file": str(sae_file),
            "sae_format": "gemma_scope_npz",
            "sae_activation_fn": sae_activation_fn,
            "sae_apply_b_dec_to_input": sae_apply_b_dec_to_input,
        }
    state = torch.load(sae_file, map_location="cpu")
    b_dec = state.get("b_dec")
    return {
        "W_enc": state["W_enc"].to(device=device, dtype=torch.float32),
        "W_dec": state["W_dec"].to(device=device, dtype=torch.float32),
        "b_enc": state["b_enc"].to(device=device, dtype=torch.float32),
        "b_dec": b_dec.to(device=device, dtype=torch.float32) if b_dec is not None else None,
        "threshold": None,
        "sae_file": str(sae_file),
        "sae_format": "torch_sae_pt",
        "sae_activation_fn": sae_activation_fn,
        "sae_apply_b_dec_to_input": sae_apply_b_dec_to_input,
    }


def sae_activation(
    hidden: torch.Tensor,
    sae: dict[str, Any],
    device: str,
    feature_ids: torch.Tensor | list[int] | None = None,
) -> torch.Tensor:
    hidden = hidden.to(device=device, dtype=torch.float32)
    if sae.get("sae_apply_b_dec_to_input", False):
        b_dec = sae.get("b_dec")
        if b_dec is None:
            raise ValueError("SAE input centering requested but b_dec is unavailable")
        hidden = hidden - b_dec
    if feature_ids is None:
        W_enc = sae["W_enc"]
        b_enc = sae["b_enc"]
        threshold = sae.get("threshold")
    else:
        ids = torch.as_tensor(feature_ids, dtype=torch.long, device=device)
        W_enc = sae["W_enc"][ids]
        b_enc = sae["b_enc"][ids]
        threshold = sae.get("threshold")
        threshold = threshold[ids] if threshold is not None else None
    pre_acts = hidden @ W_enc.T + b_enc
    activation_fn = sae.get("sae_activation_fn", "relu_legacy_lace")
    if activation_fn == "relu_legacy_lace":
        return torch.relu(pre_acts)
    if activation_fn == "jumprelu":
        if threshold is None:
            raise ValueError(f"JumpReLU SAE has no threshold: {sae.get('sae_file')}")
        base_acts = torch.relu(pre_acts)
        return base_acts * (pre_acts > threshold).to(base_acts.dtype)
    raise ValueError(f"Unsupported SAE activation function: {activation_fn!r}")


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denominator = a.float().norm() * b.float().norm()
    if denominator.item() == 0:
        return 0.0
    return float(torch.dot(a.float(), b.float()).item() / denominator.item())


def direction_projection(
    direction: torch.Tensor,
    W_dec: torch.Tensor,
    z_q1: torch.Tensor,
    z_q2: torch.Tensor,
    top_k: int,
) -> dict[str, Any]:
    direction_norm = direction.float().norm() + 1e-12
    unit_direction = direction.float() / direction_norm
    decoder_alignment = W_dec.T @ unit_direction
    positive_alignment = torch.clamp(decoder_alignment, min=0)
    activation_enrichment = torch.clamp(z_q2 - z_q1, min=0)
    activation_weighted_score = positive_alignment * activation_enrichment
    k = min(top_k, W_dec.shape[1])
    aligned_values, aligned_ids = torch.topk(positive_alignment, k=k)
    weighted_values, weighted_ids = torch.topk(activation_weighted_score, k=k)
    aligned_reconstruction = W_dec[:, aligned_ids] @ aligned_values
    weighted_reconstruction = W_dec[:, weighted_ids] @ weighted_values
    return {
        "direction_norm": float(direction_norm.item()),
        "decoder_alignment": decoder_alignment,
        "positive_alignment": positive_alignment,
        "activation_weighted_score": activation_weighted_score,
        "aligned_ids": aligned_ids,
        "weighted_ids": weighted_ids,
        "top_alignment_reconstruction_cosine": cosine(aligned_reconstruction, direction),
        "top_weighted_reconstruction_cosine": cosine(weighted_reconstruction, direction),
    }


def suppression_projection(
    robust_direction: torch.Tensor,
    W_dec: torch.Tensor,
    z_q1: torch.Tensor,
    z_q2: torch.Tensor,
    top_k: int,
) -> dict[str, Any]:
    direction_norm = robust_direction.float().norm() + 1e-12
    unit_direction = robust_direction.float() / direction_norm
    decoder_alignment = W_dec.T @ unit_direction
    q1_enrichment = torch.clamp(z_q1 - z_q2, min=0)
    opposite_alignment = torch.clamp(-decoder_alignment, min=0)
    score = q1_enrichment * opposite_alignment
    k = min(top_k, W_dec.shape[1])
    values, ids = torch.topk(score, k=k)
    return {
        "ids": ids,
        "values": values,
        "score": score,
        "decoder_alignment": decoder_alignment,
        "opposite_alignment": opposite_alignment,
        "q1_enrichment": q1_enrichment,
    }


def _mean(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.fmean(values) if values else None


def _summarize(records: list[dict]) -> dict[str, Any]:
    q1 = [row for row in records if row["quadrant_group"] == "Q1"]
    q2 = [row for row in records if row["quadrant_group"] == "Q2"]
    return {
        "n": len(records),
        "q1_n": len(q1),
        "q2_n": len(q2),
        "q1_correction_rate": (
            sum(row["intervened_margin"] > 0 for row in q1) / len(q1) if q1 else None
        ),
        "q1_mean_delta_margin": _mean(q1, "delta_margin"),
        "q1_mean_delta_wrong_drop": _mean(q1, "delta_wrong_drop"),
        "q1_improved_fraction": (
            sum(row["delta_margin"] > 0 for row in q1) / len(q1) if q1 else None
        ),
        "q2_side_effect_rate": (
            sum(row["intervened_margin"] < 0 for row in q2) / len(q2) if q2 else None
        ),
        "q2_mean_delta_margin": _mean(q2, "delta_margin"),
        "q2_mean_delta_wrong_drop": _mean(q2, "delta_wrong_drop"),
        "q2_improved_fraction": (
            sum(row["delta_margin"] > 0 for row in q2) / len(q2) if q2 else None
        ),
    }


def _patch_records(
    model,
    tokenizer,
    by_pair: dict[str, dict[str, dict]],
    records: list[dict],
    layer: int,
    vector: torch.Tensor,
    strength: float,
    mode: str,
    device: str,
) -> list[dict]:
    output: list[dict] = []
    patch_delta = (float(strength) * vector).to(device=device, dtype=torch.float32)
    for record in records:
        pressure = by_pair[record["pair_id"]]["pressure"]
        target = pressure["regions"]["pressure_claim"]
        if not target:
            continue
        intervened = candidate_scores(
            model,
            tokenizer,
            pressure,
            layer,
            device,
            target_positions=target,
            patch_delta=patch_delta.repeat(len(target), 1),
        )
        baseline_correct = record["pressure_logp_correct"]
        baseline_wrong = record["pressure_logp_wrong"]
        baseline_margin = record["pressure_margin"]
        output.append(
            {
                "layer": layer,
                "mode": mode,
                "strength": strength,
                "pair_id": record["pair_id"],
                "source_id": record["source_id"],
                "template_id": record["template_id"],
                "relation": record.get("relation"),
                "quadrant": record["quadrant"],
                "quadrant_group": "Q1" if record["quadrant"] == Q1 else "Q2",
                "target_token_count": len(target),
                "baseline_logp_correct": baseline_correct,
                "baseline_logp_wrong": baseline_wrong,
                "baseline_margin": baseline_margin,
                "intervened_logp_correct": intervened["logp_correct"],
                "intervened_logp_wrong": intervened["logp_wrong"],
                "intervened_margin": intervened["margin"],
                "delta_logp_correct": intervened["logp_correct"] - baseline_correct,
                "delta_logp_wrong": intervened["logp_wrong"] - baseline_wrong,
                "delta_wrong_drop": baseline_wrong - intervened["logp_wrong"],
                "delta_margin": intervened["margin"] - baseline_margin,
            }
        )
    return output


def _feature_direction(
    W_dec: torch.Tensor,
    feature_id: int,
    residual_norm: torch.Tensor,
    positive_alignment: torch.Tensor,
    scale: str,
) -> torch.Tensor:
    column = W_dec[:, feature_id].float()
    if scale == "unit":
        norm = column.norm()
        if norm.item() > 0:
            column = column / norm
    elif scale == "alignment":
        column = residual_norm * positive_alignment[feature_id] * column
    elif scale != "raw":
        raise ValueError(f"unknown feature scale: {scale}")
    return column


def _causal_score(
    summary: dict[str, Any],
    q2_side_beta: float,
    q2_delta_beta: float,
) -> float:
    q1_delta = float(summary.get("q1_mean_delta_margin") or 0.0)
    q2_side = float(summary.get("q2_side_effect_rate") or 0.0)
    q2_delta = float(summary.get("q2_mean_delta_margin") or 0.0)
    return q1_delta - q2_side_beta * q2_side - q2_delta_beta * max(0.0, -q2_delta)


def screen_features(
    model,
    tokenizer,
    by_pair: dict[str, dict[str, dict]],
    screen_records: list[dict],
    layer: int,
    sae: dict[str, torch.Tensor],
    feature_ids: torch.Tensor,
    robust_direction: torch.Tensor,
    positive_alignment: torch.Tensor,
    sign: float,
    screen_strength: float,
    screen_feature_scale: str,
    q2_side_beta: float,
    q2_delta_beta: float,
    device: str,
) -> list[dict]:
    residual_norm = robust_direction.float().norm() + 1e-12
    decoder_norms = sae["W_dec"].float().norm(dim=0)
    rows: list[dict] = []
    for index, feature_tensor in enumerate(feature_ids, start=1):
        feature_id = int(feature_tensor.item())
        vector = _feature_direction(
            sae["W_dec"],
            feature_id,
            residual_norm,
            positive_alignment,
            screen_feature_scale,
        )
        if sign < 0:
            vector = -vector
        records = _patch_records(
            model,
            tokenizer,
            by_pair,
            screen_records,
            layer,
            vector.detach().cpu(),
            screen_strength,
            f"screen_feature_{feature_id}",
            device,
        )
        summary = _summarize(records)
        rows.append(
            {
                "feature_id": feature_id,
                "candidate_rank": index,
                "screen_strength": screen_strength,
                "screen_sign": sign,
                "causal_score": _causal_score(summary, q2_side_beta, q2_delta_beta),
                "decoder_norm": float(decoder_norms[feature_id].item()),
                **summary,
            }
        )
        if index == 1 or index == feature_ids.numel() or index % 25 == 0:
            print(
                f"layer={layer} sign={sign:+.0f} screened {index}/{feature_ids.numel()} "
                f"best={max(row['causal_score'] for row in rows):.4f}",
                flush=True,
            )
    rows.sort(key=lambda row: row["causal_score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank_by_causal_score"] = rank
    return rows
