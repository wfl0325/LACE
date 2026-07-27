from __future__ import annotations

from collections import defaultdict
from typing import Any


def parse_layer_spec(raw: str) -> list[int]:
    """Parse layer specs such as ``10-14`` or ``10,12,14``."""
    layers: list[int] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            left, right = piece.split("-", 1)
            layers.extend(range(int(left), int(right) + 1))
        else:
            layers.append(int(piece))
    return sorted(set(layers))


def parse_bands(raw: str) -> dict[str, list[int]]:
    """Parse band specs such as ``peak:10-14;readout:20-23``."""
    output: dict[str, list[int]] = {}
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        name, spec = item.split(":", 1)
        output[name.strip()] = parse_layer_spec(spec.strip())
    return output


def feature_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Global causal-feature ordering used inside a layer band.

    Older scripts selected top-k features from a band as a global list and then
    grouped the chosen features by layer for multi-layer intervention. We keep
    that semantics here: top-k is not per layer.
    """
    causal_score = float(row.get("causal_score") or 0.0)
    return (
        -causal_score,
        row.get("rank_by_causal_score") is None,
        int(row.get("rank_by_causal_score") or 10**9),
        int(row.get("layer") or 0),
        int(row.get("feature_id") or 0),
    )


def select_band_features_global(
    by_layer: dict[int, list[dict[str, Any]]],
    *,
    layers: list[int],
    top_k: int,
) -> list[dict[str, Any]]:
    """Select a global top-k feature list from all layers in a band."""
    if top_k <= 0:
        return []
    candidates: list[dict[str, Any]] = []
    layer_set = {int(layer) for layer in layers}
    for layer in sorted(layer_set):
        candidates.extend(by_layer.get(layer, []))
    candidates.sort(key=feature_sort_key)
    return candidates[:top_k]


def features_by_layer(candidates: list[dict[str, Any]], key: str = "feature_id") -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for candidate in candidates:
        grouped[int(candidate["layer"])].append(int(candidate[key]))
    return dict(sorted(grouped.items()))


def feature_uids_by_layer(candidates: list[dict[str, Any]], layers: list[int]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {str(int(layer)): [] for layer in layers}
    for candidate in candidates:
        grouped.setdefault(str(int(candidate["layer"])), []).append(str(candidate["feature_uid"]))
    return dict(sorted(grouped.items(), key=lambda item: int(item[0])))


def selected_feature_uids_by_layer_global(
    by_layer: dict[int, list[dict[str, Any]]],
    *,
    layers: list[int],
    top_k: int,
) -> dict[str, list[str]]:
    return feature_uids_by_layer(
        select_band_features_global(by_layer, layers=layers, top_k=top_k),
        layers,
    )


def selected_feature_count_by_layer(feature_uids_by_layer_map: dict[str, list[str]]) -> int:
    return sum(len(items) for items in feature_uids_by_layer_map.values())
