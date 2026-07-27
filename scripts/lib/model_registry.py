from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .io_utils import is_placeholder, load_json, resolve_path


@dataclass(frozen=True)
class ModelSpec:
    tag: str
    display_name: str
    model_path: Path
    sae_path: Path | None
    model_family: str
    dtype: str
    default_device: str
    supports_sae_intervention: bool
    sae_activation_site: str | None
    hook_module: str | None
    sae_release: str | None
    sae_variant: str | None
    sae_activation_fn: str = "relu_legacy_lace"
    sae_apply_b_dec_to_input: bool = False
    sae_normalize_activations: str | None = None
    sae_variants_by_layer: dict[str, str] = field(default_factory=dict)


def load_model_spec(config_path: Path, model_tag: str, root: Path, allow_placeholder: bool = False) -> ModelSpec:
    config = load_json(config_path)
    models = config.get("models", {})
    if model_tag not in models:
        raise KeyError(f"Unknown model_tag={model_tag!r}; available={sorted(models)}")
    entry = models[model_tag]
    raw_model_path = entry.get("model_path")
    if is_placeholder(raw_model_path) and not allow_placeholder:
        raise ValueError(
            f"model_path for {model_tag!r} is a placeholder. "
            "Copy configs/models.example.json to configs/models.local.json and fill local paths."
        )
    raw_sae_path = entry.get("sae_path")
    sae_path = None if is_placeholder(raw_sae_path) else resolve_path(raw_sae_path, root)
    return ModelSpec(
        tag=model_tag,
        display_name=entry.get("display_name", model_tag),
        model_path=resolve_path(raw_model_path, root),
        sae_path=sae_path,
        model_family=entry.get("model_family", "unknown"),
        dtype=entry.get("dtype", "bfloat16"),
        default_device=entry.get("default_device", "cuda:0"),
        supports_sae_intervention=bool(entry.get("supports_sae_intervention", False)),
        sae_activation_site=entry.get("sae_activation_site"),
        hook_module=entry.get("hook_module"),
        sae_release=entry.get("sae_release"),
        sae_variant=entry.get("sae_variant"),
        sae_activation_fn=entry.get("sae_activation_fn", "relu_legacy_lace"),
        sae_apply_b_dec_to_input=bool(entry.get("sae_apply_b_dec_to_input", False)),
        sae_normalize_activations=entry.get("sae_normalize_activations"),
        sae_variants_by_layer={
            str(layer): str(variant)
            for layer, variant in entry.get("sae_variants_by_layer", {}).items()
        },
    )


def require_residual_sae_compatibility(model: ModelSpec) -> None:
    """Fail before inference if an SAE does not match LACE's residual hook."""

    if not model.supports_sae_intervention:
        raise ValueError(f"Model {model.tag!r} is not enabled for SAE intervention")
    if model.sae_activation_site != "residual_post":
        raise ValueError(
            f"Model {model.tag!r} declares sae_activation_site={model.sae_activation_site!r}. "
            "LACE captures and edits Transformer block residual outputs, so the configured SAE "
            "must be trained on residual_post activations."
        )
    if model.hook_module != "transformer_block_output":
        raise ValueError(
            f"Model {model.tag!r} declares hook_module={model.hook_module!r}; "
            "expected 'transformer_block_output'."
        )
    if model.sae_activation_fn not in {"relu_legacy_lace", "jumprelu"}:
        raise ValueError(f"Unsupported sae_activation_fn={model.sae_activation_fn!r}")
    if model.sae_normalize_activations not in {None, "none"}:
        raise ValueError(
            "LACE does not silently approximate normalized SAE inputs; "
            f"got sae_normalize_activations={model.sae_normalize_activations!r}."
        )
    if model.sae_path and "pt-mlp" in str(model.sae_path).lower().replace("_", "-"):
        raise ValueError(
            f"Model {model.tag!r} points to an MLP SAE release but LACE hooks residual_post: "
            f"{model.sae_path}"
        )


def sae_compatibility_metadata(model: ModelSpec) -> dict[str, Any]:
    return {
        "sae_activation_site": model.sae_activation_site,
        "hook_module": model.hook_module,
        "sae_release": model.sae_release,
        "sae_variant": model.sae_variant,
        "sae_variants_by_layer": model.sae_variants_by_layer,
        "sae_activation_fn": model.sae_activation_fn,
        "sae_apply_b_dec_to_input": model.sae_apply_b_dec_to_input,
        "sae_normalize_activations": model.sae_normalize_activations,
        "compatibility_check": "residual_post_sae_with_transformer_block_output",
    }
