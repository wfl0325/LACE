import json

import pytest

from scripts.lib.model_registry import (
    load_model_spec,
    require_residual_sae_compatibility,
)


def write_registry(tmp_path, activation_site: str):
    model_path = tmp_path / "model"
    sae_path = tmp_path / "sae"
    model_path.mkdir()
    sae_path.mkdir()
    config = {
        "models": {
            "test": {
                "model_path": str(model_path),
                "sae_path": str(sae_path),
                "supports_sae_intervention": True,
                "sae_activation_site": activation_site,
                "hook_module": "transformer_block_output",
            }
        }
    }
    config_path = tmp_path / "models.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def test_residual_sae_is_accepted(tmp_path) -> None:
    config_path = write_registry(tmp_path, "residual_post")
    model = load_model_spec(config_path, "test", tmp_path)
    require_residual_sae_compatibility(model)


def test_mlp_sae_is_rejected(tmp_path) -> None:
    config_path = write_registry(tmp_path, "mlp_output")
    model = load_model_spec(config_path, "test", tmp_path)
    with pytest.raises(ValueError, match="residual_post"):
        require_residual_sae_compatibility(model)
