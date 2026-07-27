import torch

from scripts.lib.feature_primitives import sae_activation


def base_sae(activation_fn: str):
    return {
        "W_enc": torch.eye(2),
        "W_dec": torch.eye(2),
        "b_enc": torch.zeros(2),
        "threshold": torch.tensor([1.0, 2.0]),
        "sae_activation_fn": activation_fn,
        "sae_file": "synthetic",
    }


def test_qwen_legacy_relu_policy() -> None:
    hidden = torch.tensor([[0.5, 3.0]])
    actual = sae_activation(hidden, base_sae("relu_legacy_lace"), "cpu")
    assert torch.equal(actual, torch.tensor([[0.5, 3.0]]))


def test_gemma_jumprelu_uses_per_feature_threshold() -> None:
    hidden = torch.tensor([[0.5, 3.0]])
    actual = sae_activation(hidden, base_sae("jumprelu"), "cpu")
    assert torch.equal(actual, torch.tensor([[0.0, 3.0]]))


def test_selected_feature_encoding_uses_matching_threshold() -> None:
    hidden = torch.tensor([[0.5, 3.0]])
    actual = sae_activation(
        hidden,
        base_sae("jumprelu"),
        "cpu",
        feature_ids=torch.tensor([1]),
    )
    assert torch.equal(actual, torch.tensor([[3.0]]))
