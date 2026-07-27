from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def get_hidden(output):
    return output[0] if isinstance(output, tuple) else output


def replace_hidden(output, hidden):
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    return hidden


def require_direct_answer_prefix(prefix: str) -> str:
    """Reject chat prefixes that would score candidates inside a thinking block."""

    if prefix.rfind("<think>") > prefix.rfind("</think>"):
        raise ValueError(
            "Chat template left an open <think> block in the generation prefix; "
            "candidate-margin scoring requires enable_thinking=False."
        )
    return prefix


def make_chat_prefix(tokenizer, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return require_direct_answer_prefix(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
    return prompt


def encode_candidate(tokenizer, candidate: str, device: str) -> torch.Tensor:
    ids = tokenizer(candidate, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    if ids.shape[1] == 0:
        raise ValueError(f"empty candidate after tokenization: {candidate!r}")
    return ids


def candidate_logprob_from_logits(
    logits: torch.Tensor,
    prompt_len: int,
    candidate_ids: torch.Tensor,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    start = prompt_len - 1
    values = [
        log_probs[0, start + offset, candidate_ids[0, offset]]
        for offset in range(candidate_ids.shape[1])
    ]
    return torch.stack(values).mean()


def capture_hidden(model, input_ids: torch.Tensor, layer: int) -> torch.Tensor:
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        captured["hidden"] = get_hidden(output).detach()

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
    finally:
        handle.remove()
    return captured["hidden"][0].float()


def candidate_scores(
    model,
    tokenizer,
    example: dict[str, Any],
    layer: int,
    device: str,
    target_positions: list[int] | None = None,
    patch_delta: torch.Tensor | None = None,
    replace_values: torch.Tensor | None = None,
) -> dict[str, float]:
    prefix = make_chat_prefix(tokenizer, example["mc_prompt"])
    prompt_ids = tokenizer(prefix, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    positive_ids = encode_candidate(tokenizer, example["correct_option"], device)
    negative_ids = encode_candidate(tokenizer, example["wrong_option"], device)
    prompt_len = prompt_ids.shape[1]
    positive_full = torch.cat([prompt_ids, positive_ids], dim=1)
    negative_full = torch.cat([prompt_ids, negative_ids], dim=1)
    max_len = max(positive_full.shape[1], negative_full.shape[1])
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def pad_to(tensor: torch.Tensor, length: int) -> torch.Tensor:
        if tensor.shape[1] == length:
            return tensor
        pad = torch.full(
            (tensor.shape[0], length - tensor.shape[1]),
            pad_id,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        return torch.cat([tensor, pad], dim=1)

    input_ids = torch.cat([pad_to(positive_full, max_len), pad_to(negative_full, max_len)], dim=0)
    attention_mask = torch.cat(
        [
            torch.cat(
                [
                    torch.ones_like(positive_full),
                    torch.zeros(
                        (1, max_len - positive_full.shape[1]),
                        dtype=positive_full.dtype,
                        device=positive_full.device,
                    ),
                ],
                dim=1,
            ),
            torch.cat(
                [
                    torch.ones_like(negative_full),
                    torch.zeros(
                        (1, max_len - negative_full.shape[1]),
                        dtype=negative_full.dtype,
                        device=negative_full.device,
                    ),
                ],
                dim=1,
            ),
        ],
        dim=0,
    )

    handle = None
    if target_positions and (patch_delta is not None or replace_values is not None):
        delta = patch_delta.to(device=device, dtype=model.dtype) if patch_delta is not None else None
        replacement = (
            replace_values.to(device=device, dtype=model.dtype)
            if replace_values is not None
            else None
        )

        def hook(_module, _inputs, output):
            hidden = get_hidden(output)
            patched = hidden.clone()
            if replacement is not None:
                patched[:, target_positions, :] = replacement
            else:
                patched[:, target_positions, :] += delta
            return replace_hidden(output, patched)

        handle = model.model.layers[layer].register_forward_hook(hook)

    try:
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    finally:
        if handle is not None:
            handle.remove()

    logp_correct = candidate_logprob_from_logits(logits[0:1], prompt_len, positive_ids).item()
    logp_wrong = candidate_logprob_from_logits(logits[1:2], prompt_len, negative_ids).item()
    return {
        "logp_correct": float(logp_correct),
        "logp_wrong": float(logp_wrong),
        "margin": float(logp_correct - logp_wrong),
    }
