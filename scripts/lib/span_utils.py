from __future__ import annotations

from .model_primitives import make_chat_prefix


PRESSURE_CUE_TERMS = ("experts", "unanimously", "concluded", "consensus")
NEUTRAL_NOUN_TERMS = ("answer",)


def token_positions_for_span(
    offsets: list[tuple[int, int]],
    start: int,
    end: int,
) -> list[int]:
    return [
        index
        for index, (left, right) in enumerate(offsets)
        if right > left and right > start and left < end
    ]


def locate_regions(tokenizer, example: dict) -> dict:
    prefix = make_chat_prefix(tokenizer, example["mc_prompt"])
    encoded = tokenizer(prefix, add_special_tokens=True, return_offsets_mapping=True)
    input_ids = list(encoded["input_ids"])
    offsets = [tuple(pair) for pair in encoded["offset_mapping"]]

    prompt_start = prefix.find(example["mc_prompt"])
    if prompt_start < 0:
        raise ValueError(f"mc_prompt not found in chat prefix: {example.get('view_id')!r}")
    prompt_end = prompt_start + len(example["mc_prompt"])
    body_start = prefix.find(example["body"], prompt_start, prompt_end)
    if body_start < 0:
        raise ValueError(f"body not found in prompt: {example.get('view_id')!r}")
    body_end = body_start + len(example["body"])
    lowered_prefix = prefix.lower()

    def all_term_positions(term: str, start: int, end: int) -> list[int]:
        positions: list[int] = []
        if not term:
            return positions
        search_start = start
        lowered_term = term.lower()
        while True:
            term_start = lowered_prefix.find(lowered_term, search_start, end)
            if term_start < 0:
                break
            positions.extend(
                token_positions_for_span(offsets, term_start, term_start + len(term))
            )
            search_start = term_start + max(1, len(term))
        return positions

    pressure_claim = sorted(
        set(all_term_positions(example["wrong_option"], body_start, body_end))
    )
    pressure_cue = sorted(
        {
            position
            for term in PRESSURE_CUE_TERMS
            for position in all_term_positions(term, body_start, body_end)
        }
    )
    neutral_noun = sorted(
        {
            position
            for term in NEUTRAL_NOUN_TERMS
            for position in all_term_positions(term, body_start, body_end)
        }
    )

    return {
        "input_ids": input_ids,
        "offsets": offsets,
        "final": [len(input_ids) - 1],
        "prompt": token_positions_for_span(offsets, prompt_start, prompt_end),
        "body": token_positions_for_span(offsets, body_start, body_end),
        "pressure_claim": pressure_claim,
        "pressure_cue": pressure_cue,
        "neutral_noun": neutral_noun,
    }
