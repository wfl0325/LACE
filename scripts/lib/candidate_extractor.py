from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExtractedCandidate:
    text: str
    start: int
    end: int
    pattern_name: str


PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "think_answer_is_confirm_prefix",
        re.compile(r"\bI think the answer is (?P<candidate>.+?)\.\s+Is that right\?\s", re.IGNORECASE),
    ),
    (
        "confirm_whether_answer_is",
        re.compile(r"\bCould you confirm whether the answer is (?P<candidate>.+?)\?\s", re.IGNORECASE),
    ),
    (
        "think_answer_is_confirm_suffix",
        re.compile(r"\sI think the answer is (?P<candidate>.+?)\.\s+Is that right\?\s*$", re.IGNORECASE),
    ),
)

HEARD_ANSWER_PREFIX = re.compile(r"\bI heard the answer might be\s+", re.IGNORECASE)
ABBREVIATION_TAILS = {
    "dr",
    "mr",
    "mrs",
    "ms",
    "mt",
    "no",
    "prof",
    "st",
}
ROMAN_NUMERAL_TAILS = {"I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"}


def _looks_like_unfinished_abbreviation(candidate: str) -> bool:
    tail = candidate.rstrip().rsplit(" ", 1)[-1].rstrip(".")
    if tail in ROMAN_NUMERAL_TAILS:
        return False
    if len(tail) == 1 and tail.isalpha() and tail.upper() == tail:
        return True
    return tail.casefold() in ABBREVIATION_TAILS


def _extract_heard_answer_might_be(text: str) -> ExtractedCandidate | None:
    match = HEARD_ANSWER_PREFIX.search(text)
    if not match:
        return None

    candidate_start = match.end()
    for period in re.finditer(r"\.\s+", text[candidate_start:]):
        candidate_end = candidate_start + period.start()
        raw_candidate = text[candidate_start:candidate_end].strip()
        if not raw_candidate:
            continue
        if _looks_like_unfinished_abbreviation(raw_candidate):
            continue
        start, end = candidate_start, candidate_end
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        candidate = text[start:end]
        if candidate:
            return ExtractedCandidate(
                text=candidate,
                start=start,
                end=end,
                pattern_name="heard_answer_might_be",
            )
    return None


def extract_suggested_candidate(text: str) -> ExtractedCandidate | None:
    """Extract a suggested-answer span from prompt text without using gold spans.

    This helper intentionally uses only the prompt string. It must not read
    dataset-provided fields such as suggestion_char_start, suggestion_char_end,
    suggestion_prompt_char_start, or pressure_claim annotations.
    """

    heard_candidate = _extract_heard_answer_might_be(text)
    if heard_candidate is not None:
        return heard_candidate

    for name, pattern in PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        candidate = match.group("candidate").strip()
        if not candidate:
            continue
        start, end = match.span("candidate")
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        candidate = text[start:end]
        if candidate:
            return ExtractedCandidate(text=candidate, start=start, end=end, pattern_name=name)
    return None


def extractor_spec() -> dict[str, object]:
    return {
        "name": "template_or_regex_v1",
        "span_mode": "extracted",
        "uses_gold_span_fields": False,
        "default_candidate_source": "row_suggested_option_then_prompt_regex",
        "structured_candidate_allowed_in_final_method": True,
        "structured_candidate_note": (
            "suggested_option is the user-provided candidate entity, not the gold factual answer. "
            "Gold span offsets and pressure_claim annotations remain forbidden."
        ),
        "forbidden_fields": [
            "suggestion_char_start",
            "suggestion_char_end",
            "suggestion_prompt_char_start",
            "suggestion_prompt_char_end",
            "pressure_claim",
        ],
        "patterns": [name for name, _ in PATTERNS],
        "fallback": "no_op_if_no_candidate_extracted",
    }


def token_positions_for_span(offsets: list[tuple[int, int]], start: int, end: int) -> list[int]:
    return [
        index
        for index, (left, right) in enumerate(offsets)
        if right > left and right > start and left < end
    ]


def find_text_occurrences(text: str, needle: str, *, case_sensitive: bool = False) -> list[tuple[int, int]]:
    if not needle:
        return []
    haystack = text if case_sensitive else text.casefold()
    target = needle if case_sensitive else needle.casefold()
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        found = haystack.find(target, start)
        if found < 0:
            break
        spans.append((found, found + len(needle)))
        start = found + max(1, len(needle))
    return spans


def locate_candidate_positions(
    tokenizer,
    row: dict[str, Any],
    *,
    candidate_text: str | None = None,
    allow_structured_candidate: bool = True,
    prompt_key: str = "mc_prompt",
    search_scope: str = "prompt",
    case_sensitive: bool = False,
) -> dict[str, Any]:
    """Locate candidate token positions using prompt text only.

    This is the shared no-gold locator used by baselines and LACE base calibration.
    ``row["suggested_option"]`` is allowed as a structured copy of the user
    provided candidate entity. It is not the gold factual answer. This helper
    must never read dataset span annotations such as suggestion_char_start/end
    or pressure_claim.
    """

    if search_scope not in {"prompt", "body"}:
        raise ValueError("search_scope must be 'prompt' or 'body'")

    from .model_primitives import make_chat_prefix

    prompt = row[prompt_key]
    extracted = extract_suggested_candidate(prompt)
    candidate_source = "none"
    if allow_structured_candidate and candidate_text is not None:
        candidate = candidate_text
        candidate_source = "argument_candidate_text"
    elif allow_structured_candidate and row.get("suggested_option"):
        candidate = row.get("suggested_option") or ""
        candidate_source = "row_suggested_option"
    elif extracted is not None:
        candidate = extracted.text
        candidate_source = "prompt_regex"
    else:
        candidate = ""

    chat_prompt = make_chat_prefix(tokenizer, prompt)
    encoded = tokenizer(chat_prompt, add_special_tokens=True, return_offsets_mapping=True)
    input_ids = list(encoded["input_ids"])
    offsets = [tuple(pair) for pair in encoded["offset_mapping"]]

    prompt_start = chat_prompt.find(prompt)
    if prompt_start < 0:
        raise ValueError(f"{prompt_key} not found in chat prompt for view_id={row.get('view_id')!r}")
    prompt_end = prompt_start + len(prompt)

    if search_scope == "body" and row.get("body"):
        body_start = chat_prompt.find(row["body"], prompt_start, prompt_end)
        if body_start < 0:
            search_start, search_end = prompt_start, prompt_end
            resolved_scope = "prompt_body_missing_fallback"
        else:
            search_start, search_end = body_start, body_start + len(row["body"])
            resolved_scope = "body"
    else:
        search_start, search_end = prompt_start, prompt_end
        resolved_scope = "prompt"

    search_text = chat_prompt[search_start:search_end]
    relative_spans = find_text_occurrences(search_text, candidate, case_sensitive=case_sensitive)
    absolute_spans = [(search_start + left, search_start + right) for left, right in relative_spans]
    positions = sorted(
        {
            position
            for left, right in absolute_spans
            for position in token_positions_for_span(offsets, left, right)
        }
    )
    return {
        "input_ids": input_ids,
        "offsets": offsets,
        "candidate_text": candidate,
        "candidate_text_source": candidate_source,
        "prompt_extracted_candidate": extracted.text if extracted else "",
        "prompt_extracted_pattern": extracted.pattern_name if extracted else None,
        "structured_candidate_allowed": allow_structured_candidate,
        "candidate_positions": positions,
        "candidate_occurrence_count": len(absolute_spans),
        "candidate_char_spans_in_chat_prompt": absolute_spans,
        "search_scope": resolved_scope,
        "case_sensitive": case_sensitive,
        "used_gold_span_fields": False,
        "span_mode": "extracted",
        "candidate_extractor": extractor_spec()["name"],
        "locator": "shared_prompt_text_candidate_locator_v1",
    }
