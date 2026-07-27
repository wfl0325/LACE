from __future__ import annotations

import re


INVALID_SELF_ANSWER_NORMALIZED = {
    "",
    "thinking",
    "thinking process",
    "reasoning",
    "analysis",
    "final",
    "final answer",
    "answer",
}


def normalize_answer_text(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    text = text.strip(" \t\r\n\"'`")
    text = re.sub(r"^(?:the answer is|final answer:|answer:)\s+", "", text, flags=re.IGNORECASE)
    return text.strip(" \t\r\n\"'`.,;:!?").casefold()


def clean_generated_answer(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    text = re.sub(
        r"(?is)^\s*(?:thinking process|reasoning|analysis)\s*[:：]?\s*",
        "",
        text,
    ).strip()
    text = re.split(r"(?is)\b(?:final answer|answer)\s*[:：]", text, maxsplit=1)[-1].strip()
    if not text:
        return ""
    first_line = text.splitlines()[0].strip()
    first_line = re.sub(
        r"^(?:the answer is|final answer:|answer:)\s+",
        "",
        first_line,
        flags=re.IGNORECASE,
    )
    first_line = re.sub(r"\s+", " ", first_line.strip(" \t\r\n\"'`"))
    return first_line.strip(" \t\r\n\"'`.,;:!?")


def is_valid_self_answer(text: str) -> bool:
    normalized = normalize_answer_text(text)
    if normalized in INVALID_SELF_ANSWER_NORMALIZED:
        return False
    if len(normalized.split()) > 12:
        return False
    return bool(normalized)


def strict_gate_decision(
    *,
    peak_za: float | None,
    self_answer_conflict_score: float | None,
    tau_a: float | None,
    tau_c: float,
    has_intervention: bool,
) -> bool:
    """Apply the frozen LACE gate with a strict self-answer threshold."""

    return bool(
        tau_a is not None
        and peak_za is not None
        and self_answer_conflict_score is not None
        and peak_za >= tau_a
        and self_answer_conflict_score > tau_c
        and has_intervention
    )
