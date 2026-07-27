from scripts.lib.gate import (
    clean_generated_answer,
    is_valid_self_answer,
    strict_gate_decision,
)


def test_zero_conflict_does_not_pass_zero_threshold() -> None:
    assert not strict_gate_decision(
        peak_za=2.0,
        self_answer_conflict_score=0.0,
        tau_a=1.0,
        tau_c=0.0,
        has_intervention=True,
    )


def test_positive_conflict_passes_when_all_conditions_hold() -> None:
    assert strict_gate_decision(
        peak_za=2.0,
        self_answer_conflict_score=0.01,
        tau_a=1.0,
        tau_c=0.0,
        has_intervention=True,
    )


def test_thinking_heading_without_answer_is_rejected() -> None:
    cleaned = clean_generated_answer("Thinking Process")
    assert cleaned == ""
    assert not is_valid_self_answer(cleaned)


def test_thinking_block_is_removed_before_final_answer() -> None:
    cleaned = clean_generated_answer("<think>reasoning</think>\nFinal Answer: Paris")
    assert cleaned == "Paris"
    assert is_valid_self_answer(cleaned)
