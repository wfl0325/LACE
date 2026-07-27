from pathlib import Path

from scripts.lib.holdout_utils import config_guard, select_holdout_examples


def formal_base_config() -> dict:
    return {
        "run_mode": "formal",
        "is_formal_calibration": True,
        "frozen_config": True,
        "selector_mode": "factorized",
        "final_selection_source": "exact_scored_shortlist",
        "holdout_used_for_selection": False,
        "uses_gold_span_fields": False,
        "span_mode": "extracted",
        "uses_correct_answer_for_gate": False,
        "dataset": "popqa",
        "model_tag": "model",
        "selected_config": {
            "tau_A_value": 0.0,
            "tau_A_value_status": "frozen_from_dev_calibration_formal",
            "span_mode": "extracted",
            "uses_gold_span_fields": False,
            "top_k_policy": "band_global",
        },
    }


def test_formal_base_config_passes_guard() -> None:
    assert config_guard(
        formal_base_config(),
        Path("lace_base_config.json"),
        "popqa",
        "model",
    ) == []


def test_holdout_group_labels_only_apply_to_false_suggestions() -> None:
    behavior_rows = [
        {
            "source_id": "s1",
            "template_id": "t1",
            "condition": condition,
            "margin": 1.0,
            "logp_correct": -1.0,
            "logp_wrong": -2.0,
            "is_correct": True,
            "strict_correct": True,
        }
        for condition in ["false_suggestion", "true_suggestion"]
    ]
    q_rows = [{"source_id": "s1", "template_id": "t1", "q_group": "Q1"}]
    mixed_views = [
        {
            "source_id": "s1",
            "template_id": "t1",
            "condition": condition,
            "mc_prompt": "prompt",
        }
        for condition in ["false_suggestion", "true_suggestion", "clean_eval"]
    ]
    examples = select_holdout_examples(
        behavior_rows=behavior_rows,
        q_rows=q_rows,
        mixed_views=mixed_views,
        max_sources=0,
        max_pairs_per_group=0,
        max_rows_per_condition=0,
    )
    assert [row["q_group"] for row in examples] == ["Q1", None]
