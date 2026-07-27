from __future__ import annotations

from pathlib import Path

from scripts.validate_data import validate_stage


def test_tiny_dataset_dev_and_holdout_validate() -> None:
    root = Path(__file__).resolve().parents[1]
    datasets_config = root / "configs/datasets.json"

    dev = validate_stage(root, datasets_config, "tiny", "dev")
    holdout = validate_stage(root, datasets_config, "tiny", "holdout")

    assert dev["status"] == "pass"
    assert holdout["status"] == "pass"
    assert holdout["condition_counts"]["mixed_eval"]["cue_only"] == 2
