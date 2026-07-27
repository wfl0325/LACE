from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .io_utils import load_json, resolve_path


@dataclass(frozen=True)
class DatasetStage:
    dataset: str
    stage: str
    name: str
    known_fact_clean: Path
    false_suggestion: Path
    side_effect: Path
    construction_summary: Path
    mixed_eval: Path | None
    source_count: int
    allowed_for: tuple[str, ...]
    forbidden_for: tuple[str, ...]


def load_dataset_stage(config_path: Path, dataset: str, stage: str, root: Path) -> DatasetStage:
    config = load_json(config_path)
    datasets = config.get("datasets", {})
    if dataset not in datasets:
        raise KeyError(f"Unknown dataset={dataset!r}; available={sorted(datasets)}")
    if stage not in {"dev", "holdout"}:
        raise ValueError("--stage must be one of: dev, holdout")

    stage_key = "development" if stage == "dev" else "holdout"
    entry = datasets[dataset][stage_key]
    return DatasetStage(
        dataset=dataset,
        stage=stage,
        name=entry["name"],
        known_fact_clean=resolve_path(entry["known_fact_clean"], root),
        false_suggestion=resolve_path(entry["false_suggestion"], root),
        side_effect=resolve_path(entry["side_effect"], root),
        construction_summary=resolve_path(entry["construction_summary"], root),
        mixed_eval=resolve_path(entry["mixed_eval"], root) if entry.get("mixed_eval") else None,
        source_count=int(entry["source_count"]),
        allowed_for=tuple(entry.get("allowed_for", [])),
        forbidden_for=tuple(entry.get("forbidden_for", [])),
    )


def validate_dataset_stage(stage: DatasetStage) -> list[str]:
    missing = []
    for label, path in [
        ("known_fact_clean", stage.known_fact_clean),
        ("false_suggestion", stage.false_suggestion),
        ("side_effect", stage.side_effect),
        ("construction_summary", stage.construction_summary),
    ]:
        if not path.exists():
            missing.append(f"{label}: {path}")
    if stage.mixed_eval is not None and not stage.mixed_eval.exists():
        missing.append(f"mixed_eval: {stage.mixed_eval}")
    return missing

