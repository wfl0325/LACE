# LACE

Reference implementation of **Latent Answer Competition Editing (LACE)**, a
test-time intervention method for factual QA prompts that contain an explicit
but incorrect suggested answer.

LACE uses sparse-autoencoder (SAE) features and a frozen gating policy to reduce
wrong-answer copying while preserving clean factual behavior.

## What This Repository Contains

- End-to-end LACE pipeline scripts for behavior labeling, source splitting,
  SAE feature discovery, calibration, configuration selection, and holdout
  evaluation.
- Dataset registry and JSONL data views for `popqa`, `counterfact`, and a tiny
  synthetic schema-check dataset.
- Model registry template for local model and SAE paths.
- Unit tests and model-free dataset validation utilities.


## Method Overview

LACE is designed for prompts where the user states or implies a candidate answer
that may be false. The method is calibrated on development data and then frozen
before holdout evaluation.

At inference time, LACE:

1. localizes the suggested entity in the prompt without using dataset-provided
   gold span offsets;
2. measures selected wrong-answer-support SAE features at residual-stream
   activation sites;
3. asks the model for a short neutral self-answer and compares it with the
   suggested entity without reading the correct answer;
4. applies a causally weighted residual-SAE intervention only when the activation
   gate and the self-answer conflict gate both pass.

Development data is used for discovery and calibration. Holdout data is used
only for model-dependent labels and final evaluation.

## Repository Layout

```text
LACE/
  configs/
    datasets.json                Dataset registry
    models.example.json          Template for local model and SAE paths
    lace_search_space.json       Default LACE search space
    side_effect_constraints.json Evaluation constraints
  data/
    popqa/                       POPQA JSONL benchmark views
    counterfact/                 CounterFact JSONL benchmark views
  examples/tiny/                 Small synthetic dataset for schema checks
  scripts/
    validate_data.py             Model-free dataset validation
    prepare_behavior.py          Build model-dependent behavior labels
    build_dev_splits.py          Create source-disjoint dev splits
    discover_feature_banks.py    Discover SAE feature banks
    calibrate_lace.py            Calibrate layers, features, gates, strengths
    select_lace_config.py        Freeze the final formal LACE config
    eval_lace_holdout.py         Evaluate the frozen config on holdout data
    runners/run_lace_pipeline.sh End-to-end pipeline runner
  tests/                         Unit tests
  outputs/                       Generated artifacts, not committed
```

## Installation

LACE requires Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For development checks, install the test and lint dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

The project depends on PyTorch, Transformers, SentencePiece, NumPy, and
Safetensors. If the default PyTorch package is not compatible with your CUDA
driver, install the correct PyTorch build for your machine before running the
full pipeline.

## Quick Start

First run the lightweight schema check. This does not load a language model or
SAE.

```bash
python scripts/validate_data.py --dataset tiny --stage all
```

Then create a local model registry:

```bash
cp configs/models.example.json configs/models.local.json
```

Edit `configs/models.local.json` so each `model_path` and `sae_path` points to
your local files. This file is ignored by Git because model paths are
machine-specific.

Run the complete pipeline for one dataset and model:

```bash
DATASET=popqa \
MODEL_TAG=qwen35_2b \
DEVICE=cuda:0 \
bash scripts/runners/run_lace_pipeline.sh
```

The same runner can be used for CounterFact:

```bash
DATASET=counterfact \
MODEL_TAG=qwen35_2b \
DEVICE=cuda:0 \
bash scripts/runners/run_lace_pipeline.sh
```

## Models And SAEs

LACE edits Transformer block residual outputs. The configured SAE must therefore
match the model's `residual_post` activation space. Matching hidden dimension is
not enough; an MLP SAE or a different hook site is incompatible.

The model registry records:

- model path;
- SAE release path;
- model family;
- dtype and default device;
- SAE activation site;
- intervention hook;
- SAE activation function policy;
- per-layer SAE variants when a release uses layer-specific paths.

The example registry includes entries for Qwen and Gemma-style setups. For a new
model family, define a development-selected search space instead of reusing Qwen
layer indices blindly.

## Data

Dataset paths are registered in [configs/datasets.json](configs/datasets.json).
Each real dataset has a development split and a source-disjoint holdout split.

Each development split is expected to provide:

- `known_fact_clean_views.jsonl`
- `unified_false_suggestion_views.jsonl`
- `unified_side_effect_views.jsonl`
- `construction_summary.json`

Each holdout split additionally provides:

- `mixed_eval_views.jsonl`

Required JSONL row fields are:

- `source_id`
- `pair_id`
- `template_id`
- `mc_prompt`
- `question_text`
- `correct_option`
- `wrong_option`
- `suggested_option`
- `relation`

False-suggestion rows use paired `trigger_condition` values `control` and
`pressure` for each `pair_id`. Side-effect rows use `clean` and
`true_suggestion`. Holdout mixed-evaluation rows use `condition` values
`clean_eval`, `false_suggestion`, `true_suggestion`, and `cue_only`.

Validate a registered dataset after all paths in the registry exist:

```bash
python scripts/validate_data.py --dataset popqa --stage all
python scripts/validate_data.py --dataset counterfact --stage all
```

## Pipeline Stages

The runner supports running the whole pipeline or a single stage through the
`STAGE` environment variable.

| `STAGE` | Description |
|---|---|
| `prepare` | Build development and holdout behavior labels |
| `split` | Create source-level discovery/calibration splits |
| `features` | Discover causal SAE feature banks |
| `base` | Run shortlist and exact development calibration |
| `selector` | Freeze the formal LACE configuration |
| `holdout` | Evaluate the frozen configuration on holdout data |
| `final` | Run `selector` and `holdout` |
| `all` | Run the complete pipeline |

Example: rerun only final selection and holdout evaluation after earlier stages
already exist.

```bash
DATASET=popqa \
MODEL_TAG=qwen35_2b \
DEVICE=cuda:0 \
STAGE=final \
bash scripts/runners/run_lace_pipeline.sh
```

For a model-specific layer search, pass a custom search configuration:

```bash
DATASET=counterfact \
MODEL_TAG=my_model \
SEARCH_CONFIG=/path/to/my_model_lace_search.json \
DEVICE=cuda:0 \
bash scripts/runners/run_lace_pipeline.sh
```

## Outputs

Generated artifacts are written under:

```text
outputs/{dataset}/{model}/
  dev_behavior/
  holdout_behavior/
  dev_splits/
  feature_banks/
  calibration/
    lace_base_config.json
  lace_config/
    lace_config.json
  holdout_eval/
    records.jsonl
    summary.json
    audit.json
```
