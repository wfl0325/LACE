# LACE

Official implementation of **Latent Answer Competition Editing (LACE)**.
LACE is a test-time sparse-autoencoder intervention method for factual
questions that contain an explicit but incorrect suggested answer.

This repository contains only the code required to build and evaluate the
final LACE method. It does not include phenomenon studies, mechanism
exploration, baselines, or supplementary ablations.

## Method

LACE learns from two source-disjoint development subsets:

- **Discovery split:** discovers causally screened wrong-answer-support and
  factual-stability SAE features, then estimates a residual reference direction
  from suggestion-sensitive failures to robust examples.
- **Calibration split:** selects the layer or layer band, feature counts,
  feature mixture, intervention strength, and gate thresholds.

At inference time, LACE:

1. locates the user-suggested entity in the prompt without gold character
   offsets;
2. measures the standardized activation of selected wrong-answer-support
   features;
3. compares the model's short neutral self-answer with the suggested entity
   without reading the correct answer;
4. applies a causally weighted residual-SAE intervention only when both frozen
   gate conditions pass.

All method choices are frozen on development data. Holdout data is used only
for model-dependent behavior labels and final evaluation.

## Layout

```text
LACE/
  configs/                       Dataset, model, and search configuration
  data/                          Local benchmark files, ignored by Git
  examples/                      Tiny schema-check dataset
  outputs/                       Generated artifacts, ignored by Git
  scripts/
    prepare_behavior.py          Build model-dependent behavior labels
    build_dev_splits.py          Split development sources
    discover_feature_banks.py    Discover the two SAE feature banks
    calibrate_lace.py            Select the base layer/band and features
    select_lace_config.py        Freeze the final LACE configuration
    eval_lace_holdout.py         Evaluate that configuration on holdout
    internal/                    Shared scoring kernels; not CLI entry points
    lib/                         Reusable data, model, gate, and SAE utilities
    runners/run_lace_pipeline.sh
  tests/
```

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install a PyTorch build compatible with the local CUDA driver when the default
package index does not provide one.

For development or CI checks, install the developer requirements instead:

```bash
pip install -r requirements-dev.txt
```

`requirements-dev.txt` includes `requirements.txt` and adds test/lint tools.

## Models And SAEs

Create the local model registry:

```bash
cp configs/models.example.json configs/models.local.json
```

Fill in each model and SAE path. `models.local.json` is ignored by Git.

LACE captures and edits Transformer-block residual outputs. The configured SAE
must therefore be trained on the corresponding `residual_post` activation
space. Matching hidden dimensionality is not sufficient. The code fails before
inference when the registry declares an MLP SAE or another incompatible hook.

The registry also fixes the encoder nonlinearity. Qwen experiments may retain
the released `relu_legacy_lace` policy, while Gemma Scope residual SAEs use
their per-feature `jumprelu` thresholds. For releases whose average-L0 path
differs by layer, set `sae_variants_by_layer`; the legacy single
`sae_variant` field is only appropriate when one relative variant path applies
to every layer. LACE never chooses a variant silently.

## Data

Register dataset paths in [configs/datasets.json](configs/datasets.json). Each
dataset needs a development set and a source-disjoint holdout set.

Each dataset stage must provide:

- `known_fact_clean_views.jsonl`
- `unified_false_suggestion_views.jsonl`
- `unified_side_effect_views.jsonl`
- `construction_summary.json`

Holdout stages additionally provide:

- `mixed_eval_views.jsonl`

Required row fields are:

- `source_id`
- `pair_id`
- `template_id`
- `mc_prompt`
- `question_text`
- `correct_option`
- `wrong_option`
- `suggested_option`
- `relation`

False-suggestion rows must use paired `trigger_condition` values `control` and
`pressure` for each `pair_id`. Side-effect rows use `clean` and
`true_suggestion`. Holdout mixed-evaluation rows use `condition` values
`clean_eval`, `false_suggestion`, `true_suggestion`, and `cue_only`.

Dataset and model weights are not redistributed by this repository.

A tiny synthetic example is included for checking the expected file layout and
JSONL schema:

```bash
python scripts/validate_data.py --dataset tiny --stage all
```

The tiny example is not a benchmark. It does not test model quality or LACE
metrics.

## Reproduce LACE

The default search configuration reproduces the preregistered Qwen layer/band
space. Another model family must provide its own development-defined
`SEARCH_CONFIG`; it must not reuse the Qwen layer indices blindly.
The runner uses the released main-run defaults: a 128-row factorized shortlist,
at least 16 shortlisted candidates per layer band, multi-layer bands enabled,
and Q1 repair as the primary feasible-config selection criterion.

Run the complete pipeline for one dataset/model pair:

```bash
DATASET=popqa \
MODEL_TAG=qwen35_2b \
DEVICE=cuda:0 \
bash scripts/runners/run_lace_pipeline.sh
```

The runner supports these stages:

| `STAGE` | Work performed |
|---|---|
| `prepare` | Development and holdout behavior labels |
| `split` | Source-level discovery/calibration split |
| `features` | Causal SAE feature-bank discovery |
| `base` | Factorized shortlist and exact development calibration |
| `selector` | Final LACE configuration selection |
| `holdout` | Frozen-config holdout evaluation |
| `final` | `selector` followed by `holdout` |
| `all` | Complete pipeline; default |

For example, after behavior labels, development splits, feature banks, and the
base calibration already exist:

```bash
DATASET=popqa MODEL_TAG=qwen35_2b DEVICE=cuda:0 STAGE=final \
bash scripts/runners/run_lace_pipeline.sh
```

For a model-specific layer search:

```bash
DATASET=counterfact \
MODEL_TAG=my_model \
SEARCH_CONFIG=/path/to/my_model_lace_search.json \
DEVICE=cuda:0 \
bash scripts/runners/run_lace_pipeline.sh
```

The main outputs are:

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

## Protocol Guards

- Discovery and calibration read development data only.
- Q1/Q2 groups are defined only for paired clean and false-suggestion views.
- Final localization never reads dataset-provided gold span offsets.
- The gate never reads the dataset's correct answer.
- Invalid or heading-like self-answers are rejected.
- The activation score must satisfy `peak_zA >= tau_A`; the self-answer
  conflict score uses the strict condition `score > tau_C`.
- Final evaluation accepts only a formal, frozen development-selected config.
- Final method parameters cannot be overridden from the evaluation CLI.
- The SAE activation site and intervention hook must both be residual-post.

## Validation

```bash
pip install -r requirements-dev.txt
python scripts/validate_data.py --dataset tiny --stage all
pytest -q
python -m compileall -q scripts tests
bash -n scripts/runners/run_lace_pipeline.sh
```

## License

Code is released under the MIT License. Models, SAEs, and datasets retain their
original licenses.
