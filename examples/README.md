# Tiny Example Data

`examples/tiny` is a small synthetic dataset for checking paths and JSONL
schemas. It is not a benchmark and must not be used for method claims.

It mirrors the files required by a real LACE dataset:

- `known_fact_clean_views.jsonl`
- `unified_false_suggestion_views.jsonl`
- `unified_side_effect_views.jsonl`
- `mixed_eval_views.jsonl` for holdout
- `construction_summary.json`

Run the model-free validation check:

```bash
python scripts/validate_data.py --dataset tiny --stage all
```

The full LACE pipeline still requires local model weights, residual SAE weights,
and model-dependent behavior scoring.
