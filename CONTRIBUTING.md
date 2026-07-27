# Contributing

Please open an issue before changing the experimental protocol or output
schema. Bug fixes should include a focused test whenever possible.

Before submitting a pull request:

```bash
pip install -r requirements-dev.txt
pytest
python -m compileall scripts tests
```

Do not commit model weights, SAE weights, benchmark data, generated outputs, or
machine-specific `configs/models.local.json`.
