# Testing

```bash
# Fast suite (excludes @pytest.mark.slow — matches default CI)
pytest tests/ -v -m "not slow"

# Everything including slow training smoke
pytest tests/ -v
```

## Markers

Defined in `pyproject.toml`:

- `unit` — fast, isolated tests
- `integration` — cross-module or heavier workflows
- `slow` — long runs (for example mini training)

## Coverage

```bash
pytest tests/ --cov=tinyrl --cov-report=term-missing
```

## Related automation

Python CI (`.github/workflows/ci-python.yml`) runs pytest on Ubuntu for Python 3.10, 3.11, and 3.12, plus a dedicated job for the slow mini-training test. UI tests run in `.github/workflows/ci-ui.yml`.
