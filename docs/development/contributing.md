# Contributing

Primary guidelines live in the repository [CONTRIBUTING.md](https://github.com/fraware/TinyRL/blob/main/CONTRIBUTING.md) (branch strategy, security, PR checklist).

## Local setup

```bash
pip install -e ".[dev,docs]"
pre-commit install
pytest tests/ -m "not slow"
```

## Lint and format

```bash
ruff check tinyrl tests
ruff format tinyrl tests
mypy tinyrl/
```

## Documentation site

```bash
mkdocs serve
```

Build output is used in CI (`mkdocs build --site-dir docs/_build`).
