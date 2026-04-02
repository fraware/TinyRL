# CI/CD

Workflows live under [`.github/workflows/`](https://github.com/fraware/TinyRL/tree/main/.github/workflows).

## `ci-python.yml`

- **Lint**: `ruff check`, `ruff format --check` on `tinyrl/` and `tests/`
- **Tests**: `pytest` with `-m "not slow"` on Python 3.10–3.12 (Ubuntu)
- **Integration**: slow mini-training test
- **Security**: Bandit (`tinyrl/`), `pip-audit`, Trivy filesystem scan (pinned action version)
- **Advisory** (non-blocking): Pylint, Mypy, Pyright
- **Docs**: `mkdocs build`
- **Benchmarks**: short `train.py` runs, CLI `--help` smoke
- **ARM smoke**: `gcc-arm-none-eabi` + optional compile test when the toolchain is present
- **Docker + Trivy**: build [`Dockerfile`](https://github.com/fraware/TinyRL/blob/main/Dockerfile) and scan image (job may be marked continue-on-error)
- **Release prep** (on `main`): SBOM via `scripts/generate_sbom.py`, documentation artifact upload

## `ci-ui.yml`

Triggers on changes under `ui/` (and manual `workflow_dispatch`):

- Node 20 and 22: `npm ci`, `lint`, `type-check`, `test`, `build`
- On `main`: Playwright **Chromium** only (with `playwright install --deps`)

## Secrets (optional)

- **Codecov** — coverage upload
- **CHROMATIC_PROJECT_TOKEN** — visual regression (`npm run chromatic`)

## Dependabot

[`.github/dependabot.yml`](https://github.com/fraware/TinyRL/blob/main/.github/dependabot.yml) tracks pip, npm, and GitHub Actions.
