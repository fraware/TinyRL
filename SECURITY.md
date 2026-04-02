# Security policy

## Supported versions

Security updates are applied to the default branch (`main`). Tagged releases represent snapshots; use the latest release or `main` for fixes.

## Reporting a vulnerability

Please report security issues privately so we can coordinate a fix before public disclosure.

1. Open a **private** security advisory on GitHub (repository **Security** tab, **Report a vulnerability**), or email the maintainers if that option is unavailable.
2. Include steps to reproduce, affected versions or commits, and impact assessment if known.

We aim to acknowledge reports within a few business days and to ship patches as soon as practical after validation.

## Automated scanning

CI runs dependency auditing (`pip-audit`), Bandit on `tinyrl/`, Trivy on the repository filesystem, and optional Trivy scans on the training Docker image. These checks reduce risk but do not guarantee absence of vulnerabilities.

## Documentation

Contributor expectations: [CONTRIBUTING.md](CONTRIBUTING.md). Where automated checks run: [docs/development/cicd.md](docs/development/cicd.md).
