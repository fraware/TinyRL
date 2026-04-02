# TinyRL Web UI

Next.js 14 (App Router) front end for managing projects, pipelines, artifacts, and fleet views.

## Requirements

- **Node.js 20+** (see `engines` in [`package.json`](package.json))

## Setup

```bash
npm ci
```

## Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Development server |
| `npm run build` | Production build |
| `npm run start` | Serve production build |
| `npm run lint` | ESLint (Next.js config) |
| `npm run type-check` | `tsc --noEmit` |
| `npm test` | Jest |
| `npm run e2e` | Playwright |
| `npm run chromatic` | Visual tests (needs `CHROMATIC_PROJECT_TOKEN`) |

## CI

Changes under `ui/` run [`.github/workflows/ci-ui.yml`](../.github/workflows/ci-ui.yml) (lint, types, tests, build; Chromium E2E on `main`).

## Monorepo

The Python training stack lives at the repository root; this directory is only the UI. See the root [README.md](../README.md).
