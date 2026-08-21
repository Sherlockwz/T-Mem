# Contributing to T-Mem

Thanks for your interest in improving T-Mem! This document describes how to
contribute code, report issues, and run the test suite.

## Table of contents

- [Code of conduct](#code-of-conduct)
- [Getting started](#getting-started)
- [Development setup](#development-setup)
- [Running tests](#running-tests)
- [Code style](#code-style)
- [Pull request checklist](#pull-request-checklist)
- [Reporting issues](#reporting-issues)

## Code of conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting started

1. **Fork** the repository and clone your fork.
2. Create a branch: `git checkout -b feature/my-change`.
3. Make your changes.
4. Push and open a pull request against `main`.

## Development setup

```bash
# Create a virtual environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate

# Install in editable mode with test extras
pip install -e ".[test]"

# Copy the env template if you need to run the pipeline locally
cp .env.example .env
```

The package is installable via `pyproject.toml`; no `sys.path` hacks are
needed after `pip install -e .`.

## Running tests

```bash
# All unit tests
pytest tests/ -v

# A single test file
pytest tests/test_types.py -v
```

Tests live under `tests/` and never make network calls — LLM/embedding/reranker
providers are exercised through pure-logic units (JSON parsing, retry wiring,
cost-ledger recording, trigger index operations, context truncation, ...).

## Code style

- Python 3.10+, type hints on public APIs.
- Keep comments in English so the project stays accessible to an international
  community.
- Line length: aim for ≤ 100 characters.
- Use `ruff` if you have it installed: `ruff check T_mem tests`.

## Pull request checklist

Before opening a PR, please make sure:

- [ ] `pytest tests/ -q` passes.
- [ ] `python -m compileall -q T_mem benchmark_eval scripts` passes.
- [ ] No secrets or internal URLs are introduced (see `.env.example` for the
      environment-variable convention).
- [ ] New functionality is covered by a unit test where practical.

## Reporting issues

Use the [issue templates](.github/ISSUE_TEMPLATE/) — pick:

- **Bug report** — describe expected vs. actual behaviour, stack trace, and the
  environment (Python version, OS, provider endpoint if relevant).
- **Feature request** — describe the use case and expected behaviour.

Please do **not** include API keys, tokens, or internal hostnames in issues.
