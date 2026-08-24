# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — public open-source release

### Changed

- **Providers rewritten to be OpenAI-compatible.** LLM / embedding / reranker
  calls now speak the standard OpenAI protocol
  (`/chat/completions`, `/embeddings`, `/rerank`) against any endpoint
  configured via `T_MEM_LLM_BASE_URL`, `T_MEM_EMBEDDING_BASE_URL` and
  `T_MEM_RERANKER_BASE_URL`. Internal gateways and hard-coded credentials were
  removed.
- **Secrets removed from the codebase.** API keys are read exclusively from
  environment variables (see `.env.example`).
- `bootstrap.py` no longer monkey-patches providers; it only syncs env-driven
  config overrides and installs the optional JSON-failure logger.
- Config defaults point at public model names (`gpt-4o-mini`, `bge-m3`,
  `bge-reranker-v2-m3`, ...) instead of internal aliases.
- All Chinese comments converted to English. Two Chinese strings remain on
  purpose: the natural-reply prompt (experimental config) and a retryable
  upstream error message.

### Added

- `pyproject.toml` — the package is now installable via `pip install -e .`.
- `scripts/download_data.sh` — one-command download of the LoCoMo / LoCoMo-Plus
  public datasets.
- `.env.example` — documented environment-variable template.
- `tests/` — unit tests for types, datetime utilities, cost ledger, LLM
  provider (JSON parsing / retry wiring / ledger hooks), trigger index, and
  retrieval context truncation.
- GitHub Actions CI (`.github/workflows/ci.yml`) — runs the test suite on
  Python 3.10 / 3.11 / 3.12.
- Community files: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, issue templates.

### Fixed

- `T_mem.io.truncate_search_results`: zero-keep (`keep_scene=0` / `keep_item=0`)
  now preserves the inter-section byte layout, so the following `## ` header
  still truncates correctly.
- Removed a temporary per-conversation skip hack in `stage2` and stale
  internal experiment references (rebuttal / Exp-B / internal result numbers).

## [0.2.0] - 2026-06-15

### Added

- Two-layer memory extraction pipeline (scenes -> topics -> memory items) with
  a retrieval graph and trigger recall.
- Benchmark evaluation harness for LoCoMo and LoCoMo-Plus.
- Documentation and project page for the paper
  *T-Mem: Memory That Anticipates, Not Archives*
  ([arXiv:2606.15405](https://arxiv.org/abs/2606.15405)).
