# Repository Guidelines

## Overview

This repository implements and documents ContextDAG, a content-addressed, dependency-aware context runtime for multi-agent LLM systems. The current development stage is the **protocol layer**: the data model, closure materializer, and prompt-level `ref`/`require` directives. The RL training line is planned future work.

## Project Structure & Module Organization

- `src/contextdag/` — protocol layer implementation (Python, standard library only): `node.py` (data model, content-address fingerprints), `registry.py` (immutable, cycle-free DAG store), `materialize.py` (canonical topological rendering), `protocol.py` (directive parsing), `session.py` (agent-facing API).
- `tests/` — `unittest` suite mirroring the modules under test.
- `docs/` — `PROTOCOL.md` (implementation spec) and `contextdag-docs/` (research plans, literature index, proposals).
- `README.md` — project overview and quick start.

## Development Workflow

The implementation uses only the Python standard library; there is no dependency installation step.

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v   # run all tests
```

There is no dependency installation step. Before committing, ensure the full suite passes.

## Coding Style & Naming Conventions

- Python follows PEP 8: 4-space indentation, `snake_case` functions, `PascalCase` classes, type hints on public APIs. Public names are exported from `contextdag/__init__.py` and listed in `__all__`.
- Node ids are 16-hex-character content-address prefixes (see `docs/PROTOCOL.md`).
- Markdown is the primary format; keep files under `docs/contextdag-docs/` well-structured with clear `#`, `##`, and `###` headings.
- Use descriptive filenames: all-caps for index and plan documents (`INDEX.md`, `RESEARCH-PLAN.md`), lowercase kebab-case for supporting notes (`background-primer.md`).
- Research documents are written in Chinese; filenames and code identifiers use English. Cite papers by arXiv ID and keep `INDEX.md` updated.

## Testing Guidelines

Tests use the standard-library `unittest` framework. Name test files `test_<module>.py` and mirror the package layout under `tests/`. There are no coverage thresholds yet; new protocol-layer behavior must come with tests.

## Commit & Pull Request Guidelines

Commits use short imperative summaries (e.g., "Add closure materializer with canonical rendering") and group logically related changes (docs, implementation, and tests can be separate commits). Pull requests should describe what changed, why, and reference related issues or documents.

## Security & Configuration Tips

Do not commit model weights, API keys, or other secrets; keep credentials in environment variables or untracked local files. `.gitignore` excludes editor/agent working directories and Python artifacts.
