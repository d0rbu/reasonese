# File Reference

## Top level and configuration

| Path | Purpose |
|---|---|
| `README.md` | Project question, scientific status, quickstart, and claim boundary. |
| `AGENTS.md` | Agent entry point and project conventions. |
| `CLAUDE.md` | Pointer to shared agent conventions. |
| `configs/pilot.toml` | Frozen initial 14-condition, 728-trial design. |
| `configs/synthetic_demo.toml` | Known-strength offline simulator configuration. |
| `pyproject.toml` | Package metadata, CLI, dependencies, and quality tools. |
| `uv.lock` | Locked dependency graph. |
| `.pre-commit-config.yaml` | Lock, lint, type, and test hooks. |
| `.github/workflows/ci.yml` | Hosted quality gate on pushes and pull requests. |
| `LICENSE` | MIT license. |

## Package

| Path | Purpose |
|---|---|
| `reasonese/types.py` | Probability phantom type and parser. |
| `reasonese/schemas.py` | Condition, trial, response, and outcome contracts. |
| `reasonese/config.py` | Strict TOML loaders and config dataclasses. |
| `reasonese/design.py` | Counterbalanced design generation. |
| `reasonese/simulation.py` | Synthetic response backend. |
| `reasonese/scoring.py` | Completeness checks and exact scorer. |
| `reasonese/bradley_terry.py` | Penalized pairwise ranking estimator. |
| `reasonese/io.py` | Atomic JSON/JSONL persistence. |
| `reasonese/cli.py` | `design`, `simulate`, `score`, and `fit` commands. |
| `reasonese/__main__.py` | `python -m reasonese` entry point. |

## Tests

| Path | Purpose |
|---|---|
| `tests/conftest.py` | Small valid record and config factories. |
| `tests/test_schemas.py` | Schema and domain-type acceptance/rejection tests. |
| `tests/test_config_design.py` | Config strictness and full-pilot balance tests. |
| `tests/test_io_scoring_simulation.py` | Persistence, scoring, and simulator tests. |
| `tests/test_bradley_terry.py` | Estimator recovery and failure-mode tests. |
| `tests/test_cli.py` | Full offline CLI integration test. |

## Documentation

| Path | Purpose |
|---|---|
| `docs/research/` | Agenda, frozen protocol, construct validity, and related work. |
| `docs/onboarding/` | Setup, workflows, and glossary. |
| `docs/development/` | Correctness and testing contracts. |
| `docs/pipelines/` | Experiment lifecycle and evidence gates. |
| `docs/reference/` | Architecture, configuration, schemas, and this file map. |
