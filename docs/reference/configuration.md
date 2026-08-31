# Configuration

## Experiment TOML

`configs/pilot.toml` has a top-level schema version, one `[experiment]` table, and repeated
`[[conditions]]` tables. Every key is required and unknown keys fail.

Experiment fields:

| Field | Contract |
|---|---|
| `name` | Lowercase identifier included in each trial. |
| `seed` | Non-negative integer controlling final order and design identity. |
| `repetitions` | Positive integer; one complete counterbalance per repetition. |
| `system_prompt` | Neutral response-format instruction. |
| `user_preamble` | Text before the two rendered directives. |
| `response_code_pairs` | One or more pairs of distinct uppercase alphanumeric nonce codes. |

Condition fields are `id`, `family`, `template`, and `description`. A template must contain
`{target}` exactly once.

Any input that changes rendered identity changes the `design_id`. Copy configs rather than
editing an already-collected design.

## Synthetic simulation TOML

`configs/synthetic_demo.toml` declares a clearly synthetic model identifier, seed, invalid
rate, first-position logit, and one latent strength per condition. The simulator refuses a
design if any condition lacks a strength.

## Tool configuration

All Python tooling lives in `pyproject.toml`:

- Python 3.13 and `uv` packaging;
- Ruff linting;
- ty static checking;
- pytest, Hypothesis, and branch coverage;
- a `reasonese` console entry point; and
- Hatchling as the build backend.

Pre-commit uses the locked environment and runs:

```bash
uv lock --check
uv run ruff check .
uv run ty check
uv run pytest
```
