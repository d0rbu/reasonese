# Testing

## Canonical gate

```bash
uv run pre-commit run --all-files
```

The hooks run `uv lock --check`, Ruff, ty, and pytest with branch coverage.

## Focused commands

```bash
uv run pytest tests/test_config_design.py
uv run pytest tests/test_io_scoring_simulation.py
uv run pytest tests/test_bradley_terry.py
uv run pytest tests/test_cli.py
uv run pytest -m property
```

## Test layers

- `test_schemas.py`: primitive, condition, trial, response, and outcome boundaries.
- `test_config_design.py`: strict TOML and full-pilot counterbalancing.
- `test_io_scoring_simulation.py`: atomic persistence, synthetic provenance, and exact scoring.
- `test_bradley_terry.py`: recovery, counts, connectivity, malformed pairs, and convergence.
- `test_cli.py`: the complete offline command sequence and user-facing failures.

Coverage fails below 95% for the `reasonese` package. Coverage is a guardrail; full-pilot
balance assertions and explicit failure cases are more scientifically important than merely
executing lines.
