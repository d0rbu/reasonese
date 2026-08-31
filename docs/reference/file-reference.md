# File reference

| Path | Purpose |
|---|---|
| `configs/example_instructions.toml` | Two simple example base instructions |
| `reasonese/axes.py` | Four-axis definitions and instruction validation |
| `reasonese/config.py` | Strict TOML loader |
| `reasonese/planning.py` | Complete deterministic condition enumeration |
| `reasonese/io.py` | Atomic JSONL writer and strict reader |
| `reasonese/cli.py` | `axes` and `plan` commands |
| `tests/test_axes.py` | Axis and base-instruction tests |
| `tests/test_planning.py` | Completeness, ordering, identity, and schema tests |
| `tests/test_config_io.py` | Boundary-validation and JSONL tests |
| `tests/test_cli.py` | Command-line integration tests |
| `docs/research/axes.md` | Authoritative research definitions |
| `docs/research/agenda.md` | Current scope and deferred decisions |

Generated artifacts belong under ignored directories such as `out/`; none are committed as
research findings.
