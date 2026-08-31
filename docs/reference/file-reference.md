# File reference

| Path | Purpose |
|---|---|
| `configs/example_instructions.toml` | Two simple base prompts |
| `reasonese/axes.py` | Four axis types |
| `reasonese/config.py` | TOML instruction loader |
| `reasonese/planning.py` | Four-field dataclass and Cartesian planner |
| `reasonese/io.py` | JSONL writer |
| `reasonese/cli.py` | `axes` and `plan` commands |
| `tests/test_axes.py` | Enum and instruction-type tests |
| `tests/test_planning.py` | Combination and runtime-type tests |
| `tests/test_config_io.py` | TOML and output tests |
| `tests/test_cli.py` | Command-line tests |

Generated files belong under ignored directories such as `out/`.
