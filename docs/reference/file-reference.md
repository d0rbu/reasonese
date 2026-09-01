# File reference

| Path | Purpose |
|---|---|
| `configs/example_instructions.toml` | Two simple base prompts |
| `configs/example_matchup.yaml` | Two-input conversation example |
| `reasonese/axes.py` | Four axis types |
| `reasonese/config.py` | TOML instruction and YAML matchup loaders |
| `reasonese/planning.py` | Four-field dataclass and Cartesian planner |
| `reasonese/io.py` | JSONL writer |
| `reasonese/matchup.py` | Strongly typed matchup invariants and serialization |
| `reasonese/conversation.py` | Authoring requests and channel rendering |
| `docs/reference/prompt-generation.md` | Exact prompts, framing guidance, and current limitations |
| `reasonese/openrouter.py` | Synchronous and batch provider client |
| `reasonese/cache.py` | Readable message and trace caches |
| `reasonese/runner.py` | Cache-aware conversation execution |
| `reasonese/show_axes.py` | `reasonese-axes` utility |
| `reasonese/plan.py` | `reasonese-plan` utility |
| `reasonese/run_conversation.py` | `reasonese-run-conversation` utility |
| `tests/test_axes.py` | Enum and instruction-type tests |
| `tests/test_planning.py` | Combination and runtime-type tests |
| `tests/test_config_io.py` | TOML and output tests |
| `tests/test_utilities.py` | Utility integration tests |
| `tests/test_matchup_conversation.py` | Matchup and rendering tests |
| `tests/test_openrouter.py` | Provider-client contract tests |
| `tests/test_cache_runner_cli.py` | Cache, execution, and CLI tests |

Generated files belong under ignored directories such as `out/`.
