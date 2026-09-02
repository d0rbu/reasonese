# File reference

| Path | Purpose |
|---|---|
| `configs/example_instructions.toml` | Two simple base prompts |
| `configs/example_matchup.yaml` | Two-input conversation example |
| `configs/example_study.yaml` | Permutation-balanced collection example |
| `reasonese/axes.py` | Four axis types |
| `reasonese/config.py` | Instruction, matchup, and study loaders |
| `reasonese/planning.py` | Four-field dataclass and Cartesian planner |
| `reasonese/io.py` | JSONL writer |
| `reasonese/matchup.py` | Strongly typed matchup invariants and serialization |
| `reasonese/conversation.py` | Authoring requests and channel rendering |
| `reasonese/manual_messages.py` | Filesystem-backed user-authored variants |
| `prompts/user/` | Editable manual variants organized by base instruction |
| `reasonese/tools.py` | Bounded assistant tool definitions and local execution |
| `docs/reference/prompt-generation.md` | Exact prompts, framing guidance, and current limitations |
| `reasonese/openrouter.py` | Synchronous and overlapping grouped-batch provider client |
| `reasonese/cache.py` | Readable message and trace caches |
| `reasonese/message_qa.py` | Independent materialized-message compliance judgments |
| `reasonese/message_qa_cache.py` | Exact-text-keyed readable QA cache |
| `reasonese/check_messages.py` | Fail-closed QA gate and `reasonese-check-messages` utility |
| `reasonese/runner.py` | Cache-aware conversation execution |
| `reasonese/judging.py` | Independent GPT-5.6 Luna completion verdicts |
| `reasonese/judgment_cache.py` | Trace-keyed readable judgment cache |
| `reasonese/study.py` | Cell, study, and permutation/rollout trial types |
| `reasonese/observations.py` | Flat analysis-ready observation records |
| `reasonese/collect_data.py` | Resumable study collector and utility |
| `reasonese/analysis.py` | Bradley-Terry, axis, position, and diagnostic analyses |
| `reasonese/analyze.py` | `reasonese-analyze` reporting utility |
| `reasonese/show_axes.py` | `reasonese-axes` utility |
| `reasonese/plan.py` | `reasonese-plan` utility |
| `reasonese/run_conversation.py` | `reasonese-run-conversation` utility |
| `reasonese/judge_responses.py` | `reasonese-judge-responses` utility |
| `tests/test_axes.py` | Enum and instruction-type tests |
| `tests/test_planning.py` | Combination and runtime-type tests |
| `tests/test_config_io.py` | TOML and output tests |
| `tests/test_utilities.py` | Utility integration tests |
| `tests/test_matchup_conversation.py` | Matchup and rendering tests |
| `tests/test_openrouter.py` | Provider-client contract tests |
| `tests/test_cache_runner_cli.py` | Cache, execution, and CLI tests |
| `tests/test_judging.py` | Verdict, judgment-cache, and judge-utility tests |
| `tests/test_message_qa.py` | Message-QA prompt, parsing, cache, and utility tests |
| `tests/test_study_orchestration.py` | Permutation balance, collection, and resume tests |
| `tests/test_analysis.py` | Synthetic ranking, position-effect, diagnostics, and output tests |

Generated files belong under ignored directories such as `out/`.
