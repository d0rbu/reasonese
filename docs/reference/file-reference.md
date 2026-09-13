# File reference

| Path | Purpose |
|---|---|
| `configs/example_matchup.yaml` | Two-input conversation example |
| `configs/example_study.yaml` | Permutation-balanced collection example |
| `configs/instruction_pairs.yaml` | Candidate instruction-pair bank |
| `reasonese/axes.py` | Four axis types and default author/assistant selections |
| `reasonese/config.py` | Instruction, matchup, and study loaders |
| `reasonese/instructions.py` | Instruction pairs, bank loader, overlap and coverage diagnostics, manual-variant scaffolding |
| `reasonese/pair_checks.py` | Independent GPT-5.6 Luna audits of pair criteria |
| `reasonese/pair_check_cache.py` | Exact-text-keyed readable pair-audit cache |
| `reasonese/curate_instructions.py` | `reasonese-curate-instructions` utility and spot-check report |
| `docs/research/instruction-bank.md` | Bank criteria, conflict taxonomy, and curation flow |
| `reasonese/planning.py` | Four-field dataclass, Cartesian planner, and per-pair side grouping |
| `reasonese/sampling.py` | Connected, seeded within-pair condition subsampling |
| `reasonese/io.py` | JSONL writer |
| `reasonese/matchup.py` | Strongly typed matchup invariants and serialization |
| `reasonese/conversation.py` | Authoring requests and channel rendering |
| `reasonese/manual_messages.py` | Filesystem-backed user-authored variants |
| `prompts/user/` | Editable manual variants organized by base instruction |
| `reasonese/write_variants.py` | Blinded local editor for hand-written user variants |
| `reasonese/tools.py` | Bounded assistant tool definitions and local execution |
| `docs/reference/prompt-generation.md` | Exact prompts, framing guidance, and current limitations |
| `docs/research/qa-calibration.md` | Message-QA rubric calibration evidence and limitations |
| `reasonese/routing.py` | Collection route preference, paid opt-in, and invocation summaries |
| `tests/test_routes.py` | Routing, provenance, billing, and fingerprint compatibility contracts |
| `reasonese/openrouter.py` | Provider transport, scheduled completions, and overlapping grouped batches |
| `reasonese/scheduling.py` | Per-model adaptive concurrency, pacing, cooldowns, and bounded 429 retries |
| `tests/test_scheduling.py` | Limiter policy, runtime boundaries, recovery, pacing, and retry tests |
| `tests/test_scheduler_adversarial.py` | Deterministic concurrency and seeded multi-model retry stress tests |
| `tests/test_scheduler_review_regressions.py` | Admission, callback, retry-budget, and HTTP-date regressions |
| `tests/test_http_scheduler_integration.py` | Real HTTP transport and concurrent connections against localhost |
| `tests/test_collection_failure_recovery.py` | Failure persistence, batch isolation, and cache-only resume tests |
| `reasonese/cache.py` | Readable message and trace caches |
| `reasonese/message_qa.py` | Independent materialized-message compliance judgments |
| `reasonese/local_probe_qa.py` | Qualified local activation-probe scorer, preflight, and readable exact-context cache |
| `reasonese/role_probes.py` | Grouped activation-role probe training, qualification, scoring, and portable artifacts |
| `reasonese/role_probe_extraction.py` | Exact native role rendering, prefix-only model loading, activation capture, and dataset artifacts |
| `reasonese/extract_role_activations.py` | Local BF16 role-activation extraction utility |
| `reasonese/probe_rendering.py` | Exact native-template rendering and measured-span recovery for probe scoring |
| `reasonese/native_probe_activations.py` | Untouched native-dialogue extraction and checksummed activation artifacts |
| `reasonese/manage_role_probe.py` | Frozen two-stage probe training and qualification utility |
| `tests/test_role_probe_extraction.py` | Native-template controls, prefix equivalence, batching, and artifact contracts |
| `docs/reference/role-probes.md` | Role-probe method, provenance, validity gates, and scientific limits |
| `docs/research/nemotron-probe-adoption.md` | Saved standardized Nemotron adoption, amended segment qualification, and held-out screens |
| `reasonese/message_qa_cache.py` | Exact-text-keyed readable QA cache |
| `reasonese/check_messages.py` | Fail-closed QA gate and `reasonese-check-messages` utility |
| `reasonese/runner.py` | Cache-aware conversation execution |
| `reasonese/judging.py` | Independent GPT-5.6 Luna completion verdicts |
| `reasonese/judgment_cache.py` | Trace-keyed readable judgment cache |
| `reasonese/study.py` | Cell, study, and permutation/rollout trial types |
| `reasonese/observations.py` | Flat analysis-ready observation records |
| `reasonese/authoring_report.py` | Unique-input QA failures, excluded comparison graph, and axis denominators |
| `scripts/download_probe_prefix.py` | Pinned, range-only prefix-checkpoint downloader for local role-probe research |
| `tests/test_download_probe_prefix.py` | Offline selection, byte-range, resume, and artifact-integrity contracts |
| `tests/test_authoring_exclusions.py` | Exclusion accounting, whole-comparison filtering, cache replay, CLI, and retained-output parity |
| `reasonese/collect_data.py` | Resumable study collector and utility |
| `reasonese/collect_studies.py` | Cross-study batched collection utility |
| `reasonese/sample_studies.py` | Sampled study-suite planning utility |
| `reasonese/analysis.py` | Bradley-Terry, axis, position, and diagnostic analyses |
| `reasonese/lasso.py` | Sparse feature effects over within-trial comparisons: penalty path, cross-validation, screening |
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
| `tests/fixtures/message_qa_styles.yaml` | Labelled style/task-preservation contrasts for future authorized QA calibration |
| `tests/test_framing_styles.py` | Distinct framing briefs, exact QA fixture delivery, and model-only constraints |
| `tests/test_message_qa.py` | Message-QA prompt, parsing, cache, and utility tests |
| `tests/test_study_orchestration.py` | Permutation balance, collection, and resume tests |
| `tests/test_sampling.py` | Sparse-design stratification, degree, connectivity, and reproducibility tests |
| `tests/test_analysis.py` | Synthetic ranking, position-effect, diagnostics, and output tests |
| `tests/test_lasso.py` | Feature columns, optimality conditions, penalty path, cross-validation, and planted-effect recovery tests |
| `tests/test_instructions.py` | Pair loading, audit parsing, cache, scaffolding, and curation-utility tests |

Generated files belong under ignored directories such as `out/`.
The prefix downloader writes an atomic `.receipt` sidecar beside every completed or partial
shard and auxiliary file. A resume verifies each completed source range, keeps those verified
ranges, and truncates any unreceipted suffix before refetching it; that suffix can be large after
a process interruption. A completed legacy checkpoint may migrate from its persisted full-file
manifest checksum, while a legacy partial file without a receipt is rejected rather than trusted
by size alone.
