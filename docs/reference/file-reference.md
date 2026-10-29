# File reference

| Path | Purpose |
|---|---|
| `configs/example_matchup.yaml` | Two-input conversation example |
| `configs/example_study.yaml` | Permutation-balanced collection example |
| `configs/instruction_pairs.yaml` | Candidate instruction-pair bank |
| `configs/scenario_pairs.yaml` | Twelve repository-native, mutually exclusive behavior-policy pairs |
| `configs/scenarios/` | Retirement notice for authored README histories; no active YAML templates |
| `configs/pi_langchain_base.json` | Clean first-instruction-only LangChain Kimi base specification |
| `configs/pi_langchain_review.json` | Archived DeepSeek conflict-pilot manifest; not for base collection |
| `configs/pi_remaining_review.json` | Eleven Kimi base pilots using only the first instruction, review checks and environments |
| `docs/research/remaining-scenario-pilots.md` | Base task catalog, reserved opposing requirements, ordinary follow-ups and preparation scope |
| `tools/pi-transcripts/environments.mjs` | Scoped dependency setup, writable task dependencies and recorded immutable runtimes |
| `tools/pi-transcripts/network-worker.mjs` | Private routed network setup and outer firewall before unprivileged tool execution |
| `tools/pi-transcripts/backing.mjs` | Neutral physical task storage and content-addressed SDK/worker/tool mirrors |
| `tools/pi-transcripts/restore-checkpoint.py` | Safe private extraction of verified task checkpoints without following archive symlinks |
| `tools/pi-transcripts/checkpoints.mjs` | Complete per-turn task/dependency snapshots and verified restoration for matching continuation state |
| `docs/reference/pi-development-environment.md` | Neutral actual paths, tools, network isolation and checkpoint boundaries |
| `tools/pi-transcripts/baseline_*` | Executable untouched-source observations, not post-change success oracles |
| `tools/pi-transcripts/viewer/` | Static transcript review pages with scenario cards, reasoning, checks and evidence downloads |
| `tools/pi-transcripts/neutralize.mjs` | Offline path-only derivatives and exhaustive string-surface cue audit |
| `docs/research/transcript-environment-audit.md` | Path edits, remaining evaluation/environment cues and source-versus-derivative limits |
| `configs/pi_langchain_kimi_review.json` | Archived Kimi conflict-pilot manifest; not for base collection |
| `docs/research/langchain-json-recovery.md` | Historical JSON parser conflict pilots, original pair and observed results |
| `docs/research/langchain-json-base.md` | First-instruction-only Kimi source session, independent checks and viewer |
| `docs/research/kimi-base-collection.md` | Twelve collected Kimi bases, per-case validation scope, hashes and shared viewer |
| `configs/pi_transcripts.json` | Twelve base tasks with the first instruction as written, pinned repositories and limits |
| `tools/pi-transcripts/` | Pinned Pi SDK collector, real repository preparation, isolated built-in tools, and offline integration tests |
| `docs/reference/pi-transcripts.md` | Preparation, authorized transcript collection, evidence format, and scientific limits |
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
| `reasonese/conversation.py` | Authoring requests, channel rendering, and the bare and scenario conversation layouts |
| `reasonese/scenarios.py` | Optional per-pair scenario templates: loading, slot rendering, placement, and cache validity |
| `reasonese/show_scenario.py` | `reasonese-show-scenario` offline rendering utility |
| `reasonese/prompt_optimization.py` | Bounded authoring-brief evaluation, fixed-rubric reports, and offline comparison table |
| `reasonese/manual_messages.py` | Filesystem-backed user-authored variants |
| `prompts/user/` | Editable manual variants organized by base instruction |
| `reasonese/write_variants.py` | Blinded local editor for hand-written user variants |
| `reasonese/tools.py` | Bounded assistant tool definitions and local execution |
| `docs/reference/prompt-generation.md` | Exact prompts, framing guidance, and current limitations |
| `docs/research/qa-calibration.md` | Message-QA rubric calibration evidence and limitations |
| `docs/research/prompt-optimization.md` | Bounded baseline/candidate authoring comparison and report contracts |
| `docs/research/pilot-qa-recovery.md` | Trial-local tool failures, high-effort QA, framing-cutoff calibration, and recovery evidence |
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
| `reasonese/local_probe_qa.py` | Qualified local activation-probe scorer, per-target prefix capture, preflight, and readable cache |
| `reasonese/probe_qa.py` | Shared inline/replay request validation, diagnostic scoring, and coverage reports |
| `reasonese/probe_posthoc.py` | `reasonese-score-probes` replay over saved trace setups with separate identity-bound output |
| `reasonese/role_probes.py` | Grouped activation-role probe training, qualification, scoring, and portable artifacts |
| `reasonese/role_probe_extraction.py` | Exact native role rendering, prefix-only model loading, activation capture, and dataset artifacts |
| `reasonese/extract_role_activations.py` | Local BF16 role-activation extraction utility |
| `reasonese/probe_rendering.py` | Exact native-template rendering and measured-span recovery for probe scoring |
| `reasonese/native_probe_activations.py` | Untouched native-dialogue extraction and checksummed activation artifacts |
| `reasonese/manage_role_probe.py` | Frozen two-stage probe training and qualification utility |
| `tests/test_role_probe_extraction.py` | Native-template controls, prefix equivalence, batching, and artifact contracts |
| `docs/reference/role-probes.md` | Role-probe method, provenance, validity gates, and scientific limits |
| `docs/research/nemotron-probe-adoption.md` | Saved standardized Nemotron adoption, amended segment qualification, and held-out screens |
| `docs/research/probe-context-transfer.md` | Frozen Nemotron probe measurements across native and instruction-delivery contexts |
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
| `tests/test_probe_posthoc.py` | Replay parity, saved-context provenance, missing coverage, identity, and read-only source contracts |
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
| `tests/test_scenarios.py` | Scenario slots, placement, file rules, trace caching, and collection wiring tests |
| `tests/test_scenario_bank.py` | First-instruction-only base coverage, pinned provenance and immutable historical manifests |
| `tests/fixtures/scenarios/` | One synthetic scenario that exercises the template slots; not study material |
| `tests/test_openrouter.py` | Provider-client contract tests |
| `tests/test_cache_runner_cli.py` | Cache, execution, and CLI tests |
| `tests/test_tool_output_text.py` | Exact local file text in tool continuations, caches, and judge evidence |
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

Native Pi user-turn experiments use `reasonese/native_templates.py` for lossless
checkpointed plans and `reasonese/native_run.py` for hard message QA plus bounded
execution. `tools/pi-transcripts/continue.mjs` validates/restores a plan and seeds
Pi's native history. Editable full-turn inputs live in
`prompts/scenario_turns/`; see
[native-pi-experiments.md](native-pi-experiments.md) for the Max handoff.
`configs/pi_native_sources.json` pins the selected downloadable source artifacts;
`tools/pi-transcripts/fetch-sources.py` verifies and unpacks full sources or smaller
context-only bundles. `configs/native_pi_example.yaml` is an offline planning
example, not a collected result or authorization for inference.

`tools/pi-authoring/` serves the native transcript workshop, human draft/editor and
GPT 6.1 Sol high submission checks. `configs/pi_authoring_rubrics.json` binds the
36 bespoke assignment checklists to their exact sources; `tests/test_pi_authoring.py`
checks HTTP access, fail-closed paid checks and exact-text persistence. See the
[workshop guide](../../tools/pi-authoring/README.md) for deployment and exports.
