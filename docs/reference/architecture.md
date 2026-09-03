# Architecture

The package has a deterministic planning flow:

```text
instruction strings -> Cartesian product -> four-field JSONL
```

- `reasonese.axes` defines the instruction phantom type and three `StrEnum` axes.
- `reasonese.config` reads a TOML array of instruction strings.
- `reasonese.planning` defines the four-field `PromptSpec` and enumerates combinations.
- `reasonese.io` writes those dataclasses as JSONL.
- `reasonese.show_axes` prints the axis values.
- `reasonese.plan` parses paths and writes the planned combinations.

The conversation flow is:

```text
matchup -> authored messages -> independent message QA -> conversation -> assistant trace
```

- `reasonese.matchup` validates an assistant and an ordered tuple of two or more datapoints.
- `reasonese.openrouter` provides synchronous completions and model-grouped batch authoring.
- `reasonese.conversation` builds authoring requests and channel-specific chat messages.
- `reasonese.manual_messages` resolves filesystem-backed variants for the user author.
- `reasonese.tools` defines bounded file, shell, Python, and server-side web-search tools.
- `reasonese.cache` stores generated messages and raw traces in readable YAML.
- `reasonese.message_qa` audits exact materialized text against its datapoint instructions.
- `reasonese.message_qa_cache` preserves parsed QA results and raw judge responses in YAML.
- `reasonese.check_messages` provides the reusable fail-closed gate and standalone utility.
- `reasonese.runner` coordinates cache lookup, generation, construction, and execution.
- `reasonese.run_conversation` is the standalone conversation utility.

The utilities have separate console entry points. There is no package-level dispatcher
or shared command switch.

`Instruction` is a `phantom-types` string constrained to be non-empty and trimmed.
`specs_per_instruction()` returns `phantom.interval.Natural`, the library's non-negative
integer type. `beartype` checks public functions and the `PromptSpec` constructor.

`MatchupInputs` is a `phantom-types` refined tuple: it contains exactly two `PromptSpec`
objects and at least one explicit user-message channel. Repeated channels are valid.
`Assistant` shares the four model values with model-backed authors but is
separate from the four entry axes.

The OpenRouter key exists only at the transport boundary. Cache keys are structural input
coordinates. Raw intermediate tool-call responses, local results, and the final provider
response are retained so reasoning and provider metadata are not discarded.

Message QA uses GPT-5.6 Luna batch with medium reasoning and a strict `{complies, issues}` schema.
It quotes the exact output of `authoring_instructions(spec)` and the candidate as data. A false
verdict blocks assistant inference but never triggers automatic regeneration. Exact content
changes invalidate the verdict. This is an LLM quality-control judgment, not a proof of semantic
equivalence.

The judging flow is:

```text
conversation trace -> one batch item per input -> aligned boolean verdicts
```

- `reasonese.judging` builds independent strict-JSON requests for GPT-5.6 Luna batch, parses
  exact booleans, and binds the verdict tuple to the matchup's input order.
- `reasonese.judgment_cache` keys judgments by matchup and a SHA-256 fingerprint of the exact
  delivered conversation and assistant response.
- `reasonese.judge_responses` is the standalone cache-aware judging utility.

The judge receives the target base instruction, its concrete delivered text, the full visible
conversation including local tool calls and results, and the assistant's final visible response
as separately escaped XML elements inside one evidence block.
Hidden reasoning remains in the trace and its fingerprint but is not quoted as judge evidence.
The response judge does not compare instructions or force a winner.

The collection flow is:

```text
study -> both input orderings x rollouts -> traces -> judgments -> observation rows
```

- `reasonese.study` defines a cell, a strongly typed input pair, and stable ordering/rollout
  trials. Its two distinct inputs produce exactly two orderings.
- `reasonese.collect_data` batches every active assistant agent-loop round, flattens uncached
  judge requests into one batch, and resumes at trial granularity.
- `reasonese.observations` joins traces and judgments into one flat row per cell and trial.

Each rollout has its own trace-cache path, so repeated responses for the same ordered matchup
do not collapse into one cache record. Generated instructions remain shared across the study.

The analysis flow is:

```text
observation rows -> within-trial comparisons -> ranking + margins + order diagnostics
```

- `reasonese.analysis` validates complete trials, converts each within-trial pair into a win,
  loss, or half-win tie, and solves the L2-penalized Bradley–Terry likelihood with Newton steps.
- Trial-cluster bootstrap resampling preserves dependence among pairwise comparisons derived
  from the same assistant response.
- Marginal axis rates, pairwise axis contrasts, position effects, connectivity, balance, and
  regularization sensitivity are calculated alongside the ranking.
- `reasonese.analyze` writes CSV tables, machine-readable diagnostics, and a Markdown report.

The L2 penalty produces a numerical total order even under separation. Comparison-graph
components are reported because relative levels across disconnected components are not
identified by the data. Marginal axis comparisons are explicitly descriptive rather than
causal when the cell design does not independently balance those axes.

The curation flow is:

```text
candidate pairs -> overlap and coverage diagnostics -> cached pair audit -> report
```

- `reasonese.instructions` defines `InstructionPair`, its YAML loader, lexical-overlap and
  coverage diagnostics, and placeholder scaffolding for `prompts/user`.
- `reasonese.pair_checks` builds strict-JSON audit requests for GPT-5.6 Luna batch and parses
  per-instruction feasibility, tool requirement, and difficulty plus pair exclusivity.
- `reasonese.pair_check_cache` keys audits by pair identifier and exact instruction texts.
- `reasonese.curate_instructions` is the fail-closed `reasonese-curate-instructions` utility.

The audit is quality control over candidate text. It does not replace the empirical check that
collected trials rarely complete both instructions or neither.
