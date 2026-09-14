# Architecture

The package has a deterministic planning flow:

```text
instruction pairs -> per-side author-restricted Cartesian product -> four-field JSONL
```

- `reasonese.axes` defines the instruction phantom type, three `StrEnum` axes, and
  `author_framings`, which is the single source for the framings an author writes.
- `reasonese.config` reads matchup and study YAML.
- `reasonese.planning` defines the four-field `PromptSpec`, rejects a datapoint whose author does
  not write its framing, enumerates the combinations each author writes, and groups them into
  `PairSpecs` holding both sides of one instruction pair.
- `reasonese.io` writes those dataclasses as JSONL.
- `reasonese.show_axes` prints the axis values.
- `reasonese.plan` parses paths and writes the planned combinations.
- `reasonese.sampling` selects a seeded, axis-stratified, degree-aware subset of valid cell pairs
  without materializing the exhaustive edge population.
- `reasonese.sample_studies` writes the selected pairs, replicated across assistants, as one
  study-suite YAML artifact.

The conversation flow is:

```text
matchup -> authored messages -> independent message QA -> conversation -> assistant trace
```

- `reasonese.matchup` validates an assistant and an ordered tuple of two or more datapoints.
- `reasonese.openrouter` uses the shared model scheduler for synchronous completions and batch
  submissions, and overlaps
  independent model-grouped batch jobs while preserving request and group order. Requests with
  OpenRouter server tools stay on the synchronous API because those tools are rejected by the
  Batch API.
- `reasonese.scheduling` owns a separate adaptive concurrency window, start interval, and
  cooldown per requested model slug. HTTP attempts and response callbacks run in bounded thread
  workers; retries wait in model queues. One tracked future represents each active request,
  including local tool processing; model capacity is derived from these futures. HTTP error
  feedback is applied before crediting successes or admitting more
  work. Immutable requests receive a fresh retry budget each time they are queued, and limits
  persist across stages on one client.
- `reasonese.conversation` builds authoring requests and channel-specific chat messages.
- `reasonese.manual_messages` resolves filesystem-backed variants for the user author and
  snapshots the needed files once per top-level invocation.
- `reasonese.tools` defines bounded file, shell, Python, and server-side web-search tools.
- `reasonese.cache` stores generated messages and raw traces in readable YAML.
- `reasonese.message_qa` audits exact materialized text against its datapoint instructions.
- `reasonese.message_qa_cache` preserves parsed QA results and raw judge responses in YAML.
- `reasonese.check_messages` provides the reusable fail-closed gate and standalone utility.
- `reasonese.runner` coordinates cache lookup, generation, construction, and completion-driven
  assistant execution. A tool continuation is submitted as soon as its preceding response
  arrives, independently of slower peers. The shared scheduler prioritizes ready continuations
  over fresh requests within each model, under that model's adaptive limit. A lock protects the
  shared tool-runtime pool, while each conversation advances its own history sequentially.
- `reasonese.run_conversation` is the standalone conversation utility.

The utilities have separate console entry points. There is no package-level dispatcher
or shared command switch.

`Instruction` is a `phantom-types` string constrained to be non-empty and trimmed.
`specs_per_instruction()` returns `phantom.interval.Natural`, the library's non-negative
integer type. `beartype` checks public functions and the `PromptSpec` constructor.

`MatchupInputs` is a `phantom-types` refined tuple: it contains exactly two `PromptSpec`
objects and at least one explicit user-message channel. Repeated channels are valid.
`Assistant` shares the six model values with model-backed authors but is
separate from the four entry axes.

The OpenRouter key exists only at the transport boundary. Cache keys are structural input
coordinates. Raw intermediate tool-call responses, local results, and the final provider
response are retained so reasoning and provider metadata are not discarded.

Message QA uses GPT-5.6 Luna batch with medium reasoning and a strict `{complies, issues}` schema.
It quotes the exact output of `authoring_instructions(spec)` and the candidate as data. A false
verdict prevents assistant inference for that input but never triggers automatic regeneration.
Study collectors exclude each entire comparison containing a rejected input and continue with
the remaining comparisons; the standalone conversation utility still fails closed. Exact content
changes invalidate the verdict. This is an LLM quality-control judgment, not a proof of semantic
equivalence.

The judging flow is:

```text
conversation trace -> one batch item per input -> aligned boolean verdicts
```

- `reasonese.judging` builds independent strict-JSON requests for GPT-5.6 Luna batch, parses
  exact booleans, and binds the verdict tuple to the matchup's input order. Its
  `FingerprintedTrace` derives the trace fingerprint once for reuse across collection stages.
- `reasonese.judgment_cache` keys judgments by matchup and a SHA-256 fingerprint of the exact
  delivered conversation and assistant response.
- `reasonese.judge_responses` is the standalone cache-aware judging utility.

The judge receives the target base instruction, its concrete delivered text, the full visible
conversation including local tool calls and results, and the assistant's final visible response
as separately escaped XML elements inside one evidence block.
Intermediate assistant messages retain provider `annotations`; final-message annotations appear
as escaped JSON in a separate `<assistant-annotations>` element when the field is present.
This exposes OpenRouter's server-side web-search citations even without local tool calls or URLs
in the response text. Missing or empty annotations do not prove that no search occurred.
Hidden reasoning remains in the trace and its fingerprint but is not quoted as judge evidence.
The response judge does not compare instructions or force a winner.

The shared completion path validates top-level and choice-level provider errors and error
finish reasons before delivering text or executing local tools, even for HTTP 200 responses.
Missing, null, or blank final text is also a provider failure. Actual tool-call responses may
legitimately have null content. This validation covers authoring and assistant execution;
strict text consumers also reject embedded errors in cached responses.

Synchronous completion failures log their exceptions and retry the identical request on the
same route. Empty/error responses, HTTP 408/5xx, connection failures, and timeouts allow at
most two retries per author request or assistant conversation (including tool continuations),
with 1-second then 2-second delays or a longer supplied Retry-After. Provider-failure and HTTP
429 budgets are independent: each provider retry or tool continuation gets a fresh 429 budget,
while the provider-failure counter is retained across the conversation.
Embedded 429s use the same adaptive limiter and cooldown as HTTP 429s. Explicit permanent
provider errors such as 400/401/403 are not retried. Failed responses never reach local tools
or response judgments. Already executed local tools are preserved, while server-side searches
may execute again on a retry. Consumer/callback errors do not retry a successful HTTP request.

Completed batch items are individually validated, with failures logged and raised after
successful items and other submitted batches have been processed. Authoring retains successful
responses through its existing partial-result cache path. Batch item failures and ambiguous
batch-submission failures are not automatically resubmitted; this avoids duplicating paid jobs.
The existing 429 retry policy still applies to rejected batch submissions.

The collection flow is:

```text
study -> both input orderings x rollouts -> traces -> judgments -> observation rows
```

- `reasonese.study` defines a cell, a strongly typed input pair, and stable ordering/rollout
  trials. Its two distinct inputs produce exactly two validated matchups, reused by every rollout.
- `reasonese.authoring_report` describes unique QA input failures and the planned comparison
  graph, including excluded edges, original cell/trial IDs, and marginal axis counts. It does
  not modify observation schemas or fit an analysis model.
- `reasonese.collect_data` concurrently advances every assistant tool loop as responses arrive,
  flattens uncached judge requests into one batch, and resumes at trial granularity. Completion
  callbacks retain successful traces; a final bulk transaction saves them even if a peer fails.
  Failed collection stages do not advance into judging or observation generation.
- `reasonese.collect_studies` applies the same stages across repeated study paths, sharing
  materialized-message and QA caches and grouping concurrent trials and batched judgments across
  study boundaries. It also consumes sampled suite YAML, uses stable fingerprint output names,
  writes a combined observation file, and stores all suite traces and judgments in one root
  SQLite database.
- `reasonese.study_cache` loads study traces and judgments with one SQLite query per table,
  validates serialized coordinates against the known trial matchups without reparsing them, and
  writes each completed stage in one transaction keyed by stable trial ID.
- `reasonese.observations` batch-joins traces and judgments into flat rows, reusing cell IDs
  across every rollout of the same cell.

Each rollout has its own trial-keyed trace and judgment row, so repeated identical responses do
not collapse into one cache record. Generated instructions remain shared across the study.

The sparse planner streams over the eligible population to count strata defined by channel pair
and the set of axes that differ. Largest-remainder allocation turns those counts into exact
proportional sample quotas. A second streaming pass uses reservoir sampling to build a seeded
candidate pool of at most three times each quota, and power-of-eight choices prefer endpoints
with lower degree relative to their channel's target. Planning memory therefore scales with the
requested sample rather than the exhaustive edge population.

The planner then checks the sampled comparison graph. If it has `k` components, it swaps exactly
`k - 1` redundant cycle edges for valid cross-component bridges, preferring removals in the same
stratum. This is the minimum number of replacements that can connect `k` components while
preserving the requested edge count. The selected edge set is shared across assistants;
assistant models remain separate comparison components.

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
- `reasonese.lasso` refits the same comparisons as a sparse feature model along an L1 penalty
  path and cross-validates it.
- `reasonese.analyze` writes CSV tables, machine-readable diagnostics, and a Markdown report
  for both fits.

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
collected trials rarely complete both instructions or neither. That check is
`pair_exclusivity.csv`, written by `reasonese-analyze`.

Because instructions are bipartite, sampling and analysis are both scoped to one pair:

```text
pair -> bipartite edge population -> stratified degree-aware sample -> connected design
observations -> (pair, side) blocks -> per-component Bradley-Terry -> pooled axis margins
```

- `reasonese.instructions.instruction_index` maps each instruction to its pair and side and
  rejects reuse of one instruction across pairs.
- `reasonese.sampling` enumerates each pair's valid edges by rank, samples within stratum
  quotas, balances degree within channel, and repairs connectivity per pair.
- `reasonese.analysis` fits one Bradley-Terry block per connected component, which is one
  `(pair, assistant)`, so scores self-centre and pooled axis margins stay comparable.

Each Newton step accumulates every comparison's gradient and curvature with `numpy.bincount`
rather than a Python loop, and the standard errors and objective are computed only when a caller
reads them, which the bootstrap does not. Every block is a few hundred cells, so its BLAS calls
spend far longer synchronizing threads than doing arithmetic; `threadpoolctl` pins threads to one
for the duration of a fit and restores the caller's settings on exit. Together these take a pilot
analysis from hours to well under a minute.

The feature lasso in `reasonese.lasso` refits the same within-trial comparisons with every
cell's strength written as a `(pair, assistant)` side offset plus a sparse sum of feature
effects: treatment contrasts for framing, channel, and author; `self_author` and `same_family`
match indicators in place of a saturated author-by-assistant interaction; which cell was
delivered first; and the two-way interactions among framing, channel, author, and assistant
other than author by assistant. The references are the `normal` framing, the `user message`
channel, the first model author present, and the first assistant present. Columns that never
differ inside a trial are dropped, and a column identical to an earlier one, up to sign, is
reported as its alias rather than fitted twice; the rank of the remaining columns is reported
because linearly dependent columns leave the coefficients, though not the fitted
probabilities, non-unique. The L1 penalty runs down a log-spaced path from the value that
zeroes every feature. Each point is solved by proximal Newton steps: the quadratic surrogate is
minimised by coordinate descent over a Gram matrix of the active features, with the offsets in
closed form, and one pass over every column admits the features whose gradient violates the
optimality condition. A backtracking line search on the penalized objective guards each step.
K-fold cross-validation assigns folds by cell pair, so both orderings and every rollout of one
study are held out together, refits the path per fold with the penalties scaled to the fold's
share of the comparisons (the loss is a sum, so an unscaled penalty would bind harder on a
smaller fold), and picks the largest penalty within one standard error of the minimum held-out
loss; that fit is the one reported. The offsets keep the ranking's L2 penalty, which bounds
them under separation.

Collection routing is resolved in `openrouter.select_route` from invocation-local
`routing.CollectionRouting`. Collection checks paid permission before cold assistant work and
before missing warm QA/judgments. Author-message and trace records retain requested slug/transport
provenance separately from raw provider responses. Both fingerprint paths use the same shallow
response projection, preserving all metadata except recognized top-level model route suffixes.
