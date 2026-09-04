# reasonese

`reasonese` is a research codebase for studying how the same instruction changes across
three controlled axes—framing, channel, and author—applied to both sides of a
mutually exclusive instruction pair.

## Current foundation

The repository defines the axes, executes ordered multi-instruction matchups through
OpenRouter, and independently judges whether each instruction was completed. Statistical
analysis turns permutation-balanced datasets into cell rankings, axis comparisons, and order
effect diagnostics. An independent message-QA gate checks each materialized instruction against
its exact datapoint authoring instructions before assistant inference.

| Axis | Current values |
|---|---|
| framing | `normal`, `casual`, `persuasive`, `subagent`, `reasonese-normal`, `reasonese-persuasive` |
| channel | `system prompt`, `user message`, `README.md` |
| author | `user`, `Qwen3.8 Flash`, `Qwen3.8 2.4T`, `Inkling`, `Inkling Small` |

Framing and author are independent. “Normal” is the author's default rendering in clear
prose. A `user`-authored input is treated as already written and used verbatim; model authors
rewrite the base instruction according to the selected framing.

Six framings, three channels, and five authors produce `6 × 3 × 5 = 90` specifications per
base instruction. A specification is just a four-field dataclass containing those axes.

Instruction is not a treatment axis. Instructions come in 24 mutually exclusive pairs, and a
trial only ever holds the two instructions of one pair, so no comparison ever crosses a pair
boundary. Instruction selects which cells can be compared; it is a blocking factor for sampling
and analysis, not a coordinate whose levels are contrasted. See
[`docs/research/axes.md`](docs/research/axes.md).

## Quickstart

```bash
git clone https://github.com/d0rbu/reasonese.git
cd reasonese
uv sync

uv run reasonese-axes
uv run reasonese-plan \
  --pairs configs/instruction_pairs.yaml \
  --output out/example/prompt_specs.jsonl

uv run reasonese-sample-studies \
  --pairs configs/instruction_pairs.yaml \
  --pairings-per-pair 720 \
  --seed 0 \
  --output out/example/studies.yaml

export OPENROUTER_API_KEY=...
uv run reasonese-curate-instructions --output out/instructions

uv run reasonese-run-conversation \
  --matchup configs/example_matchup.yaml \
  --user-messages prompts/user \
  --message-cache out/generated_messages.yaml \
  --trace-cache out/conversation_traces.yaml

uv run reasonese-check-messages \
  --message-cache out/generated_messages.yaml \
  --qa-cache out/message_qa.yaml

uv run reasonese-judge-responses \
  --matchup configs/example_matchup.yaml \
  --trace-cache out/conversation_traces.yaml \
  --judgment-cache out/judgments.yaml

uv run reasonese-collect-data \
  --study configs/example_study.yaml \
  --user-messages prompts/user \
  --output out/example-study

uv run reasonese-collect-studies \
  --suite out/example/studies.yaml \
  --user-messages prompts/user \
  --output out/example-suite

uv run reasonese-analyze \
  --observations out/example-study/observations.jsonl \
  --pairs configs/instruction_pairs.yaml \
  --output out/example-study/analysis
```

The utilities have separate entry points. `reasonese-axes` prints the values and
`reasonese-plan` writes four-axis datapoints for both sides of every instruction pair. `reasonese-run-conversation` loads a `Matchup`,
generates any missing model-authored messages, constructs the ordered conversation, and sends
it to the selected assistant with file-read, sandboxed bash, sandboxed Python, and web-search
tools. It submits independent model-author batch jobs before polling them together, so one
author model's queue does not block another author's submission. `--no-batch` forces
synchronous authoring requests. Bash and Python execution require `bubblewrap` (`bwrap`) on the
host.

A matchup contains one assistant plus an ordered pair of inputs, at least one of which must use
the explicit `user message` channel. Repeated channels are valid. Generated
messages and complete raw assistant traces—including intermediate tool calls, tool results, and
returned reasoning fields—are cached as readable YAML. A warm trace-cache hit does not require
an API key or make a provider call once its exact messages also have cached passing QA. A
`README.md` treatment appears as an assistant `read_file` call followed by a tool result, rather
than as a wrapper inside a user message.

User-authored variants live under `prompts/user/<instruction>/`. Each directory contains the
exact base text in `instruction.txt` plus one text file for each framing. The checked-in variant
files are explicit `TODO:` placeholders; replace the variants you plan to run. A selected
placeholder or incomplete instruction directory fails before inference. Editing a manual variant
invalidates cached text and traces that contain its previous contents.
The collector reads the needed variants into one immutable snapshot per invocation, avoiding
repeated directory scans while ensuring edits are picked up on the next run.

Before a new conversation is sent to its experimental assistant, `openai/gpt-5.6-luna:batch`
independently checks every exact materialized message against the same authoring instructions
derived from its datapoint. It returns a strict `complies` boolean plus concrete issues. Any false
verdict stops the run without regenerating or selecting a replacement. Raw QA responses are
cached in `out/message_qa.yaml`; `reasonese-check-messages` exposes the same audit as a separate
nonzero-exit utility.

`reasonese-judge-responses` submits one independent judge request per matchup input in a single
`openai/gpt-5.6-luna:batch` job with medium reasoning. Each strict result is a boolean answering
“did the visible assistant response complete this request?” There is no winner constraint, so
all verdicts may be true or all may be false. Judgments and their complete raw judge responses
are cached in YAML against a fingerprint of the exact conversation trace.

`reasonese-collect-data` treats each four-axis datapoint plus the chosen assistant as one cell.
It requires exactly two distinct inputs, runs both input orderings, and collects one or more
rollouts per ordering. With `r` rollouts, the design has `2r` trials; every cell receives `2r`
verdicts and appears `r` times at each position. The collector runs uncached assistant work
through eight-worker, completion-driven concurrency by default: as soon as one response requests a local tool,
its continuation is submitted without waiting for slower peer responses. It batches judge work,
resumes from per-rollout caches, and writes flat analysis-ready rows to `observations.jsonl`.
Initial requests are admitted round-robin across assistant models, and ready tool continuations
receive freed worker slots before fresh requests, so a large group cannot monopolize the queue.
Assistant requests retain the
OpenRouter web-search tool and therefore use the synchronous API because OpenRouter does not
support that server tool in batch jobs. Definite HTTP 429 responses are retried a bounded number
of times using the provider's `Retry-After` delay when present. The same manual-message hierarchy
and message-QA gate apply to user-authored study inputs.

High-volume collector traces and judgments are JSON payloads in SQLite while retaining complete
raw provider responses. A standalone study or repeated `--study` input keeps one
`collection.sqlite3` per study. A sampled `--suite` instead uses one shared database at the suite
root, avoiding tens of thousands of duplicate database files and reducing each collection stage
to one cache read and transaction. Trace fingerprints are derived once and reused through
judgment and batched observation construction; the standalone one-conversation utilities keep
their readable YAML caches.
Study trials likewise share their two validated ordered matchups, which cache readers reuse while
checking the redundant serialized coordinates for equality.

`reasonese-collect-studies` accepts repeated `--study` paths and batches work across those study
boundaries. It shares generated-message and message-QA caches at the output root, combines all
active assistant models and trials into one completion-driven scheduler, and submits all
uncached response judgments together. Each study keeps its own directory of traces, judgments,
and observations, named after the study file's stem. Study filenames must therefore have
distinct stems.

`reasonese-sample-studies` avoids exhaustive condition pairing. It samples one instruction pair
at a time, because only the two instructions of a pair can be compared. Within a pair the two
sides are disjoint sets of specifications, so the eligible population is bipartite: every valid
edge joins a first-side cell to a second-side cell and at least one of them uses the `user
message` channel. It writes one YAML suite and replicates that exact design across the selected
assistants.

The default is 720 pairings per pair, capped by the eligible population and raised if a larger
design is needed to connect every cell. An explicit `--pairings-per-pair` overrides it. The
sampler gives exact proportional quotas to strata defined by the channel pair and by which of
framing and author differ across the edge. Instruction always differs, so it is no longer a
stratum dimension. Within each stratum it prefers cells with lower channel-normalized degree.
Both input orders and all requested rollouts are still collected. The requested count must be at
least `cells in the pair - 1` and cannot exceed that pair's population. A connectivity check then
replaces exactly `components - 1` redundant cycle edges, the minimum possible repair, preferring
same-stratum replacements. Seeds derive from the pair id, so reordering the bank does not change
any design.

With all axes enabled, each of the 24 pairs has 180 cells and 4,500 eligible pairings, for
108,000 eligible pairings per assistant. The minimum connected design uses 179 pairings per pair.
The pilot default of 720 gives every cell an average degree of 8.

Connectivity is required within a pair and is impossible between pairs, so the comparison graph
has exactly one component per `(pair, assistant)`.

By default the sampler includes every author and assistant. Repeated `--author` and
`--assistant` options restrict those sets—for example, omitting the `user` author avoids selecting
manual variants that have not been written yet. `reasonese-collect-studies --suite` consumes the
artifact directly, uses study fingerprints for resumable subdirectories, and writes the combined
analysis input to `observations.jsonl` at the suite root.

The design preserves the eligible population's stratum composition rather than weighting rare
strata equally. Degree-aware selection also means individual edges do not have uniform inclusion
probabilities. These choices and any connectivity repairs should be considered before treating
marginal axis summaries as confirmatory estimates.

`reasonese-analyze` fits an L2-penalized Bradley–Terry ordering **per connected component**,
which is one `(pair, assistant)` block. A completed cell beats an incomplete one; equal verdicts
contribute half-wins. It needs `--pairs` so it can map each instruction to its pair, side, skill,
and conflict type.

Marginal summaries and pairwise contrasts cover framing, channel, and author, the only
coordinates that vary inside a trial and therefore the only ones a within-component score can
speak to. Assistant, skill, conflict type, and pair are constant within a trial, so they get a
separate stratum table of completion rates with no Bradley–Terry column. The analysis also writes
per-pair exclusivity counts, overall position rates, cell-by-position and axis-by-position
effects, order-sensitivity ranges/correlations, position-balance checks, trial-cluster bootstrap
intervals, and rankings under 0.1×/1×/10× regularization.

Per-pair exclusivity is the empirical check on the bank. A both-completed trial means one
response satisfied both instructions, so the pair was not exclusive in practice; a
neither-completed trial means it was too hard. Bradley–Terry scores both as ties, which is why
they are reported apart.

The report keeps important limits visible: ranks are within-component and carry no meaning
across components, and axis margins are descriptive unless the collected cells form an
appropriate balanced factorial design.

## Instruction bank

Studies need instruction pairs that one response cannot both complete, so the bank in
[`configs/instruction_pairs.yaml`](configs/instruction_pairs.yaml) is a list of pairs, each with a
skill, a conflict type, and a rationale. `reasonese-curate-instructions` reports coverage and
cross-pair lexical overlap, audits every pair's feasibility, tool requirement, difficulty band, and
mutual exclusivity in one `openai/gpt-5.6-luna:batch` job, caches the raw audits, and writes
`out/instructions/report.md` with every pair's full text for manual spot-checking. It exits nonzero
when any audited pair fails. `--scaffold-user-prompts prompts/user` creates the placeholder manual
variant directories that new instructions need. The criteria and conflict taxonomy are in
[`docs/research/instruction-bank.md`](docs/research/instruction-bank.md).

## Implementation notes

The enums use their display text directly; there is no separate label mapping. Base
instructions and non-negative counts use `phantom-types`, while `beartype` checks public
function and dataclass boundaries at runtime.

- [`docs/research/axes.md`](docs/research/axes.md) defines the research constructs.
- [`docs/reference/architecture.md`](docs/reference/architecture.md) describes the small implementation.
- [`docs/reference/prompt-generation.md`](docs/reference/prompt-generation.md) traces exact prompt construction.
- [`docs/reference/output.md`](docs/reference/output.md) describes plan, trace, and judgment artifacts.

## Development

```bash
uv run pre-commit run --all-files
```

## License

MIT. See [`LICENSE`](LICENSE).
