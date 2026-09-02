# reasonese

`reasonese` is a research codebase for studying how the same instruction changes across
four controlled axes: instruction, framing, channel, and author.

## Current foundation

The repository defines the axes, executes ordered multi-instruction matchups through
OpenRouter, and independently judges whether each instruction was completed. Statistical
analysis turns permutation-balanced datasets into cell rankings, axis comparisons, and order
effect diagnostics. An independent message-QA gate checks each materialized instruction against
its exact datapoint authoring instructions before assistant inference.

| Axis | Current values |
|---|---|
| instruction | Simple base prompts such as “Write a program…” or “Find information…” |
| framing | `normal`, `casual`, `persuasive`, `subagent`, `reasonese-normal`, `reasonese-persuasive` |
| channel | `system prompt`, `user message`, `README.md` |
| author | `user`, `Qwen3.8 Flash`, `Qwen3.8 2.4T`, `Inkling`, `Inkling Small` |

Framing and author are independent. “Normal” is the author's default rendering in clear
prose. A `user`-authored input is treated as already written and used verbatim; model authors
rewrite the base instruction according to the selected framing.

Six framings, three channels, and five authors produce `6 × 3 × 5 = 90` specifications per
base instruction. A specification is just a four-field dataclass containing those axes.

## Quickstart

```bash
git clone https://github.com/d0rbu/reasonese.git
cd reasonese
uv sync

uv run reasonese-axes
uv run reasonese-plan \
  --instructions configs/example_instructions.toml \
  --output out/example/prompt_specs.jsonl

export OPENROUTER_API_KEY=...
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

uv run reasonese-analyze \
  --observations out/example-study/observations.jsonl \
  --output out/example-study/analysis
```

The utilities have separate entry points. `reasonese-axes` prints the values and
`reasonese-plan` writes four-axis datapoints. `reasonese-run-conversation` loads a `Matchup`,
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
through bounded, completion-driven concurrency: as soon as one response requests a local tool,
its continuation is submitted without waiting for slower peer responses. It batches judge work,
resumes from per-rollout caches, and writes flat analysis-ready rows to `observations.jsonl`.
Assistant requests retain the
OpenRouter web-search tool and therefore use the synchronous API because OpenRouter does not
support that server tool in batch jobs. Definite HTTP 429 responses are retried a bounded number
of times using the provider's `Retry-After` delay when present. The same manual-message hierarchy
and message-QA gate apply to user-authored study inputs.

High-volume collector traces and judgments are JSON payloads in one `collection.sqlite3` file
per study. This batches filesystem reads and transactional writes while retaining complete raw
provider responses. Trace fingerprints are derived once and reused through judgment and batched
observation construction; the standalone one-conversation utilities keep their readable YAML
caches.

`reasonese-collect-studies` accepts repeated `--study` paths and batches work across those study
boundaries. It shares generated-message and message-QA caches at the output root, combines all
active assistant models and trials into one completion-driven scheduler, and submits all
uncached response judgments together. Each study keeps its own directory of traces, judgments,
and observations, named after the study file's stem. Study filenames must therefore have
distinct stems.

`reasonese-analyze` fits an L2-penalized Bradley–Terry ordering from each trial's cell pair.
A completed cell beats an incomplete one; equal verdicts contribute half-wins. It also
writes marginal summaries and pairwise contrasts for all five coordinates, overall position
rates, cell-by-position and axis-by-position effects, order-sensitivity ranges/correlations,
comparison-graph connectivity, position-balance checks, trial-cluster bootstrap intervals,
and rankings under 0.1×/1×/10× regularization.

The report keeps important limits visible: axis margins are descriptive unless the collected
cells form an appropriate balanced factorial design, and absolute ordering between disconnected
Bradley–Terry components is regularization-dependent rather than empirically identified.

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
