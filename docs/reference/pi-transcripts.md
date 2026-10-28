# Pi source transcripts

The collector checks out real repositories, sends a user task to Pi, and lets Pi
choose and execute its tools until it produces an answer. Interactive collection
keeps that same session and workspace for operator-authored follow-ups grounded in
observed output. The pinned harness is
[`@mariozechner/pi-coding-agent` 0.73.1](https://github.com/badlogic/pi-mono/tree/v0.73.1/packages/coding-agent).
The manifest selects exactly `deepseek/deepseek-v4.1-flash-1017` or
`moonshotai/kimi-k3.1-pro` on OpenRouter, with high reasoning. Metadata and every
actual HTTP request are checked for that model; there is no substitution, fallback
model, authoring model or judge in this collection stage.

`configs/pi_transcripts.json` holds twelve repository-native tasks with pinned
commits and conflicting behavior requirements. The reviewed
[LangChain JSON parser task](../research/langchain-json-recovery.md) replaces the
original leap-year exercise. The [other eleven tasks](../research/remaining-scenario-pilots.md)
now have comparable investigation/implementation protocols, source references,
exclusive cases, shared requirements and review checks. Their dedicated manifest,
`configs/pi_remaining_review.json`, selects Kimi K3.1 Pro. Their pilots have not
started. The actual root project instructions supply background context; README
files and earlier assistant/tool messages are never invented by the collector.

These are **task-authored, model-executed source sessions**, not human participant
logs or measurements of the framing/channel/author treatments. Completion means
Pi reached a final assistant message, separately from task success. Retain the
original leap-year and both JSON parser runs as separate historical evidence.

## Prepare without model calls

Use Linux amd64 with Bubblewrap, Git, Node.js 22 or newer, uv, and Bun 1.3.12
(default `~/.bun/bin/bun`, or set `PI_BUN_PATH`):

```bash
uv sync --locked --extra probes
npm ci --prefix tools/pi-transcripts --ignore-scripts --no-audit --no-fund
node tools/pi-transcripts/collect.mjs prepare \
  --manifest configs/pi_remaining_review.json --output out/pi-remaining-review
```

Preparation downloads complete shallow checkouts at the pinned commits and gives
every scenario independent Git storage. A neighboring `out/pi-transcript-sources/`
cache shares downloads without exposing them to model tools. Preparation checks
pristine source, cited files and resolved instruction targets, installs scoped
dependencies, runs each manifest's native baseline and real-source probes in the
sandbox, and saves the collection plan and receipts. It also looks up the requested
model in OpenRouter's public catalog. **No inference requests are made.**

The environments cover editable Python subpackages, locked LangChain tests,
checksum-verified Go 1.26.9 and required module caches, Bun, and narrowly installed
JavaScript test dependencies with lifecycle scripts disabled. Expo uses a scoped
Jest configuration for its real env-package tests with the same memfs convention.
See [validation scope](../research/remaining-scenario-pilots.md#preparation-and-validation-scope)
for the exact boundaries. No full monorepo build, submodules, Docker daemon, cloud
CI execution or release publication is required.

Each tool subprocess has its own network namespace, writable checkout and private
temporary/home directories, read-only runtime/dependency mounts, and no host API
credentials. Pi extensions, skills, ancestor instructions and personal settings are
excluded. Git remotes are never pushed. Go's module network access is also disabled
inside tools; dependencies are resolved before collection.

Preparation may refresh collector hashes before any session has started; previous
plans are retained. A changed manifest or any existing trajectory requires a fresh
output directory. Failed preparation logs are preserved for diagnosis.

## Start one authorized pilot

After explicit authorization for the selected pilot, set `OPENROUTER_API_KEY` in
the environment. Select exactly one prepared case without editing the frozen plan:

```bash
node tools/pi-transcripts/collect.mjs collect --interactive --allow-paid \
  --manifest configs/pi_remaining_review.json --output out/pi-remaining-review \
  --scenario airflow-redaction-marker
```

Use another reviewed ID for a later authorized pilot. `--scenario` is a collection
selector; `prepare` always prepares the whole manifest. An unknown selector or an
interactive invocation selecting multiple cases is rejected before provider access.
Neither `--allow-paid` nor preparation itself grants authorization to start a batch.

After each completed user turn, inspect the saved checkpoint, tools and patch, then
send one JSON line with `prompt` and `rationale`, or `null` to finish. Only the prompt
is delivered to the model. See the [adaptive operator protocol](../research/remaining-scenario-pilots.md#adaptive-user-turns)
for preserving the frozen conflict spans in turn two and grounding any later
follow-up in observed evidence. Target two user turns, permit up to four, and do
not manufacture extra turns.

Each remaining pilot allows 80 model requests total, 8,192 output tokens per request,
400,000 bytes per serialized input, 40 minutes including operator review, and 120
seconds per tool. Limits apply across all turns. History is not compacted or
truncated; Pi/transport retries are disabled. Retain partial evidence on failure or
limits. These bounds constrain workload; usage and catalog-based cost estimates are
saved separately.

Repeating `collect` skips verified completed attempts. Incomplete attempts require
inspection and a fresh output directory for a separately authorized retry. Changed
plans or collector files are rejected. Do not delete failed trajectories to obtain
a clean denominator. `--allow-paid` is required even when checking a completed run.

## Adaptive JSON parser pilot

```bash
node tools/pi-transcripts/collect.mjs prepare \
  --manifest configs/pi_langchain_review.json --output out/pi-json-review
node tools/pi-transcripts/collect.mjs collect --interactive --allow-paid \
  --manifest configs/pi_langchain_review.json --output out/pi-json-review
```

Preparation installs the pinned LangChain core test environment with `uv sync --locked
--no-default-groups --group test` into a separate environment. Its lock hash and package
list are retained; the sandbox mounts it read-only while the source checkout remains
writable. This setup downloads dependencies but sends no inference requests.

After each completed user turn, read the saved messages, tool outputs and patch. Send
one JSON line on stdin with `prompt` and `rationale` strings, or `null` to finish. Only
`prompt` is sent to the model; the operator rationale is audit metadata. The CLI permits
one scenario per interactive invocation. End-of-input without an explicit finish is an
error, and failures retain their partial evidence. Maximum user turns also end a run.

This pilot allows four user turns, 80 model requests total, 400,000 serialized input
bytes per request, 8,192 output tokens per request, 40 minutes total (including operator
review), and 120 seconds per tool. Request budgets are shared across turns, and no
history is compacted or truncated. `turns/*.input.json` records each exact user message,
its rationale, and the preceding message hash; `turns/*.messages.json` and `*.patch`
retain each observed checkpoint. The same native Pi session continues throughout.

See [the scenario design](../research/langchain-json-recovery.md) for the conflict,
planned review stages, and limits on interpreting this one adaptive example.

### Kimi rerun

`configs/pi_langchain_kimi_review.json` differs from the DeepSeek pilot manifest only
in its model. Use a fresh directory for its workspace and complete evidence:

```bash
node tools/pi-transcripts/collect.mjs prepare \
  --manifest configs/pi_langchain_kimi_review.json --output out/pi-kimi-json-review
node tools/pi-transcripts/collect.mjs collect --interactive --allow-paid \
  --manifest configs/pi_langchain_kimi_review.json --output out/pi-kimi-json-review
```

The intended flow is investigation followed by implementation, with additional user
turns only for observed issues (four remains the safety limit). Keep the initial task
and conflicting instruction spans unchanged. End the session with `null` after review
when no further model work is needed; do not manufacture turns to reach the limit.

An authorized one-request preflight on 2026-10-28 confirmed that this Kimi endpoint
returned readable `reasoning` / `reasoning.text`, with no encrypted blocks in that
response. Verify the actual session too: raw SSE reasoning must match Pi's stored
thinking blocks and be preserved in subsequent tool-continuation requests. The viewer
shows provider-returned reasoning as recorded; availability does not establish that
it is a complete or causal account of the model's internal computation. Keep the probe
separate from the scenario's request count, evidence, and cost.

## Retained evidence

Each `<scenario>/trajectory/` contains:

| Artifact | Contents |
|---|---|
| `provenance.json` | Exact task, repository/revision, rule bytes and resolved source path, limits, model metadata, Pi/Node versions, and collector/dependency hashes |
| `prompt.txt`, `system.txt`, `tools.json` | Delivered user prompt, resolved Pi system prompt, and tool schemas |
| `sessions/*.jsonl` | Pi's native session format, including tool calls, results, model messages, usage, and stop reasons |
| `events.jsonl`, `messages.json` | Full Pi event stream and final ordered message list, including partial/error messages |
| `http/*.request.json`, `http/*.response.*` | Actual request payloads, HTTP status, and raw response/SSE bytes; authorization headers are excluded |
| `changes.patch`, `outcome.json` | Tracked changes including local commits, terminal status, Git status/log, request/tool counts, usage, and hashes of every retained transcript artifact |

The adjacent `workspace/` and `tool-state/` remain available, including untracked files,
ignored outputs, local commits, and temporary files. `completed` means Pi reached a normal
final assistant message; it does not certify task success or instruction adherence. Raw
responses retain the provider's reported model and metadata separately from the requested slug.
Treat all output as research data; generated artifacts belong below ignored `out/`.

## Use as the basis for realistic scenarios

Inspect these source sessions before choosing prefixes and insertion locations for controlled
comparisons. Preserve the unmodified source transcript and record every subsequent edit and its
source message/tool-call identifiers. A real read may expose nested instructions or additional
constraints, and a trajectory may finish without reading README at all; those are observations,
not reasons to fabricate a read.

The current `--scenarios` format accepts only a restricted hand-authored history. This PR does
not silently squeeze full Pi sessions into that format, fabricate trial placements, judge the
source runs, or treat them as analysis-ready observations. The source corpus is the basis for
that next design step. The former Expo debug-print and commit-title exercises have
been replaced by native dotenv and runnable-fallback behavior conflicts.

## Offline verification

```bash
npm --prefix tools/pi-transcripts test
uv run pre-commit run --all-files
```

The integration tests run the actual pinned Pi agent against a deterministic localhost HTTP
fixture and execute all four real tools in Bubblewrap. They verify tool-result round trips,
native-session fidelity, local Git changes, isolation, opt-in, request/size/time limits, error
retention, and no paid retries. Fixture evidence is labeled `offline_test_fixture` and removed
after tests. Passing these checks validates the collector, not future model behavior.
