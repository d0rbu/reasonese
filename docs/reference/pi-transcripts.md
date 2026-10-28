# Pi source transcripts

The original scenario YAML files contain authored conversation histories. This collector
instead checks out the real repositories, sends a user task to Pi, and lets Pi choose and
execute its tools until it produces an answer. Interactive collection keeps that same
session and workspace for operator-authored follow-ups grounded in observed output. The pinned harness is
[`@mariozechner/pi-coding-agent` 0.73.1](https://github.com/badlogic/pi-mono/tree/v0.73.1/packages/coding-agent).
The manifest selects exactly `deepseek/deepseek-v4.1-flash-1017` or
`moonshotai/kimi-k3.1-pro` on OpenRouter, with high reasoning. The requested slug is
checked against model metadata and every actual HTTP request; it is never substituted.
There is no fallback model, authoring model, or judge in this collection stage.

`configs/pi_transcripts.json` specifies one task for each of the 12 scenario pairs,
with pinned source repository commits. The first, `langchain-json-recovery`, replaces
the leap-year task with a real parser investigation and conflicting final-response
policies. Its single-scenario manifest is `configs/pi_langchain_review.json`. The other
eleven prompts retain their original small exercises and omit the competing constraints. The repository's
actual root `AGENTS.md` (or its resolved `CLAUDE.md` target) supplies the project instructions.
Pi supplies its own system prompt. README files and earlier assistant/tool messages are never
invented, replaced, or inserted by the collector.

These are **task-authored, model-executed source sessions**, not transcripts from human users
and not measurements of the framing/channel/author treatments. Even a normally completed
session may fail its task or ignore a rule. Completion is recorded separately from task success,
which remains unjudged. The original one-turn leap-year pilot has completed. Its evidence is retained separately;
the revised JSON parser scenario uses a fresh output directory.

## Prepare without model calls

Use Linux with Bubblewrap, Git, Node.js 22 or newer, and the repository's Python environment:

```bash
uv sync --locked --extra probes
npm ci --prefix tools/pi-transcripts --ignore-scripts --no-audit --no-fund
node tools/pi-transcripts/collect.mjs prepare --output out/pi-transcripts-initial
```

Preparation downloads complete shallow checkouts at the pinned commits and gives every
scenario independent Git storage. A neighboring `out/pi-transcript-sources/` cache shares the
downloads across preparations; it is not exposed to the agent's tools. Preparation verifies pristine files and the resolved instruction
source, checks Python/pytest/Git inside the tool sandbox, and saves an exact collection plan.
It also checks the requested model against OpenRouter's public model catalog. Neither
repository downloads nor catalog lookup send prompts to a model.

Preparation may be repeated to refresh collector hashes before any session has started; earlier
plans are retained. A changed task manifest or any existing trajectory requires a fresh run.

The eleven legacy tasks use Python's standard library, pytest, and local Git commits. Upstream submodules,
large project build environments, and full repository test suites are not installed. Each
workspace retains the real upstream files; this is a deliberately bounded coding environment,
not a claim to reproduce each project's development machine. Pi can inspect further files
normally, but extensions, skills, ancestor instructions, and personal Pi settings from the host
are excluded. Tool subprocesses have their own network namespace, writable checkout and
private temporary/home directories, read-only runtime dependencies, and no API credentials.
Socket tasks can bind local ports inside that namespace. Git remotes are never pushed.

## Start an authorized collection

Once the prepared plan and collection spending have been approved, set
`OPENROUTER_API_KEY` in the environment and run:

```bash
node tools/pi-transcripts/collect.mjs collect \
  --output out/pi-transcripts-initial --allow-paid
```

The 12-task manifest now contains an interactive task and cannot be collected as an
unattended batch. Select a reviewed subset in a separate manifest. Legacy sessions are
sequential. Each is limited to 24 model requests,
8,192 output tokens per request, 100,000 bytes per serialized input request, 15 minutes per
session, and 120 seconds per tool invocation. Input limits reject a request rather than
truncate its history. Pi compaction and both Pi/transport retries are disabled; a bounded or
failed session stops the batch with its partial evidence intact. These controls bound the
workload; usage and the catalog-based cost estimates are saved for inspection.

Repeating `collect` skips verified completed attempts. Any existing incomplete attempt requires
inspection and a new output directory for a separately authorized retry. The collector also
rejects changed plans or collector files. Do not delete failed trajectories to obtain a clean
denominator. The `--allow-paid` opt-in is required even when checking an already completed run.

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
that next design step. In particular, the Expo Python task and the commit-title task still need
scope review before use as exclusive comparisons: their upstream rules discuss `console.log`
and PR titles respectively.

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
