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

`configs/pi_transcripts.json` holds twelve base-transcript tasks. Each has one
`baseInstruction`, copied verbatim from the first instruction in
`configs/scenario_pairs.yaml`, and a consistent investigation/implementation flow.
`configs/pi_remaining_review.json` selects the remaining eleven with Kimi K3.1 Pro;
`configs/pi_langchain_base.json` selects the first-scenario Kimi run. Its
[clean base transcript](../research/langchain-json-base.md) completed in two user
turns. The [complete twelve-session collection](../research/kimi-base-collection.md)
now includes all eleven remaining cases, also with two user turns each. No second
instruction or framing assignment appears in an active collection manifest.

The existing templating system will introduce conflicting instructions and style
variants later, using the recorded base context. The pair bank retains both sides
for that step. The assistant's actual root project instructions remain background
context; README reads and assistant/tool history are never invented.

These are task-authored, model-executed source sessions. They are not human logs or
framing measurements. The previous DeepSeek and Kimi JSON parser runs included
both requirements in turn two and are preserved as historical conflict pilots,
not clean bases. Their original manifests remain unchanged for provenance and are
rejected by the current base-collection command.

## Prepare without model calls

Use Linux amd64 with Bubblewrap, unshare, slirp4netns, nftables, Git, rg,
Node.js 22 or newer, npm, uv, and Bun 1.3.12
(default `~/.bun/bin/bun`, or set `PI_BUN_PATH`):

```bash
uv sync --locked --extra probes
npm ci --prefix tools/pi-transcripts --ignore-scripts --no-audit --no-fund
node tools/pi-transcripts/collect.mjs prepare \
  --manifest configs/pi_remaining_review.json --output out/pi-base-remaining-review
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

Each tool subprocess has its own routed network namespace, neutral checkout/cwd,
private temporary/home directories, writable task-owned dependency caches and
read-only shared runtimes, with no host API credentials. Public development
network access is allowed while host loopback, private networks and metadata
addresses are blocked by an outer-namespace firewall. Pi extensions, skills, ancestor instructions and personal settings are
excluded. Git remotes are never pushed. Dependencies are resolved before collection and tools can download additional
public dependencies. See the [development environment](pi-development-environment.md)
for exact mounts, network boundaries, tools and complete per-turn checkpoints.

Preparation may refresh collector hashes before any session has started; previous
plans are retained. A changed manifest or any existing trajectory requires a fresh
output directory. Failed preparation logs are preserved for diagnosis.

## Start one authorized pilot

After explicit authorization for the selected pilot, set `OPENROUTER_API_KEY` in
the environment. Select exactly one prepared case without editing the frozen plan:

```bash
node tools/pi-transcripts/collect.mjs collect --interactive --allow-paid \
  --manifest configs/pi_remaining_review.json --output out/pi-base-remaining-review \
  --scenario airflow-redaction-marker
```

Use another reviewed ID for a later authorized pilot. `--scenario` is a collection
selector; `prepare` always prepares the whole manifest. An unknown selector or an
interactive invocation selecting multiple cases is rejected before provider access.
Neither `--allow-paid` nor preparation itself grants authorization to start a batch.

After each completed user turn, inspect the saved checkpoint, tools and patch, then
send one JSON line with `prompt` and `rationale`, or `null` to finish. Only the prompt
is delivered to the model. See the [operator protocol](../research/remaining-scenario-pilots.md#ordinary-multi-turn-collection)
for following the first instruction as written in turn two and grounding later
follow-ups in observed evidence while keeping that same task. Target two user
turns, permit up to four, and do not manufacture extra turns.

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

## First-scenario base transcript

The historical parser runs are not reused as bases. Prepare a fresh workspace for
ordinary investigation and implementation of the first instruction (JSON recovery):

```bash
node tools/pi-transcripts/collect.mjs prepare \
  --manifest configs/pi_langchain_base.json --output out/pi-json-base
```

Only after collection is explicitly authorized:

```bash
node tools/pi-transcripts/collect.mjs collect --interactive --allow-paid \
  --manifest configs/pi_langchain_base.json --output out/pi-json-base
```

Preparation installs the pinned LangChain core test environment with `uv sync
--locked --no-default-groups --group test`. Lock/package records are retained and
the dependencies are mounted read-only. The selected requirement stays recovery
through implementation and subsequent reviews; rejection is reserved for later
template injection. Target two user turns, with at most four if observed work needs
follow-up. The first authorized run under this protocol completed with two user
turns, 17 model requests and 15 tool results. See the
[recorded base session](../research/langchain-json-base.md) for its exact scope,
independent checks and viewer.

## Historical conflict pilots

`configs/pi_langchain_review.json` and `configs/pi_langchain_kimi_review.json`
record the earlier DeepSeek/Kimi setup. Both sent the recovery and rejection
requirements in turn two with a casual/persuasive assignment. Those deliveries were
a design error for the requested base-corpus purpose. Preserve the original raw
messages and manifests; do not relabel them as clean bases or resume their plans
with the new collector. See the [historical record](../research/langchain-json-recovery.md).

The earlier Kimi run returned readable provider reasoning. Future authorized runs
must retain whatever the provider actually returns, including raw SSE and Pi
thinking blocks; do not assume reasoning availability or claim that it completely
or causally describes internal computation.

## Retained evidence

Each `<scenario>/trajectory/` contains:

| Artifact | Contents |
|---|---|
| `provenance.json` | Exact task, repository/revision, rule bytes and resolved source path, limits, model metadata, Pi/Node versions, and collector/dependency hashes |
| `prompt.txt`, `system.txt`, `tools.json` | Delivered user prompt, resolved Pi system prompt, and tool schemas |
| `sessions/*.jsonl` | Pi's native session format, including tool calls, results, model messages, usage, and stop reasons |
| `events.jsonl`, `messages.json` | Full Pi event stream and final ordered message list, including partial/error messages |
| `http/*.request.json`, `http/*.response.*` | Actual request payloads, HTTP status, and raw response/SSE bytes; authorization headers are excluded |
| `turns/NNN.workspace.tar.gz`, `turns/NNN.workspace.json` | Complete task state before collection and after each user turn, including Git, ignored/untracked files, mutable dependencies and home/tmp; hash and Git-state receipt |
| `changes.patch`, `outcome.json` | Tracked changes including local commits, terminal status, Git status/log, request/tool counts, usage, and hashes of every retained transcript artifact |

The adjacent `workspace/` and `tool-state/` remain available, including untracked files,
ignored outputs, local commits, and temporary files. `completed` means Pi reached a normal
final assistant message; it does not certify task success or instruction adherence. Raw
responses retain the provider's reported model and metadata separately from the requested slug.
Treat all output as research data; generated artifacts belong below ignored `out/`.

## Review website

The current website uses explicitly labeled path-normalized copies. Original
captures and immutable source hashes remain available separately. See the
[environment audit](../research/transcript-environment-audit.md) for substitutions,
remaining evaluation/sandbox cues and the limits of offline editing.

[`tools/pi-transcripts/viewer/`](../../tools/pi-transcripts/viewer/README.md) provides
static pages with a shared scenario catalog and clickable cards in the left sidebar.
Each page includes the operator's explainer, all original messages, expandable
provider-returned reasoning, changed files, independent checks and raw evidence.
Search and role filters change only the display. Annotations and validation
receipts remain separate from the recorded user and assistant messages.

Copy the viewer assets beside each exported `data.json`, retain evidence downloads,
and write `/kimi-transcripts/catalog.json` as described in the viewer README. The
approved first transcript keeps `/kimi-json-recovery/`; the remaining cases use
`/kimi-transcripts/<scenario-id>/`. Historical conflict-pilot exports stay at their
archive paths. Export only reviewed public artifacts, check source hashes and
credentials, and verify navigation plus served bytes through the public tunnel.

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
