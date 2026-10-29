# Native Pi templates and framing handoff

This path substitutes complete user messages into real Pi trajectories. It shares
`PromptSpec`, `Study`, `Trial`, author/framing/channel coordinates, and both input
orderings with the existing experiment planner. It preserves native reasoning,
tool calls, tool results, whitespace, system text and tool schemas. The older
README-backed scenarios and their runner remain available.

The pilot fixes the executor to `Kimi K3.1 Pro`
(`moonshotai/kimi-k3.1-pro` on OpenRouter), high reasoning, and channel to `user
message`. Message authors are human or Kimi, each with normal, casual and
persuasive framing. Author and executor remain independent axes. The full pilot
has 36 A/B format combinations per task, 2 orderings and 2 momentum levels: 144 plans
per task, 1728 for 12 tasks. Planning is offline; those counts are not collected
outcomes or an instruction to start paid runs.

## Source downloads

The pinned catalog in `configs/pi_native_sources.json` identifies the source
artifacts and their SHA-256/size. Use a fresh ignored directory:

```bash
uv run python tools/pi-transcripts/fetch-sources.py \
  --catalog configs/pi_native_sources.json --output out/native-context --context-only
```

Context downloads contain native history, system/tools and review metadata but
omit workspace archives. They are enough to read the source while authoring
messages. For planning/continuations, fetch complete sources into a different
fresh directory by omitting `--context-only`; add `--case CASE` to select one task.
Large source archives use ordered parts, each individually pinned; assembly must
also match the complete archive hash before extraction. Failed downloads and
partial files are retained. A full checkpoint still needs its recorded immutable
runtimes installed on the execution host.

## Max's editing workflow

The [prompt bank](../../prompts/scenario_turns/README.md) has 108 human TODOs and 108
Kimi-written complete user messages. Preserve the canonical policy, examples,
public APIs and task constraints while writing each style. `source.txt` is
verbatim source; `authoring-source.txt` is a separately documented adaptation.
The Kimi authoring passes used 117 requests, 0 failed requests and provider-reported
USD 0.803202, selecting 108 current messages. Nine outputs were regenerated after
review removed stale prior-edit narration and a policy-side narrative confound.
The original outputs and raw responses remain in the evidence bundle. It has structural checks and a manual semantic review; it has not
been approved by the independent message judge or tested in framing trials.

The source corpus is a separate immutable input. Each bank case has an explicit `source-binding.json` review receipt tying its
authoring-source/provenance files to the exact complete descriptor, native history
and checkpoint hashes. The descriptor pins the system, tools, restored environment,
limits and Git state; plans must equal its verified contents. A fresh resample requires a new applicability review and binding; the
compiler rejects missing or stale bindings and changed canonical policies. The
original source hashes remain historical provenance, not equality claims about
new user turns.

A source descriptor at
`out/CORPUS/CASE/template.json` pins message/system/tool files and checkpoints 0, 1, 2.
A checkpoint includes independent Git state, ignored/untracked project files,
mutable installed dependencies and tool home/tmp. Processes, sockets and remote
state are not captured. Read-only system/runtime mounts require the corresponding
prepared runtime on the execution host; a downloaded archive alone is not a
portable virtual machine. Restore validates hashes and rejects unsafe tar entries.

Compile the entire pilot as previews (human TODOs permitted):

```bash
uv run reasonese-native-plan \
  --corpus out/CORPUS --variants prompts/scenario_turns \
  --pilot-grid --allow-placeholders --output out/native-previews
```

For a selected experiment, provide an ordinary study suite with user-message
coordinates and the Kimi executor:

```bash
uv run reasonese-native-plan \
  --corpus out/CORPUS --variants prompts/scenario_turns \
  --suite configs/native_pi_example.yaml --output out/native-plans
```

Omit `--allow-placeholders` for executable plans. Unfilled selected human files
fail immediately. File paths and hashes are operator metadata; only the selected
message text goes to the assistant. Plan files are local resolved artifacts and
must be recompiled after relocating a corpus or editing a variant.

For one replacement, select a turn and a complete variant directly:

```bash
uv run reasonese-native-turn \
  --source out/CORPUS/langchain-json-recovery/template.json \
  --turn 2 --slot implementation --side b \
  --author "Kimi K3.1 Pro" --framing normal --output out/replacement.json
```

Use `--turn 1 --slot investigation` to redo investigation, or `--turn 3` to
append a later assignment after a two-turn source implementation. For longer
sources, append at the source user-turn count plus one.

## Checkpoints and downstream history

`replace_user_turn(source, n, variant)` restores checkpoint n−1 and keeps only
history before user turn n. All subsequent recorded messages and follow-up assignments are
discarded. Follow-ups need to be authored for the new assistant work; they are
not copied from an obsolete trajectory. For a two-turn source, appending turn 3
restores the completed implementation checkpoint 2. Investigation variants restart
from checkpoint 0. Investigation changes require a fresh investigation before
later implementation work.

The controlled pilot holds the investigation prefix fixed, restores checkpoint 1,
and supports two explicit schedules:

- `after_investigation`: deliver the first and second policy assignments as
  consecutive distinct user messages, then let Pi respond. There is no model work
  between these assignments. A provider may serialize adjacent user messages
  differently; raw HTTP captures are the evidence for what it received.
- `after_implementation`: deliver the first assignment, let Pi finish its complete
  tool trajectory, then deliver the second assignment in the same session and
  filesystem. This measures a condition with completed prior work and possible
  sunk-cost effects. The first implementation is regenerated for its actual
  authored input; no recorded implementation is attached to a changed prompt.

Both schedules counterbalance A→B and B→A. The source only pursues A; B is introduced
by experimental plans. A raw base transcript is never relabeled as a human
conversation or a conflict experiment.

## QA and bounded execution

The live wrapper rejects TODOs and validates exact source, prefix, checkpoint,
variant, and delivery identities before constructing a provider client. It uses
the existing hard Luna message-QA gate, then binds its receipt to the exact plan
bytes. Changed text or failed QA cannot proceed to the Pi executor.

```bash
uv run reasonese-native-run \
  --plan out/native-plans/after_implementation/TRIAL.json \
  --qa-cache out/native-message-qa.yaml \
  --out out/native-runs/TRIAL --allow-paid
```

Run this only after authorization for the concrete live experiment. `--allow-paid`
is required for uncached QA and for continuation inference; free routes do not
remove the QA spending requirement. The default source limits remain 80 actual
HTTP requests, 8192 output tokens per request, 400000 request bytes, 40 minutes per
trajectory and 120 seconds per tool. Retries/compaction are disabled. Failed and
partial output directories are retained and cannot be overwritten by a rerun.

Each continuation stores the exact plan and QA receipt, source provenance, native
Pi session, raw request/SSE captures, user inputs, full messages, patches, final Git
state and checkpoint archives. Model completion status remains separate from task
correctness. These captures do not automatically become the old runner's judged
analysis rows: deterministic task checks or a separately specified outcome judge
must be applied before reporting instruction-following outcomes.

Offline fixtures exercise actual Pi tool execution and continuation restoration
without a paid provider. They establish integration behavior only. The fresh
awareness-selected source cohort is a selection-conditioned set of captures;
absence of verbalized awareness does not establish absence of internal awareness.
