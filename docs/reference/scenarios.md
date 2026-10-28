# Scenario conversations

A bare conversation contains only a matchup's two inputs: a system message, a user message, or a
`README.md` read per input. A scenario places the same two inputs inside shared, longer context
so the conversation resembles a real session. Scenarios are optional. A run that does not pass
`--scenarios` builds bare conversations exactly as before, and so does any pair without a
scenario file.

## Scientific status

The twelve current pairs in `configs/scenario_pairs.yaml` now use
[repository-native Pi tasks](pi-transcripts.md), including the reviewed
[JSON parser pilot](../research/langchain-json-recovery.md) and
[eleven redesigned pilots](../research/remaining-scenario-pilots.md). Their old
synthetic README histories are retired. `configs/scenarios/` contains a retirement
notice and no active YAML templates; passing it to `--scenarios` is an error.
Native Pi histories are not silently converted to this restricted template format.

This page documents the optional template mechanism for separately reviewed future
definitions. A scenario is a fixed property of a pair, not an axis. Both inputs see
the same conversation; analysis still uses the ordinary trial representation.
Channel effects include how a template presents each channel, and differing early
and late embeddings can confound delivery position with framing. Inspect all five
channel orderings with `reasonese-show-scenario` before collecting comparisons.

`tests/fixtures/scenarios/project-tree-bash-vs-python.yaml` is an explicitly
synthetic renderer fixture, not study material. `tests/test_scenarios.py` retains
coverage of slots, all channel orderings, placement and cache validity.
`tests/test_scenario_bank.py` checks the native catalog's pair coverage, pinned
provenance, main-bank disjointness and preservation of the reviewed first pilot.

## Current task bank

The twelve base pairs express authored behavior requirements on inspected upstream
code. They are not quotations from upstream agent rules. Each redesigned task has
two incompatible outcomes for the same concrete case and substantial shared
implementation work. Framing, channel and author remain separate from those base
instructions. The native manifest records the intended delivery and operator review
checks; it does not certify treatment labels or model behavior.

These pairs remain separate from `configs/instruction_pairs.yaml`; no manual user
author variants or controlled transcript-prefix comparisons have been prepared for
them. Use the [Pi preparation and collection commands](pi-transcripts.md) for source
sessions. The generic collector without a template falls back to bare instructions,
which does not create a native multi-turn experiment.

## File format

A scenario is `<pair id>.yaml` in the selected directory, named after a pair in the instruction
bank:

```yaml
source: where the real case came from
adaptation: what was changed from the source
messages:
  - role: system
    content: You are a coding agent working in a repository.
  - role: system
    when: system.early
    content: "{{ system.early }}"
  - role: user
    content: Read the README before we start.
  - role: user
    when: user.early
    content: "{{ user.early }}"
  - role: assistant
    reads: README.md
  - role: tool
    content: |
      # Project
      {% if tool.early or tool.late %}

      ## Notes for agents

      {{ tool.early }}{{ tool.late }}
      {% endif %}
  - role: system
    when: system.late
    content: "{{ system.late }}"
  - role: user
    when: user.late
    content: "{{ user.late }}"
```

`source` and `adaptation` are required so every scenario's provenance can be audited and the
text replaced later. Each message has a `role` and Jinja `content`. `when` is a Jinja expression;
the message is dropped when it is false. An assistant message may carry `reads: README.md`, which
becomes a `read_file` call, and the `tool` message after it is that file's text. A plain
assistant message with only `content` is an earlier assistant turn.

## Slots

Templates see three slot groups, `system`, `user`, and `tool`, each with an `early` and a `late`
value. The first matchup input fills the early slot of its channel and the second fills the late
slot of its channel. The other four slots are empty strings. Authored text is passed as a value
and is never re-rendered, so braces inside it stay literal.

Because position follows the slot, the two delivery orders of a cell pair stay distinct and an
observation's `position` is still the delivery order. A scenario must support all five channel
orderings a matchup can take: system then user, user then system, user then user, user then
README, and README then user.

## Rules

Every scenario is rendered for all five orderings with two-paragraph probe text when it loads,
catching structural errors before any provider request. The checks run again on the real
authored text when a conversation is built, including occurrence counts for content-dependent
template branches that the probes do not exercise.

Templates are experiment definitions to review with the offline preview. These structural
checks do not prove that arbitrary Jinja conditions preserve the experimental design.

- Each filled slot is rendered in exactly one message, of the role its channel requires, with
  the authored text verbatim and only once.
- The early slot's message comes before the late slot's message.
- Only the message that carries a slot may depend on it. Other context must not appear, vanish,
  or change with a slot. A `README.md` read and its result stay at a fixed point in the template;
  the read cannot depend on which slots are filled.
- A conversation reads `README.md` at most once, and a tool message is the unconditional result
  of the read before it.
- No system, user, or tool message renders empty, a plain assistant turn does not either, and
  the conversation does not end on an assistant turn.

## Running with scenarios

```bash
uv run reasonese-show-scenario --scenarios scenarios --pair <pair id> \
  --first-channel "README.md" --second-channel "user message"
uv run reasonese-collect-studies --allow-paid --suite out/suite.yaml --output out/run --scenarios scenarios
```

`reasonese-show-scenario` is offline. It renders one pair's scenario with the base instructions
standing in for authored text, to read the conversation before spending on it.
`reasonese-collect-data`, `reasonese-collect-studies`, and `reasonese-run-conversation` accept
`--scenarios <directory>` and `--pairs <bank>` (default `configs/instruction_pairs.yaml`). The
directory must hold at least one scenario, and every study instruction must belong to the bank;
both are checked before any authoring request.

## What changes downstream

- Authoring and message QA are unchanged. The same authored text is used in both layouts, so
  a bare run and a scenario run of one cell differ only in the surrounding context.
- The sandbox `README.md` is the scenario's whole rendered README, whether or not an input was
  placed in it.
- A stored trace gains a `placements` list naming the message and exact authored text for each
  input. Judgment fingerprints include those placements, so changing an authored span invalidates
  the cached verdict even if the conversation text is unchanged. Bare traces and their
  fingerprints are unchanged.
- Use a fresh output directory for each scenario selection, including bare mode. On resume,
  each requested cached trial must match the conversation its selected template would render;
  a mismatch stops collection instead of overwriting that trace. Collection does not audit
  other studies in the directory or coordinate separate collectors' output directories.
- Observation rows do not record the layout. The output directory is what identifies a run as
  bare or scenario, so keep the two in separate directories when analyzing.

## Current limits

- The only tool context a scenario can show is one `README.md` read.
- Role probes measure an input as a whole message field. Inline probe diagnostics, probe
  rendering, and post-hoc probe scoring reject scenario conversations instead of measuring a
  partial span.
