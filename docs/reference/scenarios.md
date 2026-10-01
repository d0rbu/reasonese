# Scenario conversations

A bare conversation contains only a matchup's two inputs: a system message, a user message, or a
`README.md` read per input. A scenario places the same two inputs inside shared, longer context
so the conversation resembles a real session. Scenarios are optional. A run that does not pass
`--scenarios` builds bare conversations exactly as before, and so does any pair without a
scenario file.

## Scientific status

A scenario is a fixed property of one instruction pair, not an axis. Both inputs of a trial see
the same conversation, so the comparison inside a trial is unchanged and the rankings, margins,
and feature lasso run without modification. Two cautions apply when reading their output:

- The context around an input can differ by channel, for example a README that gains a section
  only when an input is placed in it. Channel effects from a scenario run therefore include how
  the scenario presents each channel.
- If a template frames an input differently in its early and late slot, for example embedded in
  a long system prompt when early but a standalone message when late, delivery position is
  confounded with that framing for the channel. `reasonese-show-scenario` reports such channels
  as `asymmetric_channels`; the example below and the test fixture have none.

The repository ships one scenario, `tests/fixtures/scenarios/project-tree-bash-vs-python.yaml`,
which is synthetic, exists only to exercise the template slots, and must not be used in a study.
No study scenario has been written or collected yet.

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
so a broken template fails before any provider request. The same checks run again on the real
authored text when a conversation is built.

- Each filled slot is rendered in exactly one message, of the role its channel requires, with
  the authored text verbatim and only once.
- The early slot's message comes before the late slot's message.
- Only the message that carries a slot may depend on it. Other context must not appear, vanish,
  or change with a slot; the one exception is a `README.md` read that moves with its slot.
- A conversation reads `README.md` at most once, and a tool message is the unconditional result
  of the read before it.
- No system, user, or tool message renders empty, a plain assistant turn does not either, and
  the conversation does not end on an assistant turn.

## Running with scenarios

```bash
uv run reasonese-show-scenario --scenarios scenarios --pair <pair id> \
  --first-channel "README.md" --second-channel "user message"
uv run reasonese-collect-studies --suite out/suite.yaml --output out/run --scenarios scenarios
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
- A stored trace gains a `placements` list naming the message that carries each input. Bare
  traces are stored exactly as before.
- Use a fresh output directory for each scenario selection. A run that finds traces collected
  under another selection, or under a scenario file that has since been edited, stops with an
  error instead of collecting over them.
- Observation rows do not record the layout. The output directory is what identifies a run as
  bare or scenario, so keep the two in separate directories when analyzing.

## Current limits

- The only tool context a scenario can show is one `README.md` read.
- Role probes measure an input as a whole message field. Inline probe diagnostics, probe
  rendering, and post-hoc probe scoring reject scenario conversations instead of measuring a
  partial span.
