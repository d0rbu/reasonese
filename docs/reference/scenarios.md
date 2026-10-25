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
  confounded with that framing for the channel. Inspect both delivery orders with
  `reasonese-show-scenario` when reviewing a template.

The repository ships a synthetic fixture, `tests/fixtures/scenarios/project-tree-bash-vs-python.yaml`,
which exists only to exercise the template slots and must not be used in a study, and the study
scenarios below `configs/scenarios/`, described in [Study scenarios](#study-scenarios). No
scenario has been collected yet.

## Study scenarios

`configs/scenario_pairs.yaml` is a second instruction bank whose pairs are built around real agent
rules. In every pair at least one instruction carries a rule taken from a public `AGENTS.md`; the
commit-pinned link is recorded in the scenario's `source`. `configs/scenarios/<pair id>.yaml`
wraps each pair in a short session for that repository: a system prompt, a user line, and a
`README.md` whose "Main notes for models" section is the README channel. The system instruction is
appended as its own message rather than embedded in the system prompt, so position is not
confounded with embedding. Everything that is not quoted from the source was written by hand, and
`adaptation` says for each pair whether the rule is quoted verbatim or reworded.

These pairs are kept out of `configs/instruction_pairs.yaml` because the `user` author has no
manual variants for them yet, so runs over them use model authors only (the sampler's default).
Pass the bank to both the sampler and the collector:

```bash
uv run reasonese-sample-studies --pairs configs/scenario_pairs.yaml --output out/scenario-suite.yaml
uv run reasonese-collect-studies --suite out/scenario-suite.yaml --output out/scenario-run \
  --pairs configs/scenario_pairs.yaml --scenarios configs/scenarios
```

`tests/test_scenario_bank.py` loads the bank, checks that every pair has exactly one scenario with
provenance and shares no id or text with the main bank, and renders every scenario in all five
channel orderings with its base instructions.

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
