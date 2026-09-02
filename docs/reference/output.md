# Plan output

`reasonese-plan` writes one JSON object per line with exactly the four experimental axes:

```json
{
  "author": "Inkling Small",
  "channel": "README.md",
  "framing": "reasonese-persuasive",
  "instruction": "Write a Python program that counts the unique words in a text file."
}
```

These are unrendered condition specifications, not generated prompts or model outputs.

## Generated-message cache

The message cache is YAML under a top-level `messages` list. Each record contains the full
four-axis `input`, its generated `content`, and the unmodified OpenRouter `response`. A
user-authored message has a null response because its content comes from the selected manual
variant file. Records are keyed by all four input coordinates. Manual files remain authoritative;
changed manual text replaces its cached message and invalidates traces containing the old text.

## Conversation-trace cache

The trace cache is YAML under a top-level `traces` list. Each record contains the complete
`matchup`, the materialized ordered `conversation`, zero or more `tool_steps`, and the unmodified
final assistant `response`. Each tool step retains the raw assistant response that requested the
function call plus every local tool result sent back. This intentionally retains returned
`reasoning`, `reasoning_details`, token usage, web citations, and other provider fields. Records
are keyed by the complete matchup, including assistant and input order.

## Judgment cache

The judgment cache is YAML under a top-level `judgments` list. Each record contains:

- the complete `matchup`;
- a `trace_fingerprint` over the matchup, delivered conversation, local tool steps, and final
  assistant response; and
- one ordered verdict per input, with the four-axis input, exact `completed` boolean, and
  unmodified raw judge response.

The fingerprint prevents reuse after a trace changes. The verdicts are independent rather
than one-hot: `[true, true]`, `[false, false]`, and mixed outcomes are all valid.

## Collected study

`reasonese-collect-data --output DIRECTORY` writes:

- `study.yaml`: the exact assistant, rollout count, and input cells;
- `generated_messages.yaml`: shared materialized instruction cache;
- `trials/TRIAL_ID/trace.yaml`: one raw conversation trace per input ordering and rollout;
- `judgments.yaml`: raw and parsed judgments keyed by concrete trace; and
- `observations.jsonl`: one flat row per cell verdict.

Each observation contains the cell ID and five coordinates, permutation, rollout, one-based
position, completion boolean, trace fingerprint, and available assistant/judge response IDs.
This is the input to downstream analysis. Cached study traces are reused only while their
user-authored contents still match the selected manual files.
