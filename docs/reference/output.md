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
