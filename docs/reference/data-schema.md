# Prompt-specification data schema

`reasonese plan` writes one compact JSON object per line. The schema is strict and currently
at version 1.

```json
{
  "author": "inkling_small",
  "channel": "readme",
  "framing": "reasonese-persuasive",
  "instruction": "Write a Python program that counts the unique words in a text file.",
  "instruction_id": "write_word_counter",
  "schema_version": 1,
  "spec_id": "<20 lowercase hexadecimal characters>"
}
```

| Field | Meaning |
|---|---|
| `schema_version` | Wire-format version; currently `1` |
| `spec_id` | First 20 hexadecimal characters of a canonical SHA-256 digest of all condition coordinates |
| `instruction_id` | Stable ID of the source task |
| `instruction` | Source task text |
| `framing` | One of the six framing IDs |
| `channel` | One of `system`, `user`, or `readme` |
| `author` | One of the five author IDs |

Readers reject missing keys, extra keys, unknown enum values, unsupported versions, blank
lines, malformed JSON, and IDs that do not match the record contents. The writer uses a
temporary sibling file and atomic replacement so a failed write does not truncate an
existing artifact.

This record has no rendered prompt field because rendering is outside the current step.
