# Instruction configuration

The input is a TOML array of base-prompt strings:

```toml
instructions = [
    "Write a Python program that counts the unique words in a text file.",
    "Find information about pathlib.Path.glob.",
]
```

At least one instruction is required. Each string must be non-empty and have no surrounding
whitespace. See [`../../configs/example_instructions.toml`](../../configs/example_instructions.toml).

## Matchup configuration

A matchup selects one assistant and an ordered list of entry datapoints:

```yaml
assistant: Qwen3.8 Flash
inputs:
  - instruction: Solve the task called foobar and explain the result briefly.
    framing: normal
    channel: system prompt
    author: Inkling
  - instruction: What is two plus two? Answer with the number only.
    framing: casual
    channel: user message
    author: Inkling Small
```

The list must contain at least two entries and at least one entry whose channel is
`user message`. It may contain more entries and may repeat any channel. List order is the
conversation order. See [`../../configs/example_matchup.yaml`](../../configs/example_matchup.yaml).
