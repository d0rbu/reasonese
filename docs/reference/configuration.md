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

The list must contain exactly two entries and at least one entry whose channel is
`user message`. It may repeat any channel. List order is the
conversation order. See [`../../configs/example_matchup.yaml`](../../configs/example_matchup.yaml).

## Manual user-author configuration

An entry with `author: user` resolves its text from the `--user-messages` root used by both the
conversation runner and study collector; it defaults to `prompts/user`. Create one child
directory per instruction. Its `instruction.txt` must exactly match the YAML instruction, and it
must contain `normal.txt`, `casual.txt`, and `persuasive.txt`, the three framings the `user`
author writes. Directory names are descriptive only. A
selected file beginning with `TODO:` is an intentional placeholder and is rejected before
inference.

## Study configuration

A study adds a positive rollout count to an assistant and an unordered pair of inputs:

```yaml
assistant: Qwen3.8 Flash
rollouts_per_permutation: 2
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

The two inputs must be distinct, and at least one must use the explicit `user message` channel.
The collector enumerates both orderings and repeats each
ordering `rollouts_per_permutation` times. See
[`../../configs/example_study.yaml`](../../configs/example_study.yaml).

## Instruction pair configuration

The instruction bank is a YAML list of pairs:

```yaml
pairs:
  - id: leap-years-count-vs-list
    skill: python
    conflict: output format
    first: Using Python, count how many leap years fall between 1900 and 2100 inclusive. Reply with only the count.
    second: Using Python, list every leap year between 1900 and 2100 inclusive, one per line, and do not state a total count anywhere in your reply.
    rationale: The first allows only a count and the second forbids stating a count.
```

`id` is a lowercase hyphen-separated identifier and must be unique across the bank. `skill` is one
of `python`, `bash`, `web search`, or `python and web search`. `conflict` is one of the conflict
types in [`../research/instruction-bank.md`](../research/instruction-bank.md). `first` and
`second` are exact base instructions and must differ. See
[`../../configs/instruction_pairs.yaml`](../../configs/instruction_pairs.yaml).

Pass multiple study files as repeated `--study` arguments to `reasonese-collect-studies` when
collecting a larger design. Their filename stems must be distinct because each stem names that
study's output subdirectory.

## Sampled study suite

`reasonese-sample-studies` builds all four-axis cells from an instruction TOML file, selects a
seeded axis-stratified, degree-aware, connected subset of valid unordered pairs, and writes them
under one `studies` key:

```yaml
studies:
  - assistant: Qwen3.8 Flash
    rollouts_per_permutation: 1
    inputs:
      - instruction: Write a program.
        framing: normal
        channel: system prompt
        author: Inkling
      - instruction: Find the requested information.
        framing: casual
        channel: user message
        author: Inkling Small
```

Every study is validated by the ordinary two-input contract. Suite entries must be distinct.
The same selected input pairs appear for every requested assistant, enabling assistant-specific
analyses over matched comparison designs. Use `reasonese-collect-studies --suite PATH` to collect
the suite; fingerprint-named output directories avoid dependence on thousands of filenames.
