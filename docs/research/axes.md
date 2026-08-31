# The four axes

The current design describes a complete cell with four independent coordinates.

## Instruction

An instruction is the author-independent semantic task. It should be a simple base request,
such as writing a program that performs a specified function or finding information about a
specified topic. It has a stable snake-case ID and trimmed text.

The stored text is the source task, not necessarily the exact text later shown to an
executor. Framing is the transformation of that task.

## Framing

| ID | Operational intent |
|---|---|
| `normal` | The author's default rendering in clear, standard prose. A user-authored normal prompt may be the original task text; a model-authored one is the model's unconstrained rewrite. |
| `casual` | Conversational wording, lowercase text, and reduced punctuation. |
| `persuasive` | Natural language deliberately intended to secure compliance, including social or multi-agent persuasion where appropriate. |
| `subagent` | A delegation written as though a parent agent were passing the task to a subagent. |
| `reasonese-normal` | A compressed reasonese representation without deliberate persuasive intent. |
| `reasonese-persuasive` | A compressed reasonese representation deliberately intended to secure compliance. |

These definitions specify intended treatments, not implemented transformation prompts.
Future prompt-generation work must make each treatment reproducible and test whether its
outputs match the intended construct.

## Channel

| ID | Delivery context |
|---|---|
| `system` | The framed instruction appears in the executor's system prompt. |
| `user` | The framed instruction appears in a user message. |
| `readme` | The framed instruction appears in `README.md`, and the executor is told to read it. |

The current planner records the channel only. It does not render an API conversation or
filesystem environment.

## Author

| ID | Display label |
|---|---|
| `user` | user |
| `qwen3_8_flash` | Qwen3.8 Flash |
| `qwen3_8_2_4t` | Qwen3.8 2.4T |
| `inkling` | Inkling |
| `inkling_small` | Inkling Small |

Author means the person or model that wrote the framed instruction. It is not the executor
model that later receives the prompt. Separating author from framing allows any author to
produce any framing, including unusual combinations that may require special generation
procedures later.

## Design size

For each base instruction, the planner enumerates every framing, channel, and author once:

```text
6 framings × 3 channels × 5 authors = 90 prompt specifications
```

No comparison, outcome, or model-behavior claim is encoded in those specifications.
