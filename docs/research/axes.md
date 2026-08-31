# The four axes

Each planned condition has four independent coordinates.

## Instruction

An instruction is a simple base prompt, such as asking for a program that does something or
requesting information about a topic. It is stored directly as a non-empty, trimmed string.
Framing later changes how that prompt is expressed.

## Framing

| Value | Intended treatment |
|---|---|
| `normal` | The author's default clear rendering. For the user this may be the original prompt; for a model it is an unconstrained rewrite. |
| `casual` | Conversational wording, lowercase text, and reduced punctuation. |
| `persuasive` | Natural language deliberately intended to secure compliance. |
| `subagent` | A delegation written as though a parent agent were instructing a subagent. |
| `reasonese-normal` | A compressed reasonese representation without deliberate persuasive intent. |
| `reasonese-persuasive` | A compressed reasonese representation deliberately intended to secure compliance. |

These values name intended treatments. This step does not implement the transformations.

## Channel

- `system prompt`: place the framed instruction in the executor's system prompt.
- `user message`: place it in a user message.
- `README.md`: place it in `README.md` and tell the executor to read the file.

The planner records the channel but does not yet render the corresponding environment.

## Author

- `user`
- `Qwen3.8 Flash`
- `Qwen3.8 2.4T`
- `Inkling`
- `Inkling Small`

The enum strings above are the display values; there is no second label or identifier map.
Author means whoever writes the framed instruction, not the executor model.

## Design size

```text
6 framings × 3 channels × 5 authors = 90 specifications per instruction
```

No outcome or model-behavior claim is encoded in a specification.
