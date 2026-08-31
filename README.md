# reasonese

`reasonese` is a research codebase for studying how the same instruction changes across
four controlled axes: instruction, framing, channel, and author.

## Current foundation

The repository currently defines the axes and enumerates their combinations. It does not
yet generate framed text, call a model, collect responses, or analyze outcomes.

| Axis | Current values |
|---|---|
| instruction | Simple base prompts such as “Write a program…” or “Find information…” |
| framing | `normal`, `casual`, `persuasive`, `subagent`, `reasonese-normal`, `reasonese-persuasive` |
| channel | `system prompt`, `user message`, `README.md` |
| author | `user`, `Qwen3.8 Flash`, `Qwen3.8 2.4T`, `Inkling`, `Inkling Small` |

Framing and author are independent. “Normal” is the author's default rendering in clear
prose: for the user it may preserve the original instruction, while for a model it is the
model's unconstrained rewrite.

Six framings, three channels, and five authors produce `6 × 3 × 5 = 90` specifications per
base instruction. A specification is just a four-field dataclass containing those axes.

## Quickstart

```bash
git clone https://github.com/d0rbu/reasonese.git
cd reasonese
uv sync

uv run reasonese axes
uv run reasonese plan \
  --instructions configs/example_instructions.toml \
  --output out/example/prompt_specs.jsonl
```

The example contains two instructions, so it writes 180 JSONL records. Each record contains
only `instruction`, `framing`, `channel`, and `author`.

## Implementation notes

The enums use their display text directly; there is no separate label mapping. Base
instructions and non-negative counts use `phantom-types`, while `beartype` checks public
function and dataclass boundaries at runtime.

- [`docs/research/axes.md`](docs/research/axes.md) defines the research constructs.
- [`docs/reference/architecture.md`](docs/reference/architecture.md) describes the small implementation.
- [`docs/reference/output.md`](docs/reference/output.md) shows the JSONL output.

## Development

```bash
uv run pre-commit run --all-files
```

## License

MIT. See [`LICENSE`](LICENSE).
