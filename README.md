# reasonese

`reasonese` is a research codebase for studying how the same instruction changes across
four controlled axes: instruction, framing, channel, and author.

## Current foundation

The repository currently defines and validates the experimental design. It does not yet
generate framed text, call a model, collect responses, or analyze outcomes.

The four axes are:

| Axis | Meaning | Current values |
|---|---|---|
| instruction | The author-independent base task | Configured task text, such as “Write a program…” or “Find information…” |
| framing | The style or representation of the task | `normal`, `casual`, `persuasive`, `subagent`, `reasonese-normal`, `reasonese-persuasive` |
| channel | Where the instruction is presented | system prompt, user message, `README.md` |
| author | Who wrote the framed instruction | user, Qwen3.8 Flash, Qwen3.8 2.4T, Inkling, Inkling Small |

Framing and author are separate. For example, persuasive prose may be written by the user
or by any listed model. “Normal” is the author's default rendering in clear standard prose:
for the user it may preserve the original instruction, while for a model it is the model's
unconstrained rewrite.

With six framings, three channels, and five authors, each base instruction expands to
`6 × 3 × 5 = 90` unrendered prompt specifications. These specifications identify intended
conditions; they do not claim that the corresponding prompt text has already been created.

## Quickstart

```bash
git clone https://github.com/d0rbu/reasonese.git
cd reasonese
uv sync

# Inspect the exact axis definitions.
uv run reasonese axes

# Enumerate all 90 cells for each example instruction.
uv run reasonese plan \
  --instructions configs/example_instructions.toml \
  --output out/example/prompt_specs.jsonl
```

The example contains two instructions, so the command writes 180 deterministic,
versioned JSONL records.

## Research and implementation notes

- [`docs/research/axes.md`](docs/research/axes.md) defines the constructs and their boundaries.
- [`docs/research/agenda.md`](docs/research/agenda.md) states what this foundation does and does not cover.
- [`docs/reference/data-schema.md`](docs/reference/data-schema.md) specifies the TOML and JSONL contracts.
- [`docs/reference/architecture.md`](docs/reference/architecture.md) maps the implementation.

## Development

```bash
uv run pre-commit run --all-files
```

This checks the lockfile, formatting and lint rules, static types, and tests.

## License

MIT. See [`LICENSE`](LICENSE).
