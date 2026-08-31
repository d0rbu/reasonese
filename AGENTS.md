# AGENTS.md - reasonese

`reasonese` is a small research codebase for controlled prompt-authoring experiments. Keep
the implementation simple and the scientific status explicit.

## Read these first

- [`README.md`](README.md) - project status and quickstart
- [`docs/research/axes.md`](docs/research/axes.md) - authoritative axis definitions
- [`docs/research/agenda.md`](docs/research/agenda.md) - current scope and deferred work
- [`docs/reference/architecture.md`](docs/reference/architecture.md) - implementation shape
- [`docs/reference/file-reference.md`](docs/reference/file-reference.md) - repository map

## Scientific conventions

- Keep instruction, framing, channel, and author as independent axes.
- Treat an instruction as the simple base prompt, before framing.
- Use `author` for whoever writes a framed instruction, not the model that later executes it.
- Treat reasonese as an operational prompt representation, not hidden reasoning or a
  model-native language.
- Do not call specifications generated prompts, model responses, or research results.
- Do not add or invoke live or paid model providers without explicit authorization.

## Engineering conventions

- Prefer direct values and small dataclasses over identifiers, registries, or versioning.
- Use `phantom-types` for meaningful unary constraints and `beartype` at runtime boundaries.
- Keep relational checks explicit when they cannot be expressed by a single value's type.
- Write generated artifacts below an ignored directory such as `out/`.
- Update documentation when commands, axis values, or file responsibilities change.

## Required gate

```bash
uv run pre-commit run --all-files
```

Passing the offline gate validates the code only, not future prompt transformations or
model behavior.
