# AGENTS.md - reasonese

`reasonese` is a correctness-first research codebase for controlled prompt-authoring
experiments. Keep implementation status and construct boundaries explicit.

## Read these first

- [`README.md`](README.md) - project status and quickstart
- [`docs/research/axes.md`](docs/research/axes.md) - authoritative axis definitions
- [`docs/research/agenda.md`](docs/research/agenda.md) - current scope and deferred work
- [`docs/reference/data-schema.md`](docs/reference/data-schema.md) - strict artifact contracts
- [`docs/reference/file-reference.md`](docs/reference/file-reference.md) - repository map

## Scientific conventions

- Keep instruction, framing, channel, and author as independent axes.
- Treat an instruction as author-independent task content, not as one specific rendering.
- Use the exact six framing IDs and distinguish normal from persuasive intent and natural
  language from reasonese representation.
- Use `author` for whoever wrote a framed instruction. Do not use it for the model that
  later receives or executes the instruction.
- Treat reasonese as an operational prompt representation to be specified and validated.
  Do not equate it with hidden reasoning or claim it is a model-native language.
- Do not call prompt specifications generated prompts, model responses, or research results.
- Do not add or invoke live or paid model providers without explicit authorization.

## Engineering conventions

- Use `uv sync` to install and `uv run ...` to invoke tools.
- Keep boundary records strict, deterministic, and schema-versioned.
- Reject unknown TOML and JSON keys rather than silently accepting drift.
- Write generated artifacts below an ignored output directory such as `out/`.
- Prefer typed dataclasses and explicit validation over loosely structured dictionaries.
- Update docs and `docs/reference/file-reference.md` when behavior or layout changes.

## Required gate

```bash
uv run pre-commit run --all-files
```

Passing the offline gate validates the software foundation only. It does not validate
future prompt transformations or model behavior.
