# AGENTS.md - reasonese

`reasonese` is a correctness-first research codebase for controlled studies of instruction
arbitration. Keep scientific status, data provenance, and construct boundaries explicit.

## Read these first

- [`README.md`](README.md) - project question, status, and quickstart
- [`docs/research/agenda.md`](docs/research/agenda.md) - hypotheses, phases, and non-goals
- [`docs/research/protocol.md`](docs/research/protocol.md) - pilot design and analysis contract
- [`docs/research/construct-validity.md`](docs/research/construct-validity.md) - claim boundaries
- [`docs/reference/architecture.md`](docs/reference/architecture.md) - package architecture
- [`docs/reference/data-schema.md`](docs/reference/data-schema.md) - artifact schemas
- [`docs/reference/file-reference.md`](docs/reference/file-reference.md) - file map

## Scientific conventions

- Never call synthetic responses or rankings empirical model results.
- Call `zz_compact` *incident-inspired compressed text* or *neuralese-like*, not a sample of
  hidden reasoning or a model-native language.
- Preserve all four counterbalances for every condition pair and code pair.
- Keep invalid responses in the scored artifact and report their rate separately.
- Treat the Bradley-Terry ranking as descriptive and conditional on exact valid outputs.
- Record exact model identifiers, endpoint dates, decoding parameters, prompt hashes, and
  raw responses before making cross-model or longitudinal claims.
- Do not add or invoke paid/live providers without explicit authorization.
- Keep any future agentic study sandboxed and benign; simulate costs rather than creating
  destructive or unauthorized tasks.

## Engineering conventions

- Use `uv sync` to install and `uv run ...` to invoke tools.
- Keep raw boundary records strict and versioned; do not ignore unknown JSON or TOML keys.
- Write generated artifacts below an ignored output directory such as `out/`.
- Prefer typed dataclasses and explicit validation over loosely structured dictionaries.
- Use `phantom-types` for bounded primitives, NumPy for numerical routines, and Hypothesis
  for invariants that benefit from generated cases.
- Update docs and `docs/reference/file-reference.md` when behavior or layout changes.

## Required gate

```bash
uv run pre-commit run --all-files
```

If a live study is not run, say so plainly; passing the offline gate does not validate model
behavior or the experimental construct.
