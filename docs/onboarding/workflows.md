# Offline workflows

## Inspect definitions

```bash
uv run reasonese-axes
```

Use this rather than duplicating axis values in scripts.

## Plan an instruction set

Create a TOML file following [`../reference/configuration.md`](../reference/configuration.md),
then run:

```bash
uv run reasonese-plan --instructions path/to/instructions.toml --output out/specs.jsonl
```

The summary reports the number of instructions and specifications. Re-running with identical
input produces identical ordered records.

## Validate a change

```bash
uv run pre-commit run --all-files
```

Do not add model calls to these workflows without separate authorization and a documented
artifact contract.
