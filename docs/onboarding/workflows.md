# Workflows

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

## Run a matchup

Create YAML following [`../reference/configuration.md`](../reference/configuration.md), then:

```bash
export OPENROUTER_API_KEY=...
uv run reasonese-run-conversation --matchup path/to/matchup.yaml
```

For each user-authored instruction, add its exact base text to `instruction.txt` and replace the
selected framing placeholder under `prompts/user/<instruction>/`. Use `--user-messages` when the
manual hierarchy lives elsewhere.

The first run may submit authoring batches and then one synchronous assistant request. A warm
trace-cache hit makes no network call unless a selected manual variant changed, in which case the
stale trace is replaced. Cache files belong under ignored output directories and must not contain
the API key.

## Validate a change

```bash
uv run pre-commit run --all-files
```

Unit tests replace the provider transport and make no network calls.
