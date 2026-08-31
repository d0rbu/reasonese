# Getting started

Install Python 3.13 and [`uv`](https://docs.astral.sh/uv/), then run:

```bash
uv sync
uv run reasonese-axes
uv run reasonese-plan \
  --instructions configs/example_instructions.toml \
  --output out/example/prompt_specs.jsonl
```

The first utility prints the canonical definitions. The second validates the two example
base instructions and writes 180 prompt specifications: 90 for each instruction.

Inspect a few records with:

```bash
head -n 3 out/example/prompt_specs.jsonl
```

These records are an experimental plan, not generated prompts or model outputs.

To execute the example matchup, provide the API key through the environment:

```bash
export OPENROUTER_API_KEY=...
uv run reasonese-run-conversation --matchup configs/example_matchup.yaml
```

The default readable caches are `out/generated_messages.yaml` and
`out/conversation_traces.yaml`. Repeating the same matchup returns the cached trace without
requiring the key. Use `--no-batch` to disable batch authoring where supported.
