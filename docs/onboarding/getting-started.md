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
