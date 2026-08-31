# Getting Started

## Prerequisites

- Python 3.13
- [`uv`](https://docs.astral.sh/uv/)

```bash
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

## Run the offline demo

```bash
uv run reasonese design \
  --config configs/pilot.toml \
  --output out/demo/design.jsonl
uv run reasonese simulate \
  --design out/demo/design.jsonl \
  --config configs/synthetic_demo.toml \
  --output out/demo/responses.jsonl
uv run reasonese score \
  --design out/demo/design.jsonl \
  --responses out/demo/responses.jsonl \
  --output out/demo/outcomes.jsonl
uv run reasonese fit \
  --outcomes out/demo/outcomes.jsonl \
  --reference plain \
  --output out/demo/ranking.json
```

Everything under `out/demo/` is generated and ignored by git. The simulator samples from
the known strengths in `configs/synthetic_demo.toml`; its ranking is not a model result.

## Development setup

```bash
uv run pre-commit install
uv run pytest
uv run ty check
uv run ruff check .
```

Before changing treatment text, read the pilot protocol. Once live collection begins,
create a new versioned config rather than editing the run's config in place.
