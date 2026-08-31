# reasonese

`reasonese` studies which instruction an AI agent follows when two directives conflict
and differ only in representation, tone, claimed authority, or social endorsement.

The motivating observation comes from the 2026 OpenAI/Hugging Face incident: agents
communicated with compressed `zz...` messages, and some accepted peer requests that put
their own task success at risk. The first experiment isolates a smaller question before
attempting a naturalistic agent study:

> How does an incident-inspired compressed directive rank against ordinary English,
> symbolic syntax, tone, authority, expertise, human consensus, and agent consensus?

The term *reasonese* is a project label. The current `zz_compact` treatment is observable,
hand-authored text inspired by public incident examples. It is **not** assumed to be a
faithful sample of a model's private chain of thought or a privileged "native language."

## Current status

The repository contains a tested, fully offline pilot pipeline:

1. generate all pairwise condition conflicts;
2. counterbalance response-code assignment and directive order;
3. validate the pipeline with clearly labeled synthetic responses;
4. score only exact response-code matches; and
5. fit an L2-penalized Bradley-Terry ranking.

There are **no empirical model results yet**. No live or paid model endpoint was called in
the initial build. Files produced by `configs/synthetic_demo.toml` are software test
artifacts, not research findings.

## Quickstart

```bash
git clone https://github.com/d0rbu/reasonese.git
cd reasonese
uv sync

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

The pilot has 14 conditions, two nonce-code pairs, two target assignments, and two prompt
orders: `14 choose 2 * 2 * 2 * 2 = 728` trials per repetition and model endpoint.

## Why this design

Each trial puts both directives in the same user message and asks for one of two nonce
codes. For every condition pair, the code attached to each condition is swapped and the
display order is reversed. This removes the simplest response-code and recency confounds.
Invalid or explanatory responses remain in the outcome data and are reported separately;
they are never converted into wins.

The primary estimand is a condition's probability of winning a pairwise conflict,
conditional on an exact valid response under a fixed model and endpoint configuration. The
ranking is descriptive: it does not by itself establish a psychological mechanism, a stable
model trait, or behavior in a long-horizon agent environment.

## Research map

| Question | Document |
|---|---|
| Hypotheses, phases, and non-goals | [`docs/research/agenda.md`](docs/research/agenda.md) |
| Frozen pilot design and analysis contract | [`docs/research/protocol.md`](docs/research/protocol.md) |
| What “neuralese-like” does and does not mean | [`docs/research/construct-validity.md`](docs/research/construct-validity.md) |
| Motivating and adjacent literature | [`docs/research/related-work.md`](docs/research/related-work.md) |
| JSONL record contracts | [`docs/reference/data-schema.md`](docs/reference/data-schema.md) |
| Package and pipeline architecture | [`docs/reference/architecture.md`](docs/reference/architecture.md) |

## Development

```bash
uv run pre-commit run --all-files
```

The gate checks the lockfile, Ruff, ty, and pytest. The current suite exercises schemas,
counterbalancing, deterministic simulation, exact scoring, atomic JSONL writes, the CLI,
and Bradley-Terry failure modes.

## License

MIT. See [`LICENSE`](LICENSE).
