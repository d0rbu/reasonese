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

## Judge a response

Point the judge utility at the same matchup and trace cache:

```bash
export OPENROUTER_API_KEY=...
uv run reasonese-judge-responses \
  --matchup path/to/matchup.yaml \
  --trace-cache out/conversation_traces.yaml \
  --judgment-cache out/judgments.yaml
```

The utility requires an existing matching trace. It batches one independent medium-reasoning
GPT-5.6 Luna request per input, prints the ordered boolean list, and caches the raw judge
responses. Repeating an unchanged trace is network-free.

## Collect a balanced study

Create study YAML following [`../reference/configuration.md`](../reference/configuration.md):

```bash
export OPENROUTER_API_KEY=...
uv run reasonese-collect-data \
  --study path/to/study.yaml \
  --user-messages prompts/user \
  --output out/my-study
```

The collector runs both input orderings and every requested rollout, then emits
`observations.jsonl`. It uses shared generated-message caching, separate trace caches for each
rollout, and trace-sensitive judgment caching. Re-running an entirely cached study needs no
key. User-authored cells use the same manual hierarchy as a single matchup; editing a selected
variant invalidates every affected rollout. Use `--no-batch` only when synchronous assistant
execution is intentionally desired; the Luna judge remains a batch model.

Check the design size before a live run: two inputs and `r` rollouts require `2r` assistant
responses and `4r` judge verdicts.

## Analyze collected observations

One or more observation files can be combined:

```bash
uv run reasonese-analyze \
  --observations out/study-a/observations.jsonl out/study-b/observations.jsonl \
  --output out/combined-analysis \
  --l2 1.0 \
  --bootstrap-samples 200 \
  --seed 0
```

Inspect `report.md` first, then `diagnostics.json`. A disconnected comparison graph means the
L2 penalty numerically places components on one list, but the data do not identify their
between-component ordering. Position imbalance or large cell/axis position effects should be
resolved or modeled before interpreting the primary ranking.

## Validate a change

```bash
uv run pre-commit run --all-files
```

Unit tests replace the provider transport and make no network calls.
