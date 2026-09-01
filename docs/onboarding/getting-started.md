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

If the matchup contains `author: user`, first replace the selected `TODO:` variant under
`prompts/user`, or pass another hierarchy with `--user-messages`.

The default readable caches are `out/generated_messages.yaml`, `out/message_qa.yaml`, and
`out/conversation_traces.yaml`. Before the assistant runs, Luna batch audits each exact
materialized message against its datapoint instructions. Repeating the same matchup returns the
cached trace without requiring the key once those exact messages have passing cached QA and its
manual files have not changed. Use `--no-batch` to disable batch authoring where supported.

Inspect or rerun the message audit separately with:

```bash
uv run reasonese-check-messages
```

It exits nonzero if any exact cached message is noncompliant and never regenerates text.

Once the trace exists, judge every input with:

```bash
uv run reasonese-judge-responses --matchup configs/example_matchup.yaml
```

This uses `openai/gpt-5.6-luna:batch` at medium reasoning and writes
`out/judgments.yaml`. A warm judgment-cache hit also needs no API key.

For a balanced dataset rather than one matchup, run:

```bash
uv run reasonese-collect-data \
  --study configs/example_study.yaml \
  --user-messages prompts/user \
  --output out/example-study
```

The example's two cells produce two ordering trials and four observation rows. Increase
`rollouts_per_permutation` in the study YAML for repeated assistant responses.

Analyze the collected rows with:

```bash
uv run reasonese-analyze \
  --observations out/example-study/observations.jsonl \
  --output out/example-study/analysis
```

The default uses L2 penalty 1.0, 200 trial-cluster bootstrap samples, and seed 0. The primary
entry point is `analysis/report.md`; all complete tables remain available as CSV.
