# Workflows

## Inspect definitions

```bash
uv run reasonese-axes
```

Use this rather than duplicating axis values in scripts.

## Plan an instruction set

Use the instruction-pair YAML bank described in [`../reference/configuration.md`](../reference/configuration.md),
then run:

```bash
uv run reasonese-plan --pairs configs/instruction_pairs.yaml --output out/specs.jsonl
```

The summary reports the number of instructions and specifications. Re-running with identical
input produces identical ordered records.

## Write the manual variants

Studies that include the `user` author need its 144 hand-written variants first. Serve the
blinded editor and work through the queue:

```bash
uv run reasonese-write-variants \
  --pairs configs/instruction_pairs.yaml \
  --user-messages prompts/user \
  --seed 0 \
  --tunnel
```

The summary reports how many variants remain and the tokenized URL to open. Collection fails
closed on any variant still holding its `TODO:` placeholder, so a half-finished set cannot be
collected by accident.

## Sample pairwise studies

Choose how many unordered pairs each assistant should receive and write one reproducible suite:

```bash
uv run reasonese-sample-studies \
  --pairs configs/instruction_pairs.yaml \
  --pairings-per-pair 720 \
  --rollouts-per-permutation 1 \
  --seed 0 \
  --output out/studies.yaml
```

The command reports each pair's exhaustive population and minimum connected sample before any
provider work occurs. Omitting `--pairings-per-pair` selects 720 pairings per pair, capped by the
eligible population and raised when more edges are needed to connect all cells. It includes all
authors and assistants unless repeated `--author` or `--assistant` filters are supplied. The
sample always covers every selected cell, is connected within each pair, and never includes a
pairing without a user-message input. Pairing quotas preserve the eligible population's
channel-pair and axis-difference strata, while candidate selection reduces degree imbalance
within each channel. Every pairing joins the two sides of one instruction pair; pairings are
never formed across pairs, because two instructions from different pairs do not conflict and
would produce a trial with no signal. Connectivity is checked afterward and minimally repaired
only when needed. A different seed changes the candidates and selected edges; the same seed and
inputs reproduce the same suite.

## Run a matchup

Create YAML following [`../reference/configuration.md`](../reference/configuration.md), then:

```bash
export OPENROUTER_API_KEY=...
uv run reasonese-run-conversation --allow-paid --matchup path/to/matchup.yaml
```

For each user-authored instruction, add its exact base text to `instruction.txt` and replace the
selected framing placeholder under `prompts/user/<instruction>/`. Use `--user-messages` when the
manual hierarchy lives elsewhere.

The first run may submit authoring batches, one Luna message-QA batch, and then the assistant
request. Any failed QA verdict stops before the assistant and remains cached for inspection; text
is never rerolled automatically. A warm trace-cache hit makes no network call unless a selected
manual variant changed or exact passing QA is absent. Cache files belong under ignored output
directories and must not contain the API key.

## Check materialized messages

The conversation runner and study collector enforce this automatically. To inspect the same gate
directly, run:

```bash
uv run reasonese-check-messages \
  --message-cache out/generated_messages.yaml \
  --qa-cache out/message_qa.yaml
```

The utility audits exact cache misses in one Luna batch, prints ordered booleans, and exits 1 when
any message fails. It preserves issue lists and raw responses without modifying the messages.

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
uv run reasonese-collect-data --allow-paid \
  --study path/to/study.yaml \
  --user-messages prompts/user \
  --output out/my-study
```

The collector runs both input orderings and every requested rollout, then emits
`observations.jsonl`. It uses shared generated-message caching, separate trace caches for each
rollout, and trace-sensitive judgment caching. Re-running an entirely cached study needs no
key once exact passing message QA is also cached. User-authored cells use the same manual
hierarchy as a single matchup; editing a selected variant invalidates every affected rollout and
its QA. Use `--no-batch` only when synchronous assistant
execution is intentionally desired; the Luna judge remains a batch model.

Check the design size before a live run: two inputs and `r` rollouts require `2r` assistant
responses and `4r` judge verdicts.

For multiple studies, keep their YAML filename stems distinct and collect them in one process:

```bash
uv run reasonese-collect-studies --allow-paid \
  --study path/to/study-a.yaml \
  --study path/to/study-b.yaml \
  --user-messages prompts/user \
  --output out/my-study-suite
```

The suite utility batches shared message QA once, runs pending assistant requests concurrently,
and judges all completed traces in one request group. Assistant inference stays synchronous so
the OpenRouter web-search server tool remains available; authoring, QA, and judging still use
compatible batch routes. Shared authoring and QA caches live at the suite root; each study's
resumable traces, judgments, and observations remain in its own subdirectory.

For a sampled suite, pass the single generated artifact instead:

```bash
uv run reasonese-collect-studies --allow-paid \
  --suite out/studies.yaml \
  --user-messages prompts/user \
  --output out/sampled-study
```

Each study is resumed below a fingerprint-named directory. One shared
`out/sampled-study/collection.sqlite3` stores all traces and judgments instead of creating a
database per child. The collector also writes all rows to `out/sampled-study/observations.jsonl`,
so analysis does not require listing every child path.

## Analyze collected observations

One or more observation files can be combined:

```bash
uv run reasonese-analyze \
  --observations out/study-a/observations.jsonl out/study-b/observations.jsonl \
  --output out/combined-analysis \
  --l2 1.0 \
  --bootstrap-samples 200 \
  --lasso-folds 5 \
  --lasso-path-length 40 \
  --seed 0
```

The feature-lasso section of `report.md` lists the contrasts that entered the fit in the order
the penalty admitted them, with their coefficients at the cross-validated one-standard-error
penalty. `--lasso-folds 0` skips cross-validation and reports the least penalized end of the
path; `--lasso-path-length` sets how many penalties are fitted between the value that zeroes
every feature and one thousandth of it. Folds are assigned by cell pair, so the two orderings
of a study never straddle a fold, and `--seed` seeds the folds as well as the bootstrap.
Cross-validation refits the path once per fold and dominates the lasso's time: about ten
seconds at the two-model pilot size and about a minute and a half with every author and
assistant selected.

Inspect `report.md` first, then `diagnostics.json`. A disconnected comparison graph means the
L2 penalty numerically places components on one list, but the data do not identify their
between-component ordering. Position imbalance or large cell/axis position effects should be
resolved or modeled before interpreting the primary ranking.

## Validate a change

```bash
uv run pre-commit run --all-files
```

Unit tests replace the provider transport and make no network calls.
