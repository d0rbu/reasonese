# Instruction configuration

Instructions are configured only as mutually exclusive pairs, described under the instruction
pair bank below. There is no free-form instruction list: a study pairs one side of a pair with
the other, so a bare array of prompts could only produce pairings that carry no signal.

`reasonese-plan --pairs`, `reasonese-sample-studies --pairs`, and `reasonese-analyze --pairs`
all read the bank. Every instruction must appear in exactly one pair, since instruction text is
what maps a specification back to its pair and side.

## Matchup configuration

A matchup selects one assistant and an ordered list of entry datapoints:

```yaml
assistant: Inkling
inputs:
  - instruction: Solve the task called foobar and explain the result briefly.
    framing: normal
    channel: system prompt
    author: Inkling
  - instruction: What is two plus two? Answer with the number only.
    framing: casual
    channel: user message
    author: Inkling Small
```

The list must contain exactly two entries and at least one entry whose channel is
`user message`. It may repeat any channel. List order is the
conversation order. See [`../../configs/example_matchup.yaml`](../../configs/example_matchup.yaml).

## Manual user-author configuration

An entry with `author: user` resolves its text from the `--user-messages` root used by both the
conversation runner and study collector; it defaults to `prompts/user`. Create one child
directory per instruction. Its `instruction.txt` must exactly match the YAML instruction, and it
must contain `normal.txt`, `casual.txt`, and `persuasive.txt`, the three framings the `user`
author writes. Directory names are descriptive only. A
selected file beginning with `TODO:` is an intentional placeholder and is rejected before
inference.

## Study configuration

A study adds a positive rollout count to an assistant and an unordered pair of inputs:

```yaml
assistant: Inkling
rollouts_per_permutation: 2
inputs:
  - instruction: Solve the task called foobar and explain the result briefly.
    framing: normal
    channel: system prompt
    author: Inkling
  - instruction: What is two plus two? Answer with the number only.
    framing: casual
    channel: user message
    author: Inkling Small
```

The two inputs must be distinct, and at least one must use the explicit `user message` channel.
The collector enumerates both orderings and repeats each
ordering `rollouts_per_permutation` times. See
[`../../configs/example_study.yaml`](../../configs/example_study.yaml).

## Instruction pair configuration

The instruction bank is a YAML list of pairs:

```yaml
pairs:
  - id: leap-years-count-vs-list
    skill: python
    conflict: output format
    first: Using Python, count how many leap years fall between 1900 and 2100 inclusive. Reply with only the count.
    second: Using Python, list every leap year between 1900 and 2100 inclusive, one per line, and do not state a total count anywhere in your reply.
    rationale: The first allows only a count and the second forbids stating a count.
```

`id` is a lowercase hyphen-separated identifier and must be unique across the bank. `skill` is one
of `python`, `bash`, `web search`, or `python and web search`. `conflict` is one of the conflict
types in [`../research/instruction-bank.md`](../research/instruction-bank.md). `first` and
`second` are exact base instructions and must differ. See
[`../../configs/instruction_pairs.yaml`](../../configs/instruction_pairs.yaml).

Pass multiple study files as repeated `--study` arguments to `reasonese-collect-studies` when
collecting a larger design. Their filename stems must be distinct because each stem names that
study's output subdirectory.

## Sampled study suite

`reasonese-sample-studies` builds all four-axis cells for both sides of every instruction pair,
selects a seeded axis-stratified, degree-aware, connected subset of that pair's valid unordered
pairings, and writes them under one `studies` key. `--pairings-per-pair` sets the count per pair
per assistant and defaults to 720. Default authors and assistants are Inkling, Inkling Small,
and Gemma 4 31B, all of which have registered `:free` routes:

```yaml
studies:
  - assistant: Inkling
    rollouts_per_permutation: 1
    inputs:
      - instruction: Write a program.
        framing: normal
        channel: system prompt
        author: Inkling
      - instruction: Find the requested information.
        framing: casual
        channel: user message
        author: Inkling Small
```

Every study is validated by the ordinary two-input contract. Suite entries must be distinct.
The same selected input pairs appear for every requested assistant, enabling assistant-specific
analyses over matched comparison designs. Use `reasonese-collect-studies --suite PATH` to collect
the suite; fingerprint-named output directories avoid dependence on thousands of filenames.

## Collection route selection and paid work

All three collection commands (`reasonese-run-conversation`, `reasonese-collect-data`, and
`reasonese-collect-studies`) accept `--route {free,paid,batch}` and `--allow-paid`. Routing is
invocation configuration, never part of the study YAML or a treatment axis.

| Preference | Model authors | Assistants |
|---|---|---|
| `free` (default) | Registered `:free` slug synchronously; otherwise paid synchronous fallback | Same |
| `paid` | Paid synchronous slug | Paid synchronous slug |
| `batch` | Batch API where registered and compatible; otherwise paid synchronous fallback | Paid synchronous slug under the existing web-search harness |

Inkling, Inkling Small, and Gemma 4 31B have registered free routes; both Qwen models lack one.
A failed free request never triggers a paid retry. `--no-batch` keeps authoring synchronous and
cannot be combined with `--route batch`. Neither option changes the fixed Luna QA or response
judge routes or their batching. Batch requests use the unsuffixed model slug on the Batch API,
not a `:batch` slug sent to the synchronous endpoint.

**No uncached chargeable collection work is allowed without `--allow-paid`.** This includes
paid authoring, message QA, response judgments, and assistant requests that expose chargeable
server-side web search. Thus even a cold collection using free model routes needs this opt-in:
free token pricing is not free end-to-end collection. The harness does not remove tools or skip
QA to avoid charges. Missing assistant work is rejected before authoring starts. A completely
warm cache still runs without an API key or paid permission; missing QA or judgments require it.
Standalone curation, message-checking, and response-judging commands retain their existing
explicit paid-provider behavior; this flag governs the three collection commands.

Before collection, stderr lists routes selected for missing work and the paid services involved.
The final JSON summary's `routes` lists author-message and assistant-trial counts by source,
requested slug, transport, and provider-reported final model. A cache hit retains its original
route regardless of the new preference; null provenance means the historical request is unknown.
Author counts refer to distinct materialized messages per pass, not token usage or billed requests;
assistant counts refer to trials, not the number of tool continuations. Mixed partial resumes
can report the same author in both cached and materialized groups.

The planner accepts repeated `--author` and the sampler accepts repeated `--author` and
`--assistant` filters, including `Gemma 4 31B`. Both default author selection and default assistant selection
contain only Inkling, Inkling Small, and Gemma 4 31B. Supplying a filter replaces its default
set; it does not append to it. Other supported models and `author: user` remain available
explicitly. Duplicate values are rejected. Filtering preserves
existing specification order; the planner summary counts only selected conditions.

These defaults apply when generating new plans and suites. Existing suite YAML and cached
responses keep their original models and provenance; regenerate a suite to adopt the new model
selection, and use separate caches if you need fresh collection on the free routes.

## Pilot web search

Assistant web search uses OpenRouter's Parallel engine with `mode: fast`, five results per
search, at most ten results and two search calls per API request. These bounds reset on a new
API request, including local-tool continuations; they are not a whole-trial or run spending cap.
The author and assistant model routes remain `free` by default. Search and the existing Luna
QA/judges require `--allow-paid`. Use a fresh output directory when changing the search engine
so cached traces from another search configuration are not silently reused.

The Inkling free endpoints are restricted to agentic harnesses. Their presence in the model
catalog and zero token prices do not establish access for this runner: a September 12, 2026
smoke test was rejected during authoring with HTTP 403 (`Gate Free Endpoints by Agentic Harness`).
Resolve provider access before launching the pilot; failed free calls do not trigger paid fallback.
