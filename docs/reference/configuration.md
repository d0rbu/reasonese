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
assistant: Nemotron 3.5 Lightning
inputs:
  - instruction: Solve the task called foobar and explain the result briefly.
    framing: normal
    channel: system prompt
    author: Nemotron 3.5 Lightning
  - instruction: What is two plus two? Answer with the number only.
    framing: casual
    channel: user message
    author: Gemma 4 31B
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
assistant: Nemotron 3.5 Lightning
rollouts_per_permutation: 2
inputs:
  - instruction: Solve the task called foobar and explain the result briefly.
    framing: normal
    channel: system prompt
    author: Nemotron 3.5 Lightning
  - instruction: What is two plus two? Answer with the number only.
    framing: casual
    channel: user message
    author: Gemma 4 31B
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
per assistant and defaults to 720. Default authors and assistants are Nemotron 3.5 Lightning
and Gemma 4 31B, both of which have registered `:free` routes:

```yaml
studies:
  - assistant: Nemotron 3.5 Lightning
    rollouts_per_permutation: 1
    inputs:
      - instruction: Write a program.
        framing: normal
        channel: system prompt
        author: Nemotron 3.5 Lightning
      - instruction: Find the requested information.
        framing: casual
        channel: user message
        author: Gemma 4 31B
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

Nemotron 3.5 Lightning, Inkling, Inkling Small, and Gemma 4 31B have registered free routes; both Qwen models lack one.
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
contain only Nemotron 3.5 Lightning and Gemma 4 31B. Supplying a filter replaces its default
set; it does not append to it. Other supported models and `author: user` remain available
explicitly. Duplicate values are rejected. Filtering preserves
existing specification order; the planner summary counts only selected conditions.

These defaults apply when generating new plans and suites. Existing suite YAML and cached
responses keep their original models and provenance; regenerate a suite to adopt the new model
selection, and use separate caches if you need fresh collection on the free routes.

Nemotron 3.5 Lightning uses `nvidia/nemotron-3.5-lightning:free` by default, or
`nvidia/nemotron-3.5-lightning` with `--route paid`. No Nemotron batch route is registered;
`--route batch` uses the ordinary paid synchronous fallback for this model. Both Inklings
remain supported via explicit filters, but their free endpoints rejected this runner's authoring
requests with an agentic-harness access restriction. Catalog listings do not establish live access.

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


## Adaptive API scheduling

One `OpenRouterClient` keeps independent limiter state for each **requested model slug** across
its sequential authoring and assistant stages. Free and paid slugs have separate state, and
multiple groups using the same slug share one limit. This applies to chat completions, tool
continuations, and batch submissions. Batch status polling retains its existing polling policy.

Each model starts with at most two in-flight requests. On success its concurrency window grows
by `1 / window`, up to `OpenRouterClient(sync_workers=8)` by default. `sync_workers` is now a
**per-model ceiling**, so two healthy models may have sixteen HTTP requests in flight in total.
Requests use asynchronous thread workers behind the existing blocking Python entry points.
An admitted HTTP attempt retains its model slot through response processing, including local
tools. Cooldowns and retries remain in the scheduler's queues. Response callbacks may run
concurrently across requests; each conversation's callbacks remain sequential. Ready tool
continuations take priority over new requests for the same model.

A definite HTTP 429 halves that model's window (floor one), doubles its minimum request-start
interval (initially one second, capped at thirty), and pauses **all new attempts for that model**
until the cooldown expires. The cooldown is at least that interval and honors the full
`Retry-After` value, including all three HTTP-date formats interpreted in UTC and values above
thirty seconds. Individual waits are capped at sixty seconds while retaining the full deadline,
so very large valid headers do not overflow platform timeouts. Missing, malformed,
negative, or nonfinite headers use the adaptive interval. Each subsequent successful request
reduces the interval by 20%; intervals below 10 ms return to zero. Successes from attempts already
in flight when a 429 arrived cannot undo the backoff. Each 429 logs the model, current
concurrency, start interval, and remaining cooldown to stderr. No model or paid-route fallback
occurs.

`OpenRouterClient(rate_limit_retries=3)` permits three retries after each request's initial
attempt, including each tool continuation. Retry configuration moved from `RequestsTransport`
to the client; the transport sends a single attempt. Reusing a request specification starts a
fresh budget. Empty answers have a separate limit of two retries per conversation; each
retry retains the model limiter and receives its own HTTP 429 retry budget. Successes already
in flight remain available even after another request for the same model exhausts its
budget; that model's queued work and later continuations stop, while healthy models finish
their queued requests.

Other HTTP failures, ambiguous network errors, malformed responses, and response-callback
errors stop all new admission and propagate after in-flight results are processed. A callback
failure never resends its already successful HTTP request. Already accepted batches are
collected despite a peer failure; failed or malformed batches do not discard valid peer batch
results. Duplicate batch request IDs are rejected rather than overwriting a response.

Successful authored messages and complete trial traces are saved in bulk on both success and
handled failure. Resuming reuses those results; QA, judging, and observation stages remain
blocked while required collection work has failed. This is an error-recovery guarantee, not
continuous checkpointing: abrupt process termination can still lose the active stage, and
incomplete assistant tool loops are not cached as completed trials.

Limits are local to one client invocation, not shared across processes, clients, or API keys.
They cannot override provider-wide or account-wide quotas. Offline tests establish scheduling
and retry behavior only; actual throughput and suitable concurrency ceilings require a
separately authorized live measurement.
