# Correctness invariants

- Axis enum values are the strings written to output and shown to users.
- Instructions are non-empty, trimmed phantom strings.
- Every instruction produces all 90 framing, channel, and author combinations.
- Duplicate or empty instruction collections are rejected.
- One instruction belongs to exactly one pair, and to exactly one side of it.
- A study pairs the two sides of one instruction pair; pairings across pairs are never formed.
- A pair's valid edge population is a bijection onto ranks, and every edge joins opposite sides
  and contains at least one user-message channel.
- Sampled pair designs are connected, cover every cell, and are reproducible from a seed derived
  from the pair identifier rather than the pair's position in the bank.
- Instruction is not a Bradley-Terry axis; only framing, channel, and author vary within a trial.
- Bradley-Terry fits one block per connected component, and each component self-centres on zero.
- Ranks are within-component; the comparison graph is expected to have one component per
  `(pair, assistant)` rather than to be connected.
- Both-completed and neither-completed trials are counted apart, though both score as ties.
- `PromptSpec` contains exactly the four axes and is runtime-checked by `beartype`.
- `specs_per_instruction()` returns a phantom non-negative integer.
- `MatchupInputs` contains exactly two datapoints and at least one explicit user-message
  channel; repeated channels are valid.
- The assistant is matchup metadata, not a fifth coordinate on `PromptSpec`.
- Materialized messages and conversations preserve the matchup's order and duplicates.
- Model-authored cache misses are grouped by author and use a batch variant when available.
- User-authored text is used verbatim and has no provider response.
- Message QA quotes the exact datapoint-derived authoring instructions and exact materialized text.
- Message-QA verdicts contain a real boolean; pass has no issues and failure has at least one.
- Exact message changes invalidate QA, and any false verdict blocks assistant inference without
  automatic regeneration.
- The message-QA route is `openai/gpt-5.6-luna:batch` with medium reasoning.
- Trace caches preserve the complete raw assistant response, including reasoning fields.
- A complete trace-cache hit performs no provider call and needs no API key once its exact
  messages also have cached passing QA.
- Judging emits exactly one verdict per matchup input in the same order.
- Each verdict is an actual boolean; numeric or truthy substitutes are rejected.
- Verdicts are independent, so no invariant requires exactly one true value.
- The judge route is `openai/gpt-5.6-luna:batch` with medium reasoning.
- A judgment cache hit requires both the same matchup and the same exact-trace fingerprint.
- Raw judge responses are retained alongside parsed verdicts.
- A cell is exactly one four-axis datapoint plus one assistant.
- Study inputs are an exact distinct pair and include an explicit user message.
- Both input orderings receive the same positive number of rollouts.
- Every cell has `2r` observations and `r` observations at each position.
- Each rollout has a separate trace cache; repeated responses cannot collapse into one record.
- Sparse pairing quotas are proportional to exact channel-pair and axis-difference stratum
  populations before any required connectivity repair.
- Sparse selection is without replacement, balances degree within channel, and repairs `k`
  sampled components with exactly `k - 1` edge swaps.
- Sampled suites use one root SQLite cache, require globally distinct trial IDs, and preserve the
  same requests, cache validation, judgments, and observations as per-study caches.
- Observation rows preserve trial, permutation, rollout, one-based position, and trace identity.
- Analysis requires exactly two cells at positions 1 and 2 and rejects duplicate cells or
  inconsistent metadata within trials.
- Each trial's cell pair becomes a win, loss, or 0.5 tie; tied trials are not discarded.
- Bradley-Terry fitting always uses a positive L2 penalty and reports graph components.
- Bootstrap intervals resample complete trials rather than dependent pair rows.
- Every axis receives marginal summaries and pairwise contrasts.
- Cell and axis position tables expose non-monotonic effects through position-specific rates
  and rate ranges, even when a linear correlation is zero.
- Regularization sensitivity reports all cell scores and ranks at 0.1×, 1×, and 10× penalty.

A new axis value changes the design size and should update code, tests, examples, and the
research definitions together.
