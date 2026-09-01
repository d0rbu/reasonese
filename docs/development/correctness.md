# Correctness invariants

- Axis enum values are the strings written to output and shown to users.
- Instructions are non-empty, trimmed phantom strings.
- Every instruction produces all 90 framing, channel, and author combinations.
- Duplicate or empty instruction collections are rejected.
- `PromptSpec` contains exactly the four axes and is runtime-checked by `beartype`.
- `specs_per_instruction()` returns a phantom non-negative integer.
- `MatchupInputs` contains at least two datapoints and at least one explicit user-message
  channel; repeated channels and arbitrary larger tuples are valid.
- The assistant is matchup metadata, not a fifth coordinate on `PromptSpec`.
- Materialized messages and conversations preserve the matchup's order and duplicates.
- Model-authored cache misses are grouped by author and use a batch variant when available.
- User-authored text is used verbatim and has no provider response.
- Trace caches preserve the complete raw assistant response, including reasoning fields.
- A complete trace-cache hit performs no provider call and does not need an API key.
- Judging emits exactly one verdict per matchup input in the same order.
- Each verdict is an actual boolean; numeric or truthy substitutes are rejected.
- Verdicts are independent, so no invariant requires exactly one true value.
- The judge route is `openai/gpt-5.6-luna:batch` with medium reasoning.
- A judgment cache hit requires both the same matchup and the same exact-trace fingerprint.
- Raw judge responses are retained alongside parsed verdicts.
- A cell is exactly one four-axis datapoint plus one assistant.
- Study inputs are distinct, number at least two, and include an explicit user message.
- Every unique input permutation receives the same positive number of rollouts.
- Every cell has `n! × r` observations and `(n - 1)! × r` observations at each position.
- Each rollout has a separate trace cache; repeated responses cannot collapse into one record.
- Observation rows preserve trial, permutation, rollout, one-based position, and trace identity.

A new axis value changes the design size and should update code, tests, examples, and the
research definitions together.
