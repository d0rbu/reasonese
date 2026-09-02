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

A new axis value changes the design size and should update code, tests, examples, and the
research definitions together.
