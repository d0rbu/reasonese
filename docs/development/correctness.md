# Correctness invariants

- Axis enum values are the strings written to output and shown to users.
- Instructions are non-empty, trimmed phantom strings.
- Every instruction produces all 90 framing, channel, and author combinations.
- Duplicate or empty instruction collections are rejected.
- `PromptSpec` contains exactly the four axes and is runtime-checked by `beartype`.
- `specs_per_instruction()` returns a phantom non-negative integer.

A new axis value changes the design size and should update code, tests, examples, and the
research definitions together.
