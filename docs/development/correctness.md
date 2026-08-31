# Correctness invariants

The foundation enforces these invariants:

- the enum values and order are explicit;
- base instruction IDs and text are validated;
- instruction IDs are unique within a configuration;
- every base instruction produces exactly 90 specifications;
- every framing, channel, and author combination occurs exactly once per instruction;
- specification IDs change when any coordinate or instruction text changes;
- TOML and JSONL boundaries reject unknown or missing fields; and
- JSONL writes replace the target only after the complete artifact is durable.

Tests should assert these properties directly. A new axis value intentionally changes the
design size and must update code, tests, schemas, examples, and research docs together.
