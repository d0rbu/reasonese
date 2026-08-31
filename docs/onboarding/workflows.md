# Workflows

## Change an experiment design

1. Read `docs/research/protocol.md` and identify whether the change is pre- or post-collection.
2. Copy the TOML config if any response has already been collected.
3. Change the experiment identifier and treatment metadata.
4. Generate a fresh design and record its `design_id` and checksum.
5. Check the four counterbalances and update planned contrasts.
6. Update construct-validity notes for any new treatment.

## Import model responses

1. Generate and freeze the design JSONL.
2. Record an external run manifest with endpoint and decoding provenance.
3. Produce exactly one `ResponseRecord` for every `trial_id`.
4. Keep raw response text unchanged; do not repair it with a judge model.
5. Run `reasonese score`, then inspect invalid outputs before fitting.
6. Fit endpoints separately; do not silently pool deployment windows.

No provider collector is included yet. Adding or calling one requires explicit authorization
for the provider, cost, models, and run size.

## Change a schema

1. Decide whether the change is additive or breaking.
2. Increment `SCHEMA_VERSION` for incompatible records.
3. Keep readers strict: unknown and missing fields should fail.
4. Add round-trip and rejection tests.
5. Document a migration rather than guessing how to translate old artifacts.

## Before handoff

```bash
uv run pre-commit run --all-files
```

Report offline validation separately from treatment validation and live empirical status.
