# Instruction configuration

Base instructions use versioned TOML:

```toml
schema_version = 1

[[instructions]]
id = "write_word_counter"
text = "Write a Python program that counts the unique words in a text file."
```

Rules:

- the only top-level keys are `schema_version` and `instructions`;
- `schema_version` must be `1`;
- at least one instruction is required;
- each instruction has exactly `id` and `text`;
- IDs are unique and match `^[a-z][a-z0-9_]*$`; and
- text is non-empty and has no surrounding whitespace.

Unknown keys are errors. See [`../../configs/example_instructions.toml`](../../configs/example_instructions.toml)
for a two-instruction example.
