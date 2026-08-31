# Instruction configuration

The input is a TOML array of base-prompt strings:

```toml
instructions = [
    "Write a Python program that counts the unique words in a text file.",
    "Find information about pathlib.Path.glob.",
]
```

At least one instruction is required. Each string must be non-empty and have no surrounding
whitespace. See [`../../configs/example_instructions.toml`](../../configs/example_instructions.toml).
