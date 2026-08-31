# Architecture

The current package has one small offline flow:

```text
instruction strings -> Cartesian product -> four-field JSONL
```

- `reasonese.axes` defines the instruction phantom type and three `StrEnum` axes.
- `reasonese.config` reads a TOML array of instruction strings.
- `reasonese.planning` defines the four-field `PromptSpec` and enumerates combinations.
- `reasonese.io` writes those dataclasses as JSONL.
- `reasonese.cli` exposes `axes` and `plan`.

`Instruction` is a `phantom-types` string constrained to be non-empty and trimmed.
`specs_per_instruction()` returns `phantom.interval.Natural`, the library's non-negative
integer type. `beartype` checks public functions and the `PromptSpec` constructor.

There are no generated-prompt, provider, execution, response, or analysis layers yet.
