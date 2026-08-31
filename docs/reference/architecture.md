# Architecture

The package has a deterministic planning flow:

```text
instruction strings -> Cartesian product -> four-field JSONL
```

- `reasonese.axes` defines the instruction phantom type and three `StrEnum` axes.
- `reasonese.config` reads a TOML array of instruction strings.
- `reasonese.planning` defines the four-field `PromptSpec` and enumerates combinations.
- `reasonese.io` writes those dataclasses as JSONL.
- `reasonese.show_axes` prints the axis values.
- `reasonese.plan` parses paths and writes the planned combinations.

The conversation flow is:

```text
matchup -> authored messages -> ordered conversation -> assistant trace
```

- `reasonese.matchup` validates an assistant and an ordered tuple of two or more datapoints.
- `reasonese.openrouter` provides synchronous completions and model-grouped batch authoring.
- `reasonese.conversation` builds authoring requests and channel-specific chat messages.
- `reasonese.cache` stores generated messages and raw traces in readable YAML.
- `reasonese.runner` coordinates cache lookup, generation, construction, and execution.
- `reasonese.run_conversation` is the standalone conversation utility.

The two utilities have separate console entry points. There is no package-level dispatcher
or shared command switch.

`Instruction` is a `phantom-types` string constrained to be non-empty and trimmed.
`specs_per_instruction()` returns `phantom.interval.Natural`, the library's non-negative
integer type. `beartype` checks public functions and the `PromptSpec` constructor.

`MatchupInputs` is a `phantom-types` refined tuple: it contains at least two `PromptSpec`
objects and at least one explicit user-message channel. Repeated channels and more than two
inputs are valid. `Assistant` shares the four model values with model-backed authors but is
separate from the four entry axes.

The OpenRouter key exists only at the transport boundary. Cache keys are structural input
coordinates, and raw provider responses are retained so reasoning and provider metadata are
not discarded. The package does not yet judge or analyze responses.
