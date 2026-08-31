# Architecture

The package currently has one offline data flow:

```text
instruction TOML -> strict validation -> Cartesian planner -> prompt-spec JSONL
```

## Modules

- `reasonese.axes` defines the enums, display metadata, base `Instruction`, and axis manifest.
- `reasonese.config` loads a strict, versioned instruction-set TOML file.
- `reasonese.planning` enumerates `PromptSpec` records and assigns content-addressed IDs.
- `reasonese.io` atomically writes and strictly reads prompt-specification JSONL.
- `reasonese.cli` exposes `axes` and `plan` commands.

The planner creates condition specifications, not prompt text. There are no model-provider,
execution, response, or analysis layers in the current architecture.

## Determinism

Enumeration follows configured instruction order, then declared framing, channel, and
author order. Each specification ID hashes the instruction ID, instruction text, framing,
channel, and author. Identical inputs therefore produce identical ordered records, while
changing any coordinate changes the ID.
