# reasonese

`reasonese` is a research codebase for studying how the same instruction changes across
four controlled axes: instruction, framing, channel, and author.

## Current foundation

The repository defines the axes, enumerates their combinations, and executes ordered
multi-instruction matchups through OpenRouter. Judging and analysis are separate later steps.

| Axis | Current values |
|---|---|
| instruction | Simple base prompts such as “Write a program…” or “Find information…” |
| framing | `normal`, `casual`, `persuasive`, `subagent`, `reasonese-normal`, `reasonese-persuasive` |
| channel | `system prompt`, `user message`, `README.md` |
| author | `user`, `Qwen3.8 Flash`, `Qwen3.8 2.4T`, `Inkling`, `Inkling Small` |

Framing and author are independent. “Normal” is the author's default rendering in clear
prose. A `user`-authored input is treated as already written and used verbatim; model authors
rewrite the base instruction according to the selected framing.

Six framings, three channels, and five authors produce `6 × 3 × 5 = 90` specifications per
base instruction. A specification is just a four-field dataclass containing those axes.

## Quickstart

```bash
git clone https://github.com/d0rbu/reasonese.git
cd reasonese
uv sync

uv run reasonese-axes
uv run reasonese-plan \
  --instructions configs/example_instructions.toml \
  --output out/example/prompt_specs.jsonl

export OPENROUTER_API_KEY=...
uv run reasonese-run-conversation \
  --matchup configs/example_matchup.yaml \
  --message-cache out/generated_messages.yaml \
  --trace-cache out/conversation_traces.yaml
```

The utilities have separate entry points. `reasonese-axes` prints the values and
`reasonese-plan` writes four-axis datapoints. `reasonese-run-conversation` loads a `Matchup`,
generates any missing model-authored messages, constructs the ordered conversation, and sends
it to the selected assistant. It uses OpenRouter batch variants for groups that support them;
`--no-batch` forces synchronous authoring requests.

A matchup contains one assistant plus an ordered tuple of inputs. It requires at least two
inputs and at least one explicit `user message`; repeated channels are valid. Generated
messages and complete raw assistant responses, including returned reasoning fields, are cached
as readable YAML. A warm trace-cache hit does not require an API key or make a provider call.

## Implementation notes

The enums use their display text directly; there is no separate label mapping. Base
instructions and non-negative counts use `phantom-types`, while `beartype` checks public
function and dataclass boundaries at runtime.

- [`docs/research/axes.md`](docs/research/axes.md) defines the research constructs.
- [`docs/reference/architecture.md`](docs/reference/architecture.md) describes the small implementation.
- [`docs/reference/prompt-generation.md`](docs/reference/prompt-generation.md) traces exact prompt construction.
- [`docs/reference/output.md`](docs/reference/output.md) describes JSONL plans and YAML caches.

## Development

```bash
uv run pre-commit run --all-files
```

## License

MIT. See [`LICENSE`](LICENSE).
