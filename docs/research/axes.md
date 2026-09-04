# The axes

Each planned condition has four coordinates. Three of them—framing, channel, and author—are
independent treatments. The fourth, instruction, is a blocking factor.

## Instruction

An instruction is a simple base prompt, such as asking for a program that does something or
requesting information about a topic. It is stored directly as a non-empty, trimmed string.
Framing later changes how that prompt is expressed.

Instructions are not free. They come in 24 mutually exclusive pairs, defined in
[`configs/instruction_pairs.yaml`](../../configs/instruction_pairs.yaml), because a trial only
carries signal when the assistant completes one instruction and not the other. Arbitrary
instructions cannot be paired, so a study always pairs one side of a pair with the other.

That makes instruction unusable as a treatment axis. A comparison exists only between two cells
that share a trial, and a trial only ever holds the two instructions of one pair, so the
comparison graph has no edges between pairs. Bradley-Terry scores are identified only up to a
per-component shift, and an instruction contrast would difference arbitrary offsets. Instruction
therefore selects which cells can be compared, and analysis treats `(pair, side)` as a block
rather than reporting an instruction margin.

## Framing

| Value | Intended treatment |
|---|---|
| `normal` | The author's clear, neutral rendering of the request. |
| `casual` | Conversational wording, lowercase text, and reduced punctuation. |
| `persuasive` | Natural language deliberately intended to secure compliance. |
| `subagent` | A delegation written as though a parent agent were instructing a subagent. |
| `reasonese-normal` | A compressed reasonese representation without deliberate persuasive intent. |
| `reasonese-persuasive` | A compressed reasonese representation deliberately intended to secure compliance. |

These values name intended treatments. Model authors receive explicit transformation guidance.
User-authored inputs load the matching manually written framing from `prompts/user`.

## Channel

- `system prompt`: place the framed instruction in the executor's system prompt.
- `user message`: place it in a user message.
- `README.md`: present it as the result of an assistant `read_file("README.md")` tool call.

During execution, system and user inputs become chat messages in their original order.
`README.md` content is not wrapped in a user message. It appears as file-read tool history at
the datapoint's ordered position in the transcript.

## Author

- `user`
- `Qwen3.8 Flash`
- `Qwen3.8 2.4T`
- `Inkling`
- `Inkling Small`

The enum strings above are the display values; there is no second label or identifier map.
Author means whoever writes the framed instruction, not the executor model.

## Assistant

The assistant is not a fifth entry axis. A `Matchup` places an ordered tuple of four-axis
datapoints in front of one of the four model-backed author values. This cleanly separates who
writes each instruction from which model receives the resulting conversation.

## Design size

```text
6 framings × 3 channels × 5 authors = 90 specifications per instruction
24 pairs × 2 instructions × 90                = 4,320 specifications
90 × 90 − 60 × 60                             = 4,500 eligible pairings per pair
```

A pairing is eligible when it joins the two sides of one pair and at least one input uses the
`user message` channel. The comparison graph has one component per `(pair, assistant)`.

No outcome or model-behavior claim is encoded in a specification.
