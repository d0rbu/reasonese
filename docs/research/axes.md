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
| `reasonese-normal` | A first-person, self-directed task note in the author model's natural planning voice. |
| `reasonese-persuasive` | The same first-person planning style with natural self-encouragement or commitment to compliance. |
| `compressed-normal` | Terse planning shorthand using fragments, abbreviations, symbols, and omitted function words. |
| `compressed-persuasive` | Compressed shorthand with deliberate confidence, urgency, social-proof, or consensus cues. |

These values name intended treatments. Model authors receive explicit transformation guidance.
User-authored inputs load the matching manually written framing from `prompts/user`. The `user`
author writes only `normal`, `casual`, and `persuasive`. The `subagent`, `reasonese-normal`,
`reasonese-persuasive`, `compressed-normal`, and `compressed-persuasive` framings are produced
by model authors only, so no user-authored datapoint carries them and constructing one is rejected.

Reasonese targets stylistic resemblance to each author's ordinary first-person planning prose.
It is a generated instruction representation, not a captured reasoning trace or a claim that
the message is empirically indistinguishable from one. First-person intent still instructs the
future executor to perform the base task; the author must not solve it. Compression is an
independent framing choice, not evidence of a model-native language. Style fidelity requires
separate empirical validation.

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
- `Gemma 4 31B`
- `Nemotron 3.5 Lightning`

The enum strings above are the display values; there is no second label or identifier map.
Author means whoever writes the framed instruction, not the executor model.

## Assistant

The assistant is not a fifth entry axis. A `Matchup` places an ordered tuple of four-axis
datapoints in front of one of the six model-backed author values. This cleanly separates who
writes each instruction from which model receives the resulting conversation.

## Design size

```text
model authors: 8 framings × 3 channels × 6 authors = 144
user author:   3 framings × 3 channels × 1 author  =  9
                                             total = 153 specifications per instruction

24 pairs × 2 instructions × 153                    = 7,344 specifications
153 × 153 − 102 × 102                                = 13,005 eligible pairings per pair
2 × 153                                            =   306 cells per pair
```

A pairing is eligible when it joins the two sides of one pair and at least one input uses the
`user message` channel. The comparison graph has one component per `(pair, assistant)`.

No outcome or model-behavior claim is encoded in a specification.

The counts above describe all supported axis values. The planner and sampler default to
Nemotron 3.5 Lightning and Gemma 4 31B as authors and the sampler uses those same two
assistants. That subset has 48 specifications per instruction (2,304 across the bank),
96 cells and 1,280 eligible pairings per pair. Explicit author/assistant filters can select
other supported values; model availability does not change the axis definitions.
