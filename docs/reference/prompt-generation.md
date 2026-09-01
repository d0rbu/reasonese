# Prompt generation and execution

This page describes the implemented treatments and assistant harness exactly.

## Authoring request

A model-backed author receives a two-message request. Its system message says to author one
instruction, preserve the source task, change only presentation, not solve the task, and return
only the finished instruction. The user message then gives two natural-language briefs rather
than a list of axis labels:

1. A channel brief explains how the target assistant will encounter the text and what wording
   belongs naturally in that context.
2. A framing brief operationalizes the selected treatment while requiring task equivalence.

The source request appears last inside `<source_request>` tags. For example, the README brief
explains that the text will live verbatim in repository documentation and will appear to the
target as file contents. The reasonese-persuasive brief asks for compressed planning shorthand
plus confidence, urgency, or agent-consensus cues. Neither brief exposes strings such as
`Delivery channel: README.md` or `Framing: reasonese-persuasive` to the author model.

The six framing briefs distinguish:

| Framing | Operationalized presentation |
|---|---|
| `normal` | Clear, neutral, conventional prose without pressure or role-play. |
| `casual` | Mostly lowercase informal prose, lighter punctuation, and natural shorthand. |
| `persuasive` | Credible urgency, confidence, social proof, or agent-consensus cues. |
| `subagent` | A parent agent delegates a bounded task and expected deliverable. |
| `reasonese-normal` | Terse reasoning-trace-like fragments, abbreviations, and symbols without persuasion. |
| `reasonese-persuasive` | The same compressed shorthand combined with persuasive cues. |

The authoring request uses temperature 0.7 and retains returned reasoning fields. Exact repeated
datapoints reuse their generated message from the YAML cache. `Author.USER` remains a verbatim
path: its input text must already embody the chosen framing.

## Channel rendering

The materialized text is placed into the assistant transcript as follows:

| Channel | Assistant-facing representation |
|---|---|
| `system prompt` | One `system` message containing the authored text verbatim. |
| `user message` | One `user` message containing the authored text verbatim. |
| `README.md` | An `assistant` call to `read_file` with `{"path":"README.md"}`, immediately followed by the authored text as that call's `tool` result. |

The README representation is transcript history, not a user message telling the model to read
tagged content. Every input retains its matchup position. A README input occupies two chat
messages, so downstream code maps datapoints through `ConversationSetup.content_for_input()`
instead of treating a datapoint index as a chat-message index. Repeated README channels remain
valid and produce distinct read-call/result pairs with distinct call IDs.

## Assistant tools

Every assistant request exposes four tools:

- `read_file` reads UTF-8 text inside the temporary task workspace;
- `bash` runs a command in that workspace;
- `python` runs isolated Python code there; and
- `openrouter:web_search` lets OpenRouter perform bounded current-information search.

Local shell and Python executions use `bubblewrap`: the task workspace is disposable, the host
filesystem is not mounted writable, networking is unshared, and time, address space, file size,
file descriptors, and captured output are limited. `read_file` rejects absolute paths and path
traversal. The temporary `README.md` contains the matchup's README treatments in their input
order for any later model-initiated reads.

OpenRouter executes server-side web-search calls itself. For local function calls, the runner
appends the raw assistant tool-call message and the local results, then asks the model to
continue. Eight local steps are allowed before the run fails. The final trace caches every raw
intermediate response, every local result, and the raw final response, preserving reasoning,
usage, citations, and other provider metadata.

## Current boundaries

The briefs are explicit operational definitions, not a claim that model-authored reasonese is a
faithful sample of a model's latent language. There is not yet a semantic-equivalence validator
or a bank of few-shot examples. Those are experimental-design extensions rather than hidden
transformations in the pipeline.
