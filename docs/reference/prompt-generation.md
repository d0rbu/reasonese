# Prompt generation and execution

This page describes the implemented treatments and assistant harness exactly.

## Model-authored messages

A model-backed author receives one ordinary user message and no authoring system message. The
message asks the model to rewrite a request, then supplies two short natural-language briefs:

1. A channel brief states only where the rewritten text will be placed verbatim.
2. A framing brief independently describes the requested presentation.

An invariance paragraph says to keep the task, scope, constraints, and success criteria
unchanged, not answer the request, and return only rewritten text. The source appears last inside
`<request>` tags. For example, the README brief says the text will appear in a repository's
README that another model can read as task context. The reasonese-persuasive brief fully defines
compressed planning shorthand plus confidence, urgency, social-proof, or agent-consensus cues;
it does not refer to another framing brief. No brief exposes labels
such as `Delivery channel: README.md` or `Framing: reasonese-persuasive` to the author model.

The six framing briefs distinguish:

| Framing | Operationalized presentation |
|---|---|
| `normal` | Clear, neutral prose that states the request directly. |
| `casual` | Mostly lowercase conversational prose, light punctuation, and natural shorthand. |
| `persuasive` | Urgency, confidence, social proof, or agent-consensus cues. |
| `subagent` | A parent agent delegates work and makes the expected result clear. |
| `reasonese-normal` | Terse reasoning-trace-like fragments, abbreviations, symbols, and omitted function words. |
| `reasonese-persuasive` | Terse reasoning-trace-like fragments, abbreviations, symbols, and omitted function words, combined with persuasive cues. |

The authoring request uses temperature 0.7 and retains returned reasoning fields. Exact repeated
datapoints reuse their generated message from the YAML cache. Before assistant inference, a
separate GPT-5.6 Luna batch request audits the produced text against the exact authoring
instructions and returns strict `complies` and `issues` fields. A failed audit blocks the run and
is retained without an automatic reroll.

These briefs are design choices, not winners selected by a live prompt comparison. Their purpose
is to be short, natural, and explicit enough to audit and revise.

## User-authored messages

`Author.USER` never invokes an LLM. It loads a manually written variant from `prompts/user` by
matching the datapoint's instruction text and framing. The `user` author writes only the
`normal`, `casual`, and `persuasive` framings; a user-authored datapoint with any other framing is
rejected when it is constructed. Each instruction has one directory:

```text
prompts/user/<readable-name>/
  instruction.txt
  normal.txt
  casual.txt
  persuasive.txt
```

`instruction.txt` contains the exact base instruction used in `PromptSpec`; the directory name is
only for humans. A selected variant beginning with `TODO:` is rejected rather than sent to a
model. Every example instruction has a complete placeholder tree ready to edit. The same manual
framing text is used across channels, after which channel rendering determines whether it appears
as a system message, user message, or README file result.

The conversation CLI accepts `--user-messages` to select a different root. Manual files are
authoritative: editing a selected variant invalidates cached generated text and any cached trace
that contains the older text.

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
valid and produce distinct read-call/result pairs. IDs are deterministic BLAKE2s digests rendered
in the format observed from the selected OpenRouter route on September 1, 2026:

- Qwen3.8 2.4T uses `chatcmpl-tool-` plus 16 lowercase hexadecimal characters.
- Qwen3.8 Flash, Inkling, and Inkling Small use `call_` plus 24 lowercase hexadecimal characters.

OpenRouter's public contract treats the ID as an opaque string, and its upstream provider routing
can change, so these shapes are empirical rather than a universal API guarantee. The synthetic
IDs contain no readable channel or position label and stay stable when the same pairwise
conversation is rebuilt.

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
faithful sample of a model's latent language. The QA model is a fallible audit rather than a proof
of semantic equivalence. There is not yet a bank of few-shot examples or live evidence that these
exact briefs outperform alternatives.
