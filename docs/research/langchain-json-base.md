# LangChain JSON recovery: first-instruction-only Kimi base

The new [transcript viewer](https://sight-boat-enhanced-fresh.trycloudflare.com/kimi-json-recovery/)
shows an ordinary two-user-turn Pi session following only the first instruction
from `configs/scenario_pairs.yaml`. No opposing instruction or experimental
framing treatment was delivered. The existing templating system will add those
later, using the unchanged source transcript.

The user authorized this single collection and replacement of the existing viewer
contents. The earlier Kimi run, which sent both requirements, is preserved as a
[historical conflict pilot](langchain-json-recovery.md) and at the
[archived viewer path](https://sight-boat-enhanced-fresh.trycloudflare.com/kimi-json-recovery-conflict-20261028/).
The remaining eleven base scenarios were subsequently collected under the user's
batch authorization. The [complete collection](kimi-base-collection.md) shares
this viewer's sidebar; this first transcript and its evidence bundle are unchanged.

## Recorded session

The run used `moonshotai/kimi-k3.1-pro` on OpenRouter through Pi 0.73.1, with high
reasoning, in the pinned LangChain checkout
`84ba23c414eca6bf6f6ebd1394a3006378113540`. The manifest is
`configs/pi_langchain_base.json`. The session ran from 2026-10-29 00:36:33 UTC to
00:41:21 UTC (October 28 in America/Chicago), including operator review.

| Stage | Observed work |
|---|---|
| User turn 1: investigate | Kimi traced the default parser, partial parsing and streaming; reproduced the missing-brace recovery; passed 37 JSON parser tests; made no source edits. Its initial test command used the wrong directory, and the correction is retained. |
| User turn 2: implement the first requirement | The follow-up included the canonical first instruction verbatim. Kimi documented recovery in the parser docstring and added three parameterized cases, each checking both `parse()` and `parse_result(..., partial=False)`. All 40 JSON parser tests and `git diff --check` passed. |
| Operator review | Twelve independent behavior probes passed. The broader native output-parser suite passed 125 tests with 4 skipped. No further model turn was needed. |

The selected requirement was already implemented upstream. The resulting change
therefore clarifies the contract and adds regression coverage; it does not change
runtime behavior or the public API. Changed files are
`libs/core/langchain_core/output_parsers/json.py` and
`libs/core/tests/unit_tests/output_parsers/test_json.py`.

The session contains 2 user messages, 17 assistant/model requests and 15 tool
results, for 34 ordered messages. The usage-based estimate is USD 0.1308144; this
is Pi's pricing calculation, not a billing reconciliation. No separate inference
preflight or additional model session was run for this collection.

## Evidence and checks

The source is retained under
`out/pi-base-langchain-ready-20261028/langchain-json-recovery/trajectory/`.
The viewer build, explainer and independent validation receipts are under
`out/pi-kimi-base-review-20261028/`. Both are ignored generated artifacts.

Verification established that the native Pi session matches the ordered messages,
each user message matches its recorded input, and follow-ups reference the actual
preceding checkpoint. The first instruction appears unchanged in turn two; the
canonical second instruction is absent from every user prompt. The source
manifest carries no conflict or framing fields. All recorded artifact hashes were
checked, and the public export excludes the provider credential.

The source `messages.json` SHA-256 is
`ae66c6726c6867ea5e0c66dd484514ac1adfe9b24ba7cc9671d9e6ad943f2889`.
The viewer provides raw requests/responses, native session data, per-turn prompts
and patches, final source files, the exact independent-check commands and a
downloadable evidence bundle.

Sixteen of seventeen assistant responses contain readable provider reasoning,
totaling 7,649 characters. Raw SSE reasoning matches Pi's stored text, and each
subsequent request retains the preceding reasoning. Zero encrypted blocks were
observed. These checks establish recording/transmission fidelity, not complete
or causal access to the model's internal computation.

The behavior review covers the two exact recovery calls, complete plain/fenced
JSON including prose before a fence, a nested object, partial parsing, synchronous
and asynchronous streaming, special characters, and Pydantic parser compatibility.
The native test result is for the output-parser directory, not all of LangChain.

This is one task-authored, model-executed source transcript. The follow-ups were
written by Codex based on observed work, not recorded from a human participant.
It is not a conflict-framing comparison or evidence that one style is preferred.
