# Pilot QA recovery — September 15, 2026

This amendment follows the stopped Nemotron-only pilot and the user's authorization to make
local tool-budget exhaustion trial-local, use Luna message QA at high reasoning, reconsider
both framing acceptance cutoffs, and measure one authoring revision. Both QA gates remain hard
gates; compressed probe scores remain descriptive. Gemma qualification stays pending. The
probe coefficients, layer, checkpoint, and H1 capture implementation are not refitted.

## Failure and judgment contract

The old collection saved 164 of 166 QA-eligible trials before a local tool loop stopped the
supervisor. Of 720 planned comparisons, Luna excluded 386 and the probe excluded a further
251, leaving 83 comparisons. These exclusions are distinct from assistant execution failures.

A response requesting more tools after eight executed local rounds now terminates only its own
attempt. The actual tool-bearing response and eight executed steps are retained. Both completion
outcomes are false, with null judge responses and no provider judgment. The failed attempt
remains in the analysis population. Resuming reuses its terminal trace and judgments. Successful
traces keep their previous serialization and fingerprints. Per-study failure receipts include
trial IDs, both input axes, assistant, permutation, rollout, and trace fingerprints. Standalone
conversation summaries expose terminal status; standalone deterministic judgments identify
no LLM judge (`judge: null`). Replaying all 164 saved successful traces through original main
and this change produced exact equality of serialized payloads and scalar/batch fingerprints
(`trace-parity-report.json`), without provider calls or GPU work.

## Semantic QA and authoring

Luna message QA now uses high reasoning through the cheaper batch route. Response-completion
judging remains at medium. The revised rubric distinguishes additional mandatory obligations
from optional suggestions and tentative implementation plans. It evaluates requested outputs
semantically rather than demanding literal source wording. Concrete lost tools, quantities,
prohibitions, or deliverables remain failures. Task answers and rewriting commentary remain
invalid authoring outputs.

Exact request fingerprints invalidate cached QA when the rubric, effort, authoring guidance,
schema, route, specification, or text changes. Prompt comparison manifests also bind the request
policy using fixed placeholder content, permitting different candidate texts but rejecting
comparisons across changed judge settings.

The bounded comparison rejudges the saved 27 V3 development messages and one fresh 27-message
`obligation-preservation-v4` sample with the same revised judge. Fifteen prelabelled semantic
controls include six faithful/changed-obligation pairs and three disputed historical outputs.
The first authoring measurement helper was stopped after a cache API error; its raw response
was preserved. A fresh technical restart uses explicit request-response alignment. No judge
results were available or selected when this restart was made.

V3 remains the incumbent. V4 is selected only with strictly more jointly eligible DEV comparisons
and no reduction in Luna-compliant inputs under identical revised QA. Ties retain V3. The
existing fourth-pair confirmation set separately measures the same V4 brief: nine author inputs
and eighteen high-effort batch QA requests across V3/V4. Those results are excluded from prompt
selection and prompt editing. This is still one prompt revision, not another search round.

## Probe calibration design

Sixteen new tasks were split into eight CAL and eight TEST tasks before scoring. Nemotron
provided one planning-note sample per task. Manual review before seeing scores identified five
malformed controls: four task answers and one planning-style response missing the task. All
five attempts are reported and neither member of those pairs enters style calibration. Seven
CAL and four TEST tasks remain, each supplying a plain and planning instruction in all three
channels and both positions: 84 CAL and 48 TEST spans. The small TEST task count limits confidence.

The score is mean reasoning probability over the delivered target span, using the frozen
standardized layer-13 lambda-100 probe and H1 segment-prefix capture. Reasoning acceptance uses
a minimum and nonreasoning acceptance uses a separate strict maximum. The predeclared CAL loss
is `2 * false exclusion rate + wrong-style acceptance rate`, weighting classes and complete
task/channel/order cells equally. Ties prefer lower wrong-style acceptance, then proximity to
the historical 0.16399151054665906 cutoff. All-accept boundaries are diagnostic comparators,
not deployable cutoffs. A channel-specific policy must reduce equally weighted CAL loss by at
least 0.10 relative to a global pair; otherwise the global pair is preferred. This penalty and
loss are project choices, not paper thresholds. Selection is frozen before reading TEST scores.
Any overlap between acceptance ranges is explicit tolerance, not a unique classifier label.

The separate native-cutoff sensitivity review uses saved H1 segment scores and the existing
native CAL selection rule. TEST here is reused evidence, not an untouched qualification:

| Saved screen | Historical cutoff 0.163992 | Native-CAL candidate 0.085136 |
| --- | ---: | ---: |
| Native CAL reasoning recall | 12/12 | 12/12 |
| Native CAL final specificity | 12/12 | 12/12 |
| Reused native TEST reasoning recall | 11/12 | 12/12 |
| Reused native TEST final specificity | 12/12 | 12/12 |
| Reused neutral TEST reasoning recall | 50/50 | 50/50 |
| Reused neutral TEST final specificity | 30/50 | 18/50 |

Native qualification already meets its frozen criteria. This sensitivity review does not edit
the qualified NPZ or establish instruction-context validity. Token-level 85% per-role recall
remains diagnostic. Native AUC, bootstrap, and class-recall qualification criteria remain
unchanged; framing cutoffs are selected on the new instruction CAL controls.

## Measurement status

High-effort Luna comparison and instruction-context probe scores are pending. The GPU is
occupied by an unrelated job; the scorer waits for sufficient free memory. No new acceptance
cutoffs or authoring brief have been selected, and the pilot has not restarted. Results and the
per-pair, per-judge before/after table will be recorded before deployment.

Ignored source-bound measurements are under `out/pilot-qa-recovery-20260915/` in the main
repository, including `protocol.md`, `control-labels.json`, `semantic-controls.json`,
`native-threshold-review.json`, author request/response files, and live status files. Old pilot
and prompt-comparison artifacts remain preserved in their existing directories.
