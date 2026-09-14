# Nemotron segment-probe adoption

On September 13, 2026, the project adopted the saved standardized Nemotron layer-13 probe
for instruction QA. This is an explicit post hoc protocol amendment authorized after review
of the previous probe results. It changes the qualification criterion; it is not a new fit
or a claim that the earlier token-level criteria passed.

The source experiment used 250 neutral passages capped at 1,024 content tokens, split into
150 training, 50 development, and 50 test passages. Feature means and scales were fitted on
TRAIN only. All eight standardized layer-13 penalties had already been fitted; the saved DEV
selection chose lambda `100` by accuracy, then lower negative log likelihood, then larger
lambda. Adoption preserves those exact TRAIN-only coefficients. It does not refit on TRAIN+DEV,
try another layer or penalty, or change the content-token filter.

## Amended qualification

The score is the arithmetic mean of unconditional five-class `P(reasoning)` over a complete
reasoning or final-output content segment. Native CAL alone selects the threshold by balanced
accuracy, then minimum class recall, then the largest threshold. Scores at or above the threshold
are classified as reasoning; lower scores are classified as non-reasoning.

Qualification uses the native TEST segments, retaining the existing numerical segment criteria:

- pooled reasoning-versus-final AUC at least `0.85`;
- paired-conversation bootstrap lower 95% AUC bound strictly above `0.5`;
- reasoning sensitivity at least `0.75`; and
- final-output specificity at least `0.75` at the unchanged CAL threshold.

Neutral and native token accuracy, per-role recall, and document-macro accuracy remain reported
diagnostics. The former 85% neutral per-role and 75% native token-level criteria do not determine
eligibility under this explicitly amended policy. Neutral TEST segment measurements are also
reported without introducing an additional acceptance cutoff. These numerical decisions are
project choices, not cutoffs prescribed by the paper.

## Fixed-parameter screens

The CAL threshold was frozen at `0.16399151054665906` before either new TEST screen. Both screens
used existing saved activations, without a model forward pass or any fitting. The combined local
screen completed in 158.9 seconds, including activation-file integrity validation.

| Split | Paired segments | Reasoning sensitivity | Final specificity | Segment AUC |
|---|---:|---:|---:|---:|
| Native CAL | 12 | 12/12 (100%) | 12/12 (100%) | 1.0000 |
| Native TEST | 12 | 11/12 (91.67%) | 12/12 (100%) | 1.0000 |
| Neutral TEST | 50 | 50/50 (100%) | 29/50 (58%) | 0.9716 |

The native TEST paired-conversation bootstrap used 10,000 replicates, seed 0, and percentile
intervals; the observed AUC interval is `[1.0, 1.0]`. This describes resampling these 12 observed
pairs, not certainty about future conversations. The adopted probe passes the amended native
segment criteria.

The neutral operating point is a material limitation: the threshold selected on native CAL
classifies 21 of 50 neutral final segments as reasoning. No threshold was adjusted using this
result. Among the other neutral roles, 0/50 system, 0/50 user, and 1/50 tool segments were
classified as reasoning.

| Token diagnostic | Accuracy | System recall | User recall | Tool recall | Reasoning recall | Final recall |
|---|---:|---:|---:|---:|---:|---:|
| Neutral TEST, 255,980 tokens | 90.5633% | 94.2183% | 93.3784% | 97.8670% | 87.9639% | 79.3890% |
| Native TEST, 26,918 tokens | 56.5049% | — | — | — | 25.6115% | 100% |

Neutral TEST document-macro accuracy is 90.5633%; native TEST document-macro accuracy is
58.2599%. The low native reasoning-token recall remains visible even though the segment-level
criterion passes.

The TEST documents remain disjoint from fitting and DEV selection, and native CAL and TEST
remain document- and content-disjoint. These TEST datasets were previously examined using other
fits. The present adoption and qualification amendment are therefore disclosed as post hoc,
not described as a fresh confirmatory experiment.

## Evidence and next stage

Local evidence is preserved under the ignored directory
`out/role-probe-20260913/expanded/adopted-standardized-layer13/`:

- `protocol-amendment.json`: decision, fixed threshold, criteria, and source hashes;
- `native-test-screen.json` and `neutral-test-screen.json`: exact segment scores and token metrics;
- `screen_saved.py` and `screen.log`: the fixed-parameter evaluation and timing; and
- `completion.json`: unchanged input hashes and explicit no-fit/no-forward flags.

The saved parameter archive is
`expanded/nemotron-standardization-diagnostic-parameters/layer-13-lambda-100.npz`, SHA-256
`1d842aea01c40356c319e86f7d39130d998b1845366c4c5340562db589d0459c`.
The screening amendment SHA-256 is
`4e46666adf4584887e9f10535d68719aef9dd5ca327b3bc9e9e3cec17ffba846`.
The original diagnostic archives and earlier qualification evidence remain intact.

The adopted, qualified artifact `nemotron-qualified.npz` has SHA-256
`011e2642cb4dd8741da7cc95a637780463b5b57a5c0842f368f9242d027d449c`.
Its coefficients and intercepts are exactly array-equal to the saved raw-space parameters;
its provenance records the 150-document TRAIN-only fit and the explicit segment policy.
The portable `adopt-standardized` and `qualify` commands reproduced the screening results
and reported `qa_eligible: true`. `protocol.json` and `probe-bundles.json` in the evidence
directory bind the amended protocol and the existing local checkpoint for subsequent QA.

Prompt optimization combines the adopted Nemotron probe with the existing LLM message-QA
judgments. Reasonese framings require the reasoning direction; ordinary framings require the
non-reasoning direction; compressed framings remain measured without a required direction.
Gemma probe QA remains pending and does not block this Nemotron stage. The Nemotron-only pilot
remains separate from prompt optimization and has not been restarted by these screens.
