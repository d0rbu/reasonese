# Prompt optimization measurements

## Corrected development comparison

The current result uses `segment_prefix_capture_v1`: one model forward ending at each scored
segment boundary. Saved authored texts and GPT-5.6 Luna judgments are byte-for-byte unchanged;
only local Nemotron probe scores were recomputed. All 288 planned forwards completed, with 96
scores for each version and no provider, assistant, response-judge, or task-tool calls.

Three DEV pairs contribute 24 comparisons, 27 unique authored inputs, and 96 probe spans per
version. Luna rates use unique inputs. Probe rates use 84 direction-enforced spans; 12 compressed
spans are descriptive. There are no missing records.

| Instruction pair | Judge | baseline | reasonese-natural-v1 | semantic-preservation-v2 |
|---|---|---:|---:|---:|
| cpython-version-search-vs-memory | message QA (GPT-5.6 Luna) | 5/9 (55.6%) | 5/9 (55.6%) | 6/9 (66.7%) |
| cpython-version-search-vs-memory | Nemotron role probe | 17/28 (60.7%) | 18/28 (64.3%) | 19/28 (67.9%) |
| prime-1234-bare-vs-table | message QA (GPT-5.6 Luna) | 6/9 (66.7%) | 6/9 (66.7%) | 8/9 (88.9%) |
| prime-1234-bare-vs-table | Nemotron role probe | 20/28 (71.4%) | 13/28 (46.4%) | 12/28 (42.9%) |
| word-counts-bash-vs-python | message QA (GPT-5.6 Luna) | 5/9 (55.6%) | 4/9 (44.4%) | 7/9 (77.8%) |
| word-counts-bash-vs-python | Nemotron role probe | 13/28 (46.4%) | 11/28 (39.3%) | 13/28 (46.4%) |
| **Overall** | **message QA (GPT-5.6 Luna)** | **16/27 (59.3%)** | **15/27 (55.6%)** | **21/27 (77.8%)** |
| **Overall** | **Nemotron role probe** | **50/84 (59.5%)** | **42/84 (50.0%)** | **44/84 (52.4%)** |

Joint eligibility requires both message QA and every enforced probe span for a study to pass.

| Instruction pair | baseline | reasonese-natural-v1 | semantic-preservation-v2 |
|---|---:|---:|---:|
| cpython-version-search-vs-memory | 2/8 | 1/8 | 1/8 |
| prime-1234-bare-vs-table | 0/8 | 0/8 | 0/8 |
| word-counts-bash-vs-python | 0/8 | 0/8 | 0/8 |
| **Overall** | **2/24** | **1/24** | **1/24** |

The recorded rule retains the baseline unless a candidate improves joint LLM/probe eligibility
without losing LLM semantic compliance. Neither candidate passes: v1 loses Luna compliance and
joint eligibility; v2 improves Luna compliance but loses joint eligibility. The baseline is
retained under that rule. No candidate is adopted, and no pilot or reserved Everest confirmation
run has launched. The baseline was selected for the pilot protocol with both Luna semantic
compliance and the local probe retained as hard gates. The separately defined
`constraint-scope-v3` brief is unmeasured and does not alter this result.

Among the six reasonese inputs, Luna compliance was 3/6 for the baseline, 1/6 for v1, and 3/6 for
v2. Manual objections involving the official CPython source and release date remain ambiguous;
the fixed judge verdicts and denominators remain the measured authority.

## Compressed scores remain descriptive

Compressed framings have no required direction and are excluded from the enforced denominator.
Entries are mean P(reasoning) across the scored spans.

| Instruction pair | baseline | reasonese-natural-v1 | semantic-preservation-v2 | n/version |
|---|---:|---:|---:|---:|
| cpython-version-search-vs-memory | 0.314702 | 0.292808 | 0.296143 | 4 |
| prime-1234-bare-vs-table | 0.220803 | 0.219287 | 0.317374 | 4 |
| word-counts-bash-vs-python | 0.097032 | 0.232866 | 0.036312 | 4 |
| **Overall** | **0.210846** | **0.248320** | **0.216610** | **12** |

Raw per-span scores, framing, channel, order, and position remain in the reprobe reports. Repeated
anchors are dependent observations, and independently sampled texts outside changed guidance
mean these small measurements do not reliably isolate a prompt effect.

## Historical full-context results and integrity diagnosis

The original reports recorded probe passes of 46/84 for baseline, 42/84 for v1, and 38/84 for v2;
joint counts were 1/24 for all three. Luna counts were already those shown above. Those probe
counts remain historical provenance rather than the current selection result. The original v1
comparison is `development-comparison.json` (SHA-256
`c71d7b19dba9d4f067cdf7aba6b532766fd88114b89f91a81dba1a9f237c4e3b`); the original v2 comparison
is `development-v2-comparison.json` (SHA-256
`e5a0f68549f6ca332a703dcc24ba986f0a054a8147dcf866253aac068b407fbd`).

A CPU audit matched 96/96 saved keys but found three shared-prefix groups whose scores changed with
future-only suffix or sequence length. The GPU diagnosis localized the first activation difference
after MoE layer 1. Expert 45 received identical prefix inputs and router values but different BF16
batch sizes: ordinary batched projection differed by 0.001953125, while isolated-row and strict
reduction checks matched exactly. This supports numerical batch sensitivity rather than logical
future-token access for the tested case.

Segment-prefix capture removes external future-suffix dependence by ending each forward at the
scored segment boundary. It does not claim strict per-token causality within the segment. In the
corrected reprobe, all three previously identified eight-row prefix groups had zero score range.

## Frozen-probe integrity re-screen

The frozen layer-13 probe and CAL threshold 0.16399151054665906 were re-screened without fitting,
recalibration, or threshold selection. The historical qualified NPZ and CAL record remain unchanged.

| Dataset | Segment AUC, full context → prefix | Reasoning above threshold | Final below threshold |
|---|---:|---:|---:|
| Native CAL, descriptive | 1.0000 → 1.0000 | 12/12 → 12/12 | 12/12 → 12/12 |
| Native TEST, integrity gate | 1.0000 → 1.0000 | 11/12 → 11/12 | 12/12 → 12/12 |
| Neutral TEST, diagnostic | 0.9716 → 0.9732 | 50/50 → 50/50 | 29/50 → 30/50 |

Native TEST paired-bootstrap AUC remained 1.0 with 95% interval [1.0, 1.0], so the fixed gate
passed. TEST was previously exposed, making this an integrity re-screen rather than a new untouched
qualification. Token metrics and neutral TEST remain diagnostic; neutral 30/50 final specificity
is a material limitation.

## Artifact provenance

Corrected artifacts are under `out/prompt-optimization-20260914/segment-prefix-reprobe/`. The run
manifest SHA-256 is `16050a8cea4df23af9c82a4e6eca05b29d704bcb6004229c31a58fa1789c75a5`.
The v1 comparison SHA-256 is `1512e00fc0d4bf7731e778737a0f42c12abbd90a47f54552225ca5cea15383a3`;
the v2 comparison SHA-256 is `796b8581dbd46319dde8d2c0792e31ed7b42173d40a2f8cbeaa03cf6337c284c`.
The manifest binds unchanged authored-message and Luna-QA artifacts, the frozen probe, integrity
screen, source commit `f325af8bfa0accece375797ecc1c0ffe36dbfc2f`, and reprobe script. It records
288 forwards and no provider calls.

The integrity report is `out/prompt-optimization-20260914/segment-prefix-integrity-rescreen.json`
(SHA-256 `cb6d7044a71479dd91017fa6c56296ebebe51e2237a455f7b1e957dae5133f99`). Its launch receipt
SHA-256 is `74d3a60e6db3bf9b4013bc5caeace40256d303675bee54e87fc429bbd2a01483` and binds launcher SHA-256
`021826d35143e0be1c191c3c211a72e03ce017bf1e4d93e1d91f8833510ea186` to source commit
`bacec0ea727a40834907c31727ad8b2e2d8d2d92`.
