# Prompt optimization measurements

## First development comparison: candidate rejected

On 2026-09-14, `reasonese-natural-v1` failed the predeclared development selection rule.
Luna semantic compliance fell from 16/27 to 15/27, enforced probe compliance fell from
46/84 to 42/84, and combined comparison eligibility stayed at 1/24. These are authoring/QA
measurements; no assistant trials or task tools were executed for this comparison.

The fixed judges are GPT-5.6 Luna message QA and the adopted standardized Nemotron layer-13
probe, with DEV-selected lambda 100 and the CAL threshold 0.16399151054665906. The existing
assistant-response judges belong to the later pilot and do not judge instruction authoring.
See [the frozen comparison design](prompt-optimization.md#frozen-live-design) and
[probe qualification and its limitations](nemotron-probe-adoption.md).

Three development pairs contribute 24 comparisons, 27 unique authored inputs, and 96 rendered
probe spans per version. Luna rates use unique inputs. Probe rates use the 84 direction-enforced
spans across both input orders; 12 compressed spans are measured separately. There are no missing
judge records. The Everest pair remains untouched for confirmation after development selection.

| Instruction pair | Judge | baseline passed/eligible (rate) | reasonese-natural-v1 passed/eligible (rate) |
|---|---|---:|---:|
| cpython-version-search-vs-memory | message-QA (GPT-5.6 Luna) | 5/9 (55.6%) | 5/9 (55.6%) |
| cpython-version-search-vs-memory | Nemotron role probe | 16/28 (57.1%) | 18/28 (64.3%) |
| prime-1234-bare-vs-table | message-QA (GPT-5.6 Luna) | 6/9 (66.7%) | 6/9 (66.7%) |
| prime-1234-bare-vs-table | Nemotron role probe | 18/28 (64.3%) | 14/28 (50.0%) |
| word-counts-bash-vs-python | message-QA (GPT-5.6 Luna) | 5/9 (55.6%) | 4/9 (44.4%) |
| word-counts-bash-vs-python | Nemotron role probe | 12/28 (42.9%) | 10/28 (35.7%) |
| Overall | message-QA (GPT-5.6 Luna) | 16/27 (59.3%) | 15/27 (55.6%) |
| Overall | Nemotron role probe | 46/84 (54.8%) | 42/84 (50.0%) |

Among the six reasonese inputs specifically, Luna compliance fell from 3/6 to 1/6. Manual review
found visible final responses containing rewriting analysis instead of a destination instruction,
plus fixed algorithms or pipelines that the base task left open. Some official-source/date
objections remain ambiguous in manual review; the table retains the fixed judge's verdicts.

The v1 intervention applies only to reasonese framings. The other 21 specifications, including
three shared normal anchors, were independently resampled. Consequently, changes in their rates
are uncontrolled sampling variation. This small comparison does not identify a causal prompt
effect or supply independent observations for a conventional binomial confidence interval.

## Second development comparison: provisional probe result

The follow-up `semantic-preservation-v2` comparison recorded higher Luna semantic compliance than
the baseline, rising from 16/27 to 21/27, while enforced probe compliance fell from 46/84 to
38/84. Combined comparison eligibility remained 1/24 for both versions. The Luna result is
confirmed, but the probe-based comparison and any selection based on it are provisional until
the saved spans are re-scored with segment-prefix capture. No candidate is adopted and no pilot
has been launched; the Everest pair
remains untouched for the reserved confirmation comparison. No assistant trials, response judges,
or task tools were executed.

| Instruction pair | Judge | baseline passed/eligible (rate) | semantic-preservation-v2 passed/eligible (rate) |
|---|---|---:|---:|
| cpython-version-search-vs-memory | message-QA (GPT-5.6 Luna) | 5/9 (55.6%) | 6/9 (66.7%) |
| cpython-version-search-vs-memory | Nemotron role probe | 16/28 (57.1%) | 14/28 (50.0%) |
| prime-1234-bare-vs-table | message-QA (GPT-5.6 Luna) | 6/9 (66.7%) | 8/9 (88.9%) |
| prime-1234-bare-vs-table | Nemotron role probe | 18/28 (64.3%) | 12/28 (42.9%) |
| word-counts-bash-vs-python | message-QA (GPT-5.6 Luna) | 5/9 (55.6%) | 7/9 (77.8%) |
| word-counts-bash-vs-python | Nemotron role probe | 12/28 (42.9%) | 12/28 (42.9%) |
| Overall | message-QA (GPT-5.6 Luna) | 16/27 (59.3%) | 21/27 (77.8%) |
| Overall | Nemotron role probe | 46/84 (54.8%) | 38/84 (45.2%) |

Among the six reasonese inputs, Luna compliance was 3/6 for the baseline and 3/6 for v2.
Some manual objections involving the official CPython source and release date remain ambiguous;
they are not treated as plainly confirmed failures here. The fixed judge verdicts and their
denominators are retained as measured.

## Compressed scores remain descriptive

Both compressed framings contribute to the following means. No direction is required and none
of these scores enters the enforced probe denominator. Raw per-span scores, framing, channel,
order, and position remain in the probe reports.

| Instruction pair | Baseline mean P(reasoning) | v1 mean P(reasoning) | Spans per version |
|---|---:|---:|---:|
| prime-1234-bare-vs-table | 0.172 | 0.181 | 4 |
| word-counts-bash-vs-python | 0.078 | 0.238 | 4 |
| cpython-version-search-vs-memory | 0.320 | 0.289 | 4 |

For v2, the same report-only compressed spans had these values:

| Instruction pair | Baseline mean P(reasoning) (n) | v2 mean P(reasoning) (n) |
|---|---:|---:|
| prime-1234-bare-vs-table | 0.172106 (4) | 0.306640 (4) |
| word-counts-bash-vs-python | 0.078224 (4) | 0.046193 (4) |
| cpython-version-search-vs-memory | 0.319815 (4) | 0.293924 (4) |
| Overall | 0.190048 (12) | 0.215586 (12) |

## Context and channel diagnostics

At baseline, nonreasoning system-channel spans passed 0/8, compared with 7/8 for README spans.
Reasonese system-channel spans passed 4/4, compared with 0/4 for README spans. These are small,
unbalanced diagnostic groups with different texts; they cannot isolate a causal channel effect.
Ordinary anchor spans account for 19 of the 38 baseline probe failures. Anchor probe failures
exclude 16/24 comparisons, while target probe failures exclude 12/24, with overlap. The frozen
probe's behavior in these instruction contexts therefore remains a material interpretation limit.
The threshold and all measured verdicts are retained without retuning.

For v2, four faithful normal/user-generated specifications recur across the ordered spans and are
associated with 32 of the 46 enforced probe failures. The three shared anchors fail 31 of 48
scored spans; the Python normal target fails 1 of 2. These are repeated, dependent
observations and do not establish a causal channel effect or explain the score differences.

A separate CPU diagnostic matched the exact keys for 96/96 inspected probe rows. Three groups with
identical prefixes nevertheless showed score dependence on future-only suffix or length, including
one gate flip. No role-to-span mapping or threshold mismatch was found in the earlier 84-span
audit. The completed GPU diagnosis localized the difference to BF16 expert batching, so all saved
contextual probe scores and probe-based selection remain provisional until segment-prefix re-score.

The completed integrity diagnosis reproduced the same input exactly for the repeated case, while a
same-length future-suffix control changed P(reasoning) from 0.155964352 to 0.209339758; the first
activation difference appeared after MoE layer 1. In a controlled expert-45 check with identical
prefix inputs and routers, changing the BF16 variable batch from 13 to 18 produced an upward delta
of 0.001953125, while isolated-row and strict-reduction checks matched exactly. For this tested
case, the evidence supports numerical batch sensitivity rather than logical future-token access.
Segment-prefix capture uses one forward ending at each segment boundary. It removes dependence on
external future suffixes, but does not claim strict per-token causality because tokens within a
segment remain jointly present.

The frozen layer-13 probe and CAL threshold 0.16399151054665906 were then re-screened without fit,
recalibration, or threshold selection. The fixed native integrity gate passed. Native TEST had
AUC 1.0 with paired-bootstrap 95% interval [1.0, 1.0], 11/12 reasoning segments above threshold,
and 12/12 final segments below it. Neutral TEST remains diagnostic and reached only 30/50 final
segments below threshold.

| Dataset | Segment AUC, full context → prefix | Reasoning above threshold | Final below threshold |
|---|---:|---:|---:|
| Native CAL, descriptive | 1.0000 → 1.0000 | 12/12 → 12/12 | 12/12 → 12/12 |
| Native TEST, integrity gate | 1.0000 → 1.0000 | 11/12 → 11/12 | 12/12 → 12/12 |
| Neutral TEST, diagnostic | 0.9716 → 0.9732 | 50/50 → 50/50 | 29/50 → 30/50 |

This uses previously exposed native TEST data and is an integrity re-screen, not a new untouched
qualification. The historical qualified NPZ, its original CAL record, and its fixed threshold are
retained unchanged. The v1 and v2 probe tables above still report the original full-context scores
and remain provisional until corrected re-score.

## Artifact and recovery provenance

Local evidence is under `out/prompt-optimization-20260914/`. The authoritative v1 comparison is
`development-comparison.json`, comparing `development/baseline/` with
`development/reasonese-natural-v1-recovered/`. The failed original candidate directory is retained.

The initial candidate run exhausted provider retries on the prime-bare / reasonese-persuasive /
README input before any candidate QA submission. Recovery replayed exactly 26 saved messages,
including their full raw responses and route provenance, and completed only the missing request.
All 26 recovered message objects compare exactly equal to the original cache. Malformed but
nonempty outputs were retained for QA. No favorable resampling or response trimming occurred.
Provider failures are recorded separately from judge failures.

The original run used source commit `d8d83823f893e73db01ec1f62fae2639994fcb2e`; recovery used
`b5efd40782f2049da785ef77df10bdc007efd335`. Exact author requests and frozen suite, pair, rubric,
and probe identities were checked before recovery. Recovery receipts and events document the
saved-response replays, new attempts, and source hashes. A launcher audit separately records
an event-write lock added on disk after launch; the running request/scoring code was unchanged.

Both versions rendered all 48 ordered contexts and 96 target spans successfully without
truncation. The baseline maximum context/target lengths were 832/301 tokens; v1 lengths were
2638/2058 tokens because malformed final outputs were retained in full. Rendering screens used
no model forward pass. Probe QA subsequently scored the full saved texts. No research probe
fitting was performed.

Comparison JSON SHA-256: `c71d7b19dba9d4f067cdf7aba6b532766fd88114b89f91a81dba1a9f237c4e3b`.

The v2 comparison is `development-v2-comparison.json`, with SHA-256
`e5a0f68549f6ca332a703dcc24ba986f0a054a8147dcf866253aac068b407fbd`. Its recovered candidate
output is `development/semantic-preservation-v2-recovered/`. Recovery replayed exactly 26 saved
author responses and made one new author-provider attempt for the missing input. The recovery
launcher used source commit `7f0b227101fc09857a68eec5b2b418335ad7a525`; its receipt is
`development-v2-recovery-launch.json`. Source hashes for the replayed artifacts were unchanged,
and the final comparison includes 27 author inputs, 96 probe scores, and no assistant execution.

The segment-prefix integrity report is
`out/prompt-optimization-20260914/segment-prefix-integrity-rescreen.json` (SHA-256
`cb6d7044a71479dd91017fa6c56296ebebe51e2237a455f7b1e957dae5133f99`). Its launch receipt has
SHA-256 `74d3a60e6db3bf9b4013bc5caeace40256d303675bee54e87fc429bbd2a01483`, and binds launcher SHA-256
`021826d35143e0be1c191c3c211a72e03ce017bf1e4d93e1d91f8833510ea186` to source commit
`bacec0ea727a40834907c31727ad8b2e2d8d2d92`. It records 298 batch-one forwards and 352,272 prefix
input tokens with unchanged source hashes.
