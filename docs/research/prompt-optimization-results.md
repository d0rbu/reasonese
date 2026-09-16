# Prompt optimization measurements

## Interpretation update — September 16, 2026

The results below preserve measurements from the earlier joint-eligibility analysis. Their
historical scores and counts are unchanged; probe mismatches in those tables do not exclude
comparisons under the current protocol. Luna high-reasoning message QA remains the hard gate.
Current prompt-optimization runs default probe scoring off, and optional inline scores, missing
measurements, or scorer errors are diagnostic only. No probe fit, threshold, or prompt was changed.

## Final development result

The original re-score used `segment_prefix_capture_v2_nemotron_cumsum_h1`. It reused the exact
saved authored messages and GPT-5.6 Luna judgments and recomputed only the 384 local probe spans
across four briefs. No provider, assistant, task-tool, response-judge, fitting, recalibration, or
threshold-selection calls were made by that re-score.

Three development pairs contribute 24 comparisons, 27 unique authored inputs, 84 enforced probe spans, and 12 compressed descriptive spans per brief. There are no missing records.

| Instruction pair | Judge | baseline | v1 | v2 | V3 |
|---|---|---:|---:|---:|---:|
| cpython-version-search-vs-memory | Luna message QA | 5/9 (55.6%) | 5/9 (55.6%) | 6/9 (66.7%) | 7/9 (77.8%) |
| cpython-version-search-vs-memory | Nemotron role probe | 17/28 (60.7%) | 18/28 (64.3%) | 19/28 (67.9%) | 22/28 (78.6%) |
| prime-1234-bare-vs-table | Luna message QA | 6/9 (66.7%) | 6/9 (66.7%) | 8/9 (88.9%) | 8/9 (88.9%) |
| prime-1234-bare-vs-table | Nemotron role probe | 20/28 (71.4%) | 13/28 (46.4%) | 12/28 (42.9%) | 12/28 (42.9%) |
| word-counts-bash-vs-python | Luna message QA | 5/9 (55.6%) | 4/9 (44.4%) | 7/9 (77.8%) | 7/9 (77.8%) |
| word-counts-bash-vs-python | Nemotron role probe | 13/28 (46.4%) | 11/28 (39.3%) | 13/28 (46.4%) | 13/28 (46.4%) |
| **Overall** | **Luna message QA** | **16/27 (59.3%)** | **15/27 (55.6%)** | **21/27 (77.8%)** | **22/27 (81.5%)** |
| **Overall** | **Nemotron role probe** | **50/84 (59.5%)** | **42/84 (50.0%)** | **44/84 (52.4%)** | **47/84 (56.0%)** |

Historical joint eligibility required Luna compliance and every enforced probe span in the
comparison to pass. It is preserved as an earlier analysis measure, not current eligibility.

| Instruction pair | baseline | v1 | v2 | V3 |
|---|---:|---:|---:|---:|
| cpython-version-search-vs-memory | 2/8 | 1/8 | 1/8 | 3/8 |
| prime-1234-bare-vs-table | 0/8 | 0/8 | 0/8 | 0/8 |
| word-counts-bash-vs-python | 0/8 | 0/8 | 0/8 | 0/8 |
| **Overall** | **2/24** | **1/24** | **1/24** | **3/24** |

The predeclared historical rule selected V3 because joint eligibility improved from 2/24 to 3/24
while Luna compliance improved from 16/27 to 22/27. The selection is recorded in
`out/prompt-optimization-20260914/selection.json` (SHA-256
`b9c7cea2dbf6430a43f35f56eb549e3acad10ee33a6e49ad76bb5d546e92d27c`). Under the current
protocol, only Luna message QA gates collection and comparison eligibility; probe scores are
diagnostics. The small dependent development comparison does not reliably isolate a prompt effect
or establish broad superiority.

Luna and probe scores retain independent denominators: a Luna-rejected input can still receive
probe scores. The optimization runs stop before assistant execution, tool use, response judging,
and observation writing. At this artifact snapshot, no pilot had launched and Gemma probe
qualification remained pending.

## Reasonese and non-reasonese splits

| Judge/group | baseline | v1 | v2 | V3 |
|---|---:|---:|---:|---:|
| Luna, reasonese inputs | 3/6 | 1/6 | 3/6 | 4/6 |
| Luna, other inputs | 13/21 | 14/21 | 18/21 | 18/21 |
| Probe, reasonese target spans | 6/12 | 8/12 | 8/12 | 8/12 |
| Probe, non-reasonese enforced spans | 44/72 | 34/72 | 36/72 | 39/72 |

Probe results by pair and framing group are:

| Instruction pair | Group | baseline | v1 | v2 | V3 |
|---|---|---:|---:|---:|---:|
| cpython-version-search-vs-memory | Reasonese | 3/4 | 2/4 | 4/4 | 3/4 |
| cpython-version-search-vs-memory | Non-reasonese | 14/24 | 16/24 | 15/24 | 19/24 |
| prime-1234-bare-vs-table | Reasonese | 1/4 | 4/4 | 2/4 | 3/4 |
| prime-1234-bare-vs-table | Non-reasonese | 19/24 | 9/24 | 10/24 | 9/24 |
| word-counts-bash-vs-python | Reasonese | 2/4 | 2/4 | 2/4 | 2/4 |
| word-counts-bash-vs-python | Non-reasonese | 11/24 | 9/24 | 11/24 | 11/24 |

Reasonese combines `reasonese-normal` and `reasonese-persuasive`. Non-reasonese combines normal, casual, persuasive, and subagent spans, including repeated shared anchors. These are dependent rendered-span counts across both orders. V3 still has 0/6 jointly eligible reasonese comparisons. Five of those six comparisons have at least one failing normal opposite-side anchor, so their joint failures cannot be attributed solely to reasonese style.

Manual review of V3's visible messages recorded 21 clear passes, four clear task-preservation failures, and two ambiguous cases. Luna passed all 21 manual passes, rejected all four clear failures, and split the ambiguous cases. This supports the fixed judge as the measured authority while retaining the manual caveats.

## Compressed descriptive scores

Compressed framings have no required direction and do not enter eligibility. Entries are mean P(reasoning) over four spans per pair.

| Instruction pair | baseline | v1 | v2 | V3 |
|---|---:|---:|---:|---:|
| cpython-version-search-vs-memory | 0.314702 | 0.292808 | 0.296143 | 0.180800 |
| prime-1234-bare-vs-table | 0.220803 | 0.219287 | 0.317374 | 0.212575 |
| word-counts-bash-vs-python | 0.097032 | 0.232866 | 0.036312 | 0.044252 |

## Capture-policy integrity

The earlier mismatch was numerical rather than a mapping error. Identical prefixes first diverged after Nemotron's first MoE layer when future rows changed expert batch shape. Kernel controls then isolated the cross-process choice to the autotuned cumsum head tile: registered H1 reproduced the historical saved activation, while H4 through H64 produced the alternate stable activation. The production scorer now validates and scopes H1 for Nemotron CUDA captures and restores the autotuner state afterward. H1 was chosen for historical activation parity, not QA performance.

Two independent fresh H1 processes agreed exactly on all 15 controls and all 60 stored NPZ fields. Nine DEV controls matched their historical saved probabilities exactly; six native CAL controls differed from their historical probabilities by at most 0.0029773168680727324, with zero threshold-decision flips. Their reports are `cumsum-h1-fresh-process-a.json` (SHA-256 `b142475d6f37a871bcffd50cf4c6342333db0cd54de747936090adf9dffbccb1`) and `cumsum-h1-fresh-process-b.json` (SHA-256 `b1001b21de5e4acd5006ad8ce0a398e4a2df8120ad037cfc77e141d0a5c9a41e`). Segment-prefix capture removes external future-suffix dependence but does not establish strict per-token causality inside a segment.

The fixed-parameter H1 integrity screen passed the existing native gates without fitting or threshold changes:

| Dataset | AUC | Reasoning sensitivity | Final specificity |
|---|---:|---:|---:|
| Native CAL, descriptive | 1.0000 | 12/12 | 12/12 |
| Native TEST, integrity re-screen | 1.0000 | 11/12 | 12/12 |
| Neutral TEST, diagnostic | 0.9728 | 50/50 | 30/50 |

Native TEST paired-bootstrap AUC was 1.0 with 95% interval [1.0, 1.0]. TEST was previously exposed, so this is an integrity re-screen rather than a new untouched qualification. Neutral final specificity remains a material transfer limitation. The frozen layer-13 lambda-100 NPZ and CAL-selected threshold `0.16399151054665906` are unchanged.

## Artifacts and historical provenance

The generated H1 report is `out/prompt-optimization-20260914/cumsum-h1-reprobe-report.json` (SHA-256 `f2e676fa7496e60ba15a98800dc3b00f56e600c816cc93969490e96f86639e57`). The 384-score reanalysis run manifest is `segment-prefix-reprobe-cumsum-h1/run-manifest.json` (SHA-256 `23239d3c66a41b9fe1e1c2492511660ea13290557d2bd6616b534bc4721b1a15`). The H1 integrity-screen report is `segment-prefix-integrity-rescreen-cumsum-h1.json` (SHA-256 `7047fb487ba8c7162e971e1cbf9821ece9a9b14862fc1af3598fae2645a04b4c`).

The H1 comparison hashes are `714aa42e...` for v1, `70d5fe1b...` for v2, and `52b9f277...` for V3; full hashes are recorded in the generated report. Original full-context and autotuned prefix results remain historical diagnostic provenance and are not current selection evidence. The saved authored text and Luna artifacts are byte-identical to their originals.

The runtime change is commit `f69dc61f3322ff88824c834bedfff5487c5845a4`. The required pinned local offline gate passed lock, Ruff, ty, and full pytest with 95.33% coverage. CI independently collected 1,120 items and reported 1,119 passed, one skipped, and 95.31% coverage. No new probe fitting or threshold selection occurred.

## Reserved confirmation

The disjoint Everest confirmation was launched after the V3 selection was hash-bound, so it cannot retune the development decision.

| Held-out judge/group | baseline | V3 |
|---|---:|---:|
| Luna, all inputs | 6/9 | 7/9 |
| Probe, all enforced spans | 7/28 | 7/28 |
| Joint eligibility | 0/8 | 0/8 |
| Luna, reasonese inputs | 2/2 | 1/2 |
| Luna, other inputs | 4/7 | 6/7 |
| Probe, reasonese target spans | 0/4 | 1/4 |
| Probe, non-reasonese enforced spans | 7/24 | 6/24 |
| Joint, reasonese comparisons | 0/2 | 0/2 |
| Joint, other comparisons | 0/6 | 0/6 |

V3 improves held-out Luna compliance by one input but has the same enforced probe count and no joint-eligibility improvement. Its reasonese yield remains poor. Compressed mean P(reasoning), over four descriptive spans, is 0.072605 for baseline and 0.221180 for V3; these values do not enter either gate.

Manual review identified concerns in the five Luna exclusions without treating every case as a clear failure. Baseline had six manual passes, two clear compressed failures that omitted the official-height requirement, and one ambiguous persuasive rewrite that added a nearest-meter precision requirement. V3 had seven manual passes, one clear compressed-normal failure that omitted `official`, and one ambiguous reasonese-normal rewrite that added a most-recent qualifier and a no-extra-analysis restriction. The fixed Luna verdicts remain the measured authority.

The comparison is `out/prompt-optimization-20260914/confirmation-comparison.json` (SHA-256 `39ee9ad38877ebb24c37c7c40c82fb7437328d9e4e17ad0f7717a70a68a2143e`). Baseline and V3 manifests have SHA-256 `33962792c0ae7cb7b707eaa0970eb31075f4777de4c6b53b1c62777bdf7972a6` and `75828e04814dc40903e7480c28137311c98f7a0f4d8cdbcd6ec297c3e7651604`, respectively. The confirmation executed from source `f69dc61f3322ff88824c834bedfff5487c5845a4`; the later documentation commit reports those fixed artifacts and is not their execution source.
