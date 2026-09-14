# Prompt optimization measurements

## First development comparison: candidate rejected

On 2026-09-14, `reasonese-natural-v1` failed the predeclared development selection rule.
Luna semantic compliance fell from 16/27 to 15/27, enforced probe compliance fell from
46/84 to 42/84, and combined comparison eligibility stayed at 1/24. The baseline remains
selected while a second, semantic-preservation reminder is evaluated. These are authoring/QA
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

## Compressed scores remain descriptive

Both compressed framings contribute to the following means. No direction is required and none
of these scores enters the enforced probe denominator. Raw per-span scores, framing, channel,
order, and position remain in the probe reports.

| Instruction pair | Baseline mean P(reasoning) | v1 mean P(reasoning) | Spans per version |
|---|---:|---:|---:|
| prime-1234-bare-vs-table | 0.172 | 0.181 | 4 |
| word-counts-bash-vs-python | 0.078 | 0.238 | 4 |
| cpython-version-search-vs-memory | 0.320 | 0.289 | 4 |

## Context and channel diagnostics

At baseline, nonreasoning system-channel spans passed 0/8, compared with 7/8 for README spans.
Reasonese system-channel spans passed 4/4, compared with 0/4 for README spans. These are small,
unbalanced diagnostic groups with different texts; they cannot isolate a causal channel effect.
Ordinary anchor spans account for 19 of the 38 baseline probe failures. Anchor probe failures
exclude 16/24 comparisons, while target probe failures exclude 12/24, with overlap. The frozen
probe's behavior in these instruction contexts therefore remains a material interpretation limit.
The threshold and all measured verdicts are retained without retuning.

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
