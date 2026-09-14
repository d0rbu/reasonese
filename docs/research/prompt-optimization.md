# Bounded authoring-brief comparison

`reasonese-optimize-prompt` evaluates one explicitly selected authoring brief through the
authoring stage, the existing GPT-5.6 Luna message-QA stage, and the selected local role probe.
It stops after probe QA. It does not execute an assistant, execute tools, submit response judges,
or write observations. The existing default authoring prompt is the `baseline` candidate; the
repository also defines `reasonese-natural-v1`, a measured candidate that adds a short
reasonese-only reminder to preserve concrete obligations in natural first-person planning prose.
It also defines `semantic-preservation-v2`, a measured all-framing candidate that keeps supplied
obligations explicit while leaving optional methods open. Neither candidate is adopted automatically.

Each invocation requires a fresh output directory and one candidate name. The suite must contain
at most 32 one-rollout studies, at most 64 unique inputs, and at most 128 probe scores, and must
cover all eight framings. Every study already supplies both orderings; the evaluator constructs
both and scores both positions. The selected assistant is Nemotron by default. Model authors are
inferred from the suite or selected explicitly with repeated `--author` options, so Nemotron-only
authoring and mixed Nemotron/Gemma authoring are both explicit choices.

The fixed instruction-pair bank is required so every report can bind an exact base instruction to
its pair ID. For a live comparison, run the baseline and candidate separately, using distinct
directories and the same suite, pair bank, probe bundle, route, and manual-message root:

```bash
reasonese-optimize-prompt \
  --pairs configs/instruction_pairs.yaml \
  --suite out/prompt-optimization/suite.yaml \
  --role-probes out/role-probes/nemotron-bundle.json \
  --brief baseline \
  --output out/prompt-optimization/baseline \
  --author "Nemotron 3.5 Lightning" \
  --assistant "Nemotron 3.5 Lightning" \
  --allow-paid

reasonese-optimize-prompt \
  --pairs configs/instruction_pairs.yaml \
  --suite out/prompt-optimization/suite.yaml \
  --role-probes out/role-probes/nemotron-bundle.json \
  --brief reasonese-natural-v1 \
  --output out/prompt-optimization/reasonese-natural-v1 \
  --author "Nemotron 3.5 Lightning" \
  --assistant "Nemotron 3.5 Lightning" \
  --allow-paid

reasonese-compare-prompts \
  --baseline out/prompt-optimization/baseline \
  --candidate out/prompt-optimization/reasonese-natural-v1 \
  --output out/prompt-optimization/comparison.json
```

The optimization command prefers batch transport for both authoring and message QA. Pass
`--no-batch` to use synchronous transport for both stages. Free author routes such as the
registered Nemotron route are synchronous regardless of this flag; in that case the flag
selects synchronous transport for the chargeable Luna message-QA requests.

`--allow-paid` is mandatory before any uncached work because Luna message QA is chargeable even when
the model author and local probe use free routes. A provider failure is written to
`failures.jsonl`; successful author and QA responses remain in the generated-message and QA YAML
caches, including their raw provider payloads. A failed run is not reusable as a completed
candidate and its output directory is not silently resumed.

`manifest.json` binds the candidate text and fingerprint, exact author requests, pair-bank and
suite digests and study fingerprints, probe configuration and probe artifact digests, fixed QA
rubric digest, route preference and batch choice, hard work counts, and the fact that assistant
execution, response judging, and tool calls were not performed. The two reports preserve separate
denominators. Message-QA pass counts are per unique authored input. Probe counts are per rendered span across both orders;
compressed framings are descriptive and excluded from the enforced probe denominator. A study
with a message-QA failure can still receive probe scores; combined eligibility is reported
separately. Each study must keep both instructions in one declared pair. The comparison preserves
raw compressed framing reasoning probabilities and explicit per-pair descriptive counts without
putting them in the pass denominator. `reasonese-compare-prompts` emits a before/after table for
every pair and each judge, with passed/eligible fractions and pass rates, plus overall numerators,
denominators, missing records, and compressed descriptive counts.

The table is evidence for selecting a brief, not a claim that one wording is indistinguishable
from a model's private reasoning or that message QA proves semantic equivalence. Review raw
outputs and failures before changing any production default.

The completed first development comparison rejected `reasonese-natural-v1`: the baseline had
16/27 Luna passes, 46/84 enforced probe passes, and 1/24 jointly eligible studies, while v1 had
15/27, 42/84, and 1/24. Across the six reasonese-framing development cells, Luna-only eligibility
fell from 3/6 to 1/6. That v1 decision is retained as a historical report. The follow-up measured
`semantic-preservation-v2`: the baseline had 16/27 Luna passes, 46/84 enforced probe passes, and
1/24 jointly eligible studies, while v2 had 21/27, 38/84, and 1/24. Across the six
reasonese-framing cells, v2 had 3/6 Luna passes, matching the baseline. Luna's v2 result is
confirmed, while the probe-based comparison and any selection based on it remain provisional
until the saved spans are re-scored with segment-prefix capture; no candidate is adopted and no
pilot has been launched. Some manual objections involving the official CPython source and release
date remain ambiguous and are not treated as plainly confirmed failures. The original contextual
probe scores remain under re-score rather than evidence of a causal prompt or channel effect.

The completed integrity diagnosis reproduced the repeated input exactly, but a same-length
future-suffix control changed P(reasoning) from 0.155964352 to 0.209339758, with the first
difference after MoE layer 1. A controlled expert-45 check using identical prefix inputs and
routers changed the BF16 variable batch from 13 to 18 and produced an upward delta of 0.001953125;
isolated-row and strict-reduction checks matched exactly. For this case, the evidence supports
numerical batch sensitivity rather than logical future-token access. Segment-prefix capture uses
one forward ending at each segment boundary. It removes dependence on external future suffixes,
without claiming strict per-token causality within the segment.

The fixed-parameter 298-forward integrity re-screen passed the existing native gate without fit,
recalibration, or threshold selection. It retained the historical qualified NPZ and frozen CAL
threshold 0.16399151054665906.

| Dataset | Segment AUC, full context → prefix | Reasoning above threshold | Final below threshold |
|---|---:|---:|---:|
| Native CAL, descriptive | 1.0000 → 1.0000 | 12/12 → 12/12 | 12/12 → 12/12 |
| Native TEST, integrity gate | 1.0000 → 1.0000 | 11/12 → 11/12 | 12/12 → 12/12 |
| Neutral TEST, diagnostic | 0.9716 → 0.9732 | 50/50 → 50/50 | 29/50 → 30/50 |

Native TEST bootstrap AUC remained 1.0 with 95% interval [1.0, 1.0]. Because TEST was previously
exposed, this is an integrity re-screen rather than a new untouched qualification. The neutral
30/50 final specificity remains a diagnostic limitation. The v1 and v2 probe tables retain their
original full-context measurements and remain provisional until corrected re-score. The report is
`out/prompt-optimization-20260914/segment-prefix-integrity-rescreen.json` (SHA-256
`cb6d7044a71479dd91017fa6c56296ebebe51e2237a455f7b1e957dae5133f99`); its receipt binds launcher
SHA-256 `021826d35143e0be1c191c3c211a72e03ce017bf1e4d93e1d91f8833510ea186` to source commit
`bacec0ea727a40834907c31727ad8b2e2d8d2d92`.

The [measurement report](prompt-optimization-results.md) records completed comparisons,
selection decisions, diagnostic limitations, and artifact provenance.

## Frozen live design

The live comparison uses three development pairs—`prime-1234-bare-vs-table`,
`word-counts-bash-vs-python`, and `cpython-version-search-vs-memory`—plus
`everest-feet-spanish-vs-english` as a reserved confirmation pair. Each pair contributes one
study for each of the eight target framings. The target channel rotates by pair and framing index;
each study uses the fixed normal, user-channel opposite-side anchor. Both input orders are scored.
This is a diagnostic comparison rather than a balanced factorial experiment.

Each candidate version takes one stochastic author sample for each specification. The 32 studies
produce 36 unique authored inputs per version, 128 probe spans, 112 enforced probe scores, and 16
compressed descriptive scores. Specifications with unchanged non-reasonese framing or anchor
guidance are independently resampled for each version, so those control texts may differ between
versions. The comparison therefore does not establish a causal effect from a paired control.

The existing Luna message judge and the adopted Nemotron role probe are fixed across versions.
Selection uses only the three development pairs: retain the baseline unless the candidate improves
joint LLM/probe eligibility without losing LLM semantic compliance, with every rejection inspected.
The Everest confirmation pair is reported after that choice and cannot retune it. The comparison
does not refit the probe; Gemma probe qualification remains pending, and the assistant during this
comparison is Nemotron 3.5 Lightning.
