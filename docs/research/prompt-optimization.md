# Bounded authoring-brief comparison

## Current policy — September 16, 2026

The policy below records a completed comparison under the earlier joint-eligibility rule. Its
scores and counts remain historical measurements. Current runs use Luna high-reasoning message QA
as the hard gate and optional probe diagnostics; `reasonese-optimize-prompt` defaults to
probe-off, inline results do not affect eligibility, and comparisons reject mixed policies. See
the [README](../../README.md#current-foundation) for the operational probe contract. This update
changed no probe fit, threshold, or prompt.

`reasonese-optimize-prompt` evaluates one authoring brief through model authoring and GPT-5.6 Luna
message QA, then optionally records local role-probe diagnostics. Probe scoring is off by default.
The command stops before assistant execution, tool use, response judging, and observation writing.
The repository defines the unchanged `baseline` plus three measured candidates:
`reasonese-natural-v1`, `semantic-preservation-v2`, and `constraint-scope-v3`.

The historical comparison used Luna semantic preservation per unique authored input and Nemotron
probe checks for each rendered span in both orders; compressed framings were descriptive. A
comparison was jointly eligible only when both historical gates passed. Neither the probe nor its
thresholds were refit or recalibrated.

## Selected brief

The rule recorded at the time retained baseline unless a candidate improved joint Luna/probe
eligibility on the development pairs without losing Luna compliance. V3 met that historical rule:

| Brief | Luna semantic compliance | Enforced probe compliance | Joint eligibility |
|---|---:|---:|---:|
| baseline | 16/27 | 50/84 | 2/24 |
| reasonese-natural-v1 | 15/27 | 42/84 | 1/24 |
| semantic-preservation-v2 | 21/27 | 44/84 | 1/24 |
| **constraint-scope-v3** | **22/27** | **47/84** | **3/24** |

V3 was therefore selected for the reserved confirmation and subsequent pilot protocol. This is a
small dependent DEV comparison with one stochastic author sample per input; it does not reliably
isolate a prompt effect or establish general superiority. V3's six reasonese comparisons were
jointly eligible in 0/6 cases under that earlier rule.

At that artifact snapshot, no pilot had launched and Gemma probe qualification remained pending.

The selected brief asks the author to preserve every obligation and its scope, keep examples optional, change voice without adding methods or deliverables, preserve supplied text, quantities, language, and output format, and return only the destination instruction. The baseline remains the application default; selection is explicit at the collection boundary.

## Reproducible comparison contract

Each invocation requires a fresh output directory and one registered candidate. A run is limited to 32 one-rollout studies, 64 unique inputs, and 128 probe scores, and must cover all eight framings. Both instruction orders and both positions are scored. The manifest binds the candidate text and fingerprint, authoring requests, pair bank, suite, role-probe artifacts and capture policy, Luna rubric, routing, and expected work counts.

```bash
reasonese-optimize-prompt \
  --pairs configs/instruction_pairs.yaml \
  --suite out/prompt-optimization/suite.yaml \
  --probe-mode inline \
  --role-probes out/role-probes/nemotron-bundle.json \
  --brief constraint-scope-v3 \
  --output out/prompt-optimization/constraint-scope-v3 \
  --author "Nemotron 3.5 Lightning" \
  --assistant "Nemotron 3.5 Lightning" \
  --allow-paid

reasonese-compare-prompts \
  --baseline out/prompt-optimization/baseline \
  --candidate out/prompt-optimization/constraint-scope-v3 \
  --output out/prompt-optimization/comparison.json
```

The selected assistant defaults to Nemotron. Model authors are inferred from the suite unless repeated `--author` options select them explicitly. The command prefers batch transport for authoring and message QA; `--no-batch` selects synchronous transport for both stages. Free Nemotron authoring remains synchronous either way, while the flag controls whether chargeable Luna QA uses batch transport. `--allow-paid` is required before uncached Luna work. Failures remain recorded and a failed candidate directory is not silently resumed.

Luna is counted once per unique authored input and the probe per rendered span. A Luna-rejected
input can still receive diagnostic probe scores; the optimization command never runs the studied
assistant.

## Probe capture integrity

Each scored segment receives one prefix forward ending at its final scored token. For the pinned
Nemotron CUDA runtime, `segment_prefix_capture_v2_nemotron_cumsum_h1` scopes the registered Triton
cumsum configuration to `BLOCK_SIZE_H=1` and restores autotuner state afterward. When inline
diagnostics are enabled, preflight validates the registered configuration before the local forward.
H1 was chosen because it exactly reproduces the historical saved activations, not because it
improved QA results. The frozen layer-13 lambda-100 NPZ, CAL-selected threshold
`0.16399151054665906`, and qualification provenance are unchanged. The capture-policy identifier
invalidates earlier autotuned cache records and comparison identities.

Two fresh H1 processes agreed exactly on all 15 controls and all 60 stored NPZ fields. Nine DEV controls also matched their historical saved scores exactly. The six native CAL controls differed from their historical scores by at most 0.0029773168680727324, with no threshold-decision flips. The fixed-parameter 298-forward integrity screen then passed the existing native gates: CAL was 12/12 reasoning and 12/12 final descriptively; previously exposed native TEST was 11/12 reasoning, 12/12 final, AUC 1.0, and paired-bootstrap 95% interval [1.0, 1.0]. Neutral TEST remained diagnostic at 50/50 reasoning and 30/50 final specificity, with AUC 0.9728.

The diagnosis traced the earlier cross-process discrepancy to Triton's autotuned cumsum head tile. H1 reproduced the historical first-anchor activation; H4 through H64 produced the alternate stable value. Fixing H1 removes this execution-choice ambiguity for the supported runtime. Segment-prefix capture removes external future-suffix dependence but does not claim strict token-by-token causality within a segment.

## Frozen study design

Development uses three pairs: CPython search versus memory, prime output-format conflict, and Bash versus Python tool choice. Each pair contributes all eight framings, with a fixed normal opposite-side anchor and both orders. The reserved Everest language-conflict pair is disjoint confirmation evidence and cannot retune the selection.

The [measurement report](prompt-optimization-results.md) contains per-pair, per-judge,
framing-group, historical joint, and compressed results with artifact provenance. Historical
full-context and earlier autotuned segment-prefix artifacts remain available as diagnostic
provenance; the H1 report records the selection evidence from the earlier joint rule.

## Reserved confirmation

The confirmation was launched only after the V3 selection was recorded. On the held-out Everest pair, Luna compliance was 6/9 for baseline and 7/9 for V3. Both had 7/28 enforced probe passes and 0/8 jointly eligible comparisons; each also had four compressed descriptive spans. The held-out sample therefore does not reproduce the DEV joint-eligibility increase. Reasonese yield remains poor: baseline and V3 were each 0/2 jointly eligible, while Luna was 2/2 versus 1/2 and the probe was 0/4 versus 1/4. These post-selection results do not retune the brief, gates, probe, or threshold.

The confirmation comparison is `out/prompt-optimization-20260914/confirmation-comparison.json` (SHA-256 `39ee9ad38877ebb24c37c7c40c82fb7437328d9e4e17ad0f7717a70a68a2143e`). It ran with measurement source `f69dc61f3322ff88824c834bedfff5487c5845a4`; the later documentation commit records the result without changing that execution provenance.

Manual review identified concerns in the five held-out Luna rejections, including two ambiguous cases. Baseline had six manual passes, two clear compressed failures that omitted the official-height requirement, and one ambiguous persuasive rewrite that added a nearest-meter precision requirement. V3 had seven manual passes, one clear compressed-normal failure that omitted `official`, and one ambiguous reasonese-normal rewrite that added a most-recent qualifier and a no-extra-analysis restriction. These manual labels preserve the caveats without changing the fixed Luna verdicts.
