# Bounded authoring-brief comparison

`reasonese-optimize-prompt` evaluates one explicitly selected authoring brief through the
authoring stage, the existing GPT-5.6 Luna message-QA stage, and the selected local role probe.
It stops after probe QA. It does not execute an assistant, execute tools, submit response judges,
or write observations. The existing default authoring prompt is the `baseline` candidate; the
repository also defines `reasonese-natural-v1`, a measured candidate that adds a short
reasonese-only reminder to preserve concrete obligations in natural first-person planning prose.
The candidate is never adopted automatically.

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
every pair and each judge, plus overall numerators, denominators, missing records, and compressed
descriptive counts.

The table is evidence for selecting a brief, not a claim that one wording is indistinguishable
from a model's private reasoning or that message QA proves semantic equivalence. Review raw
outputs and failures before changing any production default.

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
