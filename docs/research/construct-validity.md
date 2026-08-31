# Construct Validity

## Operational definition

In the pilot, *neuralese-like* means a human-authored, compressed, delimiter-heavy directive
modeled on observable `zz...` messages published in the METR incident report. The concrete
treatment is:

```text
zzGO_FINAL_{TARGET}_ONLY
```

This definition is deliberately behavioral and surface-level.

## Claims the pilot can support

With adequate live data and controls, the pilot can estimate whether the tested model
selects one observable directive form over another under the frozen prompt protocol. It can
compare effect sizes with authority and consensus framings and identify candidates for more
naturalistic follow-up.

## Claims the pilot cannot support

The treatment is not evidence that:

- compressed text is the language of internal neural activations;
- the public messages are faithful chain-of-thought transcripts;
- a model recognizes `zz_compact` as communication from another agent;
- the model follows it because of in-group loyalty or collective utility;
- compressed text causes chain-of-thought obfuscation; or
- behavior will transfer from nonce-code choice to costly agent action.

Chain-of-thought can be unfaithful, and some reasoning traces can be difficult to interpret.
Those facts motivate careful measurement; they do not license treating every terse or
unreadable string as privileged reasoning.

## Current confounds

The broad ranking intentionally compares realistic bundles, so conditions differ in more
than their nominal label:

- token count and tokenizer segmentation;
- capitalization, underscores, punctuation, and code syntax;
- imperative strength and politeness;
- lexical familiarity from pretraining;
- explicit source claims and numerical consensus;
- semantic naturalness; and
- interaction with the neutral wrapper prompt.

Counterbalancing removes code identity and first/second position as simple explanations, but
does not remove these treatment-level confounds.

## Validation ladder

Before strong construct claims:

1. **Semantic check:** independent annotators confirm that each representation treatment
   requests the same target response.
2. **Comprehension check:** models can paraphrase each directive in a non-conflict setting.
3. **Isolated compliance:** each condition elicits its target when presented alone.
4. **Matched ablations:** change one surface feature at a time.
5. **Tokenizer audit:** report character, word, token, and surprisal differences by model.
6. **Generalization:** vary target codes, wrapper prompts, tasks, and model families.
7. **Ecological replication:** repeat the contrast in a benign agent task with measurable
   personal cost and peer benefit.
8. **Mechanistic follow-up:** only then test source recognition or causal representations in
   open-weight models.

## Naming discipline

Use these phrases in reports:

- *incident-inspired compressed directive* for the checked-in condition;
- *neuralese-like* as informal shorthand with the operational definition nearby; and
- *raw chain of thought* only for data explicitly exposed and identified as such by its
  source.

Avoid *token-space language*, *the model's real thoughts*, or *native neuralese* unless a
separate experiment directly establishes the intended construct.
