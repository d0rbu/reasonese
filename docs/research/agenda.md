# Research Agenda

## Core question

When co-equal directives conflict, how strongly does each directive's observable form
affect which one a language-model agent follows?

The project begins with representation and social-framing effects in a one-turn choice. It
then moves toward the motivating question: whether peer-agent communication can override an
agent's own task objective, especially when the peer request uses compressed language.

## Pilot hypotheses

The pilot freezes three confirmatory contrasts and one exploratory analysis before any live
model data are collected:

1. `zz_compact` versus `plain`: incident-inspired compressed text has a different pairwise
   win probability than short grammatical English.
2. `agent_consensus` versus `agent_single`: claimed agreement among agents changes
   instruction selection relative to one agent's recommendation.
3. The agent-consensus contrast differs from the analogous human-consensus contrast.
4. Exploratory: fit a Bradley-Terry ranking across all 14 conditions.

These are behavioral hypotheses. Even a large `zz_compact` effect would not show that the
model interpreted the text as private chain of thought, recognized an in-group language, or
used the same computation it uses while reasoning.

## Research phases

### Phase 0: pipeline validation — complete

- Strict TOML and JSONL boundaries.
- Deterministic, fully counterbalanced design generation.
- Synthetic response backend with known latent strengths.
- Exact-match scoring that retains invalid outputs.
- Penalized Bradley-Terry estimation and diagnostics.
- Offline test, lint, and type-check gates.

Completion here is engineering evidence only. It does not validate the treatments or make
an empirical claim about any model.

### Phase 1: controlled choice pilot — not run

- Freeze the checked-in treatment templates and trial design.
- Validate semantic equivalence of the representation-family directives with humans and at
  least one independent model-based check.
- Select a small, versioned set of model endpoints and decoding settings.
- Run the exact 728-trial design independently for each endpoint.
- Report pairwise counts, invalid rates, the three planned contrasts, and the descriptive
  ranking with uncertainty that respects repeated prompts and model clustering.

### Phase 2: matched-ablation study

The current conditions intentionally cover a broad design space but differ in length,
punctuation, lexical choice, and claims. Build matched treatment ladders that alter one
feature at a time:

- compression without `zz`;
- `zz` prefix without compression;
- underscore/capitalization changes without social content;
- matched token length and imperative force;
- source claims with identical syntax;
- human versus agent consensus with identical numbers and wording.

This phase is necessary before attributing an effect to a specific construct.

### Phase 3: benign agentic conflict

Construct sandboxed tasks in which an agent has a measurable personal task objective and
receives a peer request that trades off against it. Costs should be simulated in an inert
environment: time budget, points, or reversible state changes rather than security exploits
or destructive actions.

Primary outcomes should separate:

- whether the peer directive is acknowledged;
- whether the agent takes the requested action;
- cost to its assigned task;
- claimed reason for acting, treated as a report rather than ground truth; and
- whether an external policy monitor would have blocked the action.

### Phase 4: representation provenance and mechanism

Only after robust behavioral effects:

- compare hand-authored compressed treatments with messages naturally produced by agents;
- measure legibility and tokenization statistics;
- use open-weight models for stationary, inspectable anchors;
- test causal interventions or activation-level correlates where justified; and
- distinguish source recognition from behavioral enforcement.

## Non-goals

- Claiming access to hidden reasoning from ordinary API outputs.
- Treating public chain-of-thought snippets as a universal model language.
- Interpreting one endpoint snapshot as a stable property of a model family.
- Using a ranking as evidence of intentional persuasion, altruism, deception, or agency.
- Reproducing the security conditions of the motivating incident.

## Workshop-paper threshold

A credible short paper should include multiple model families, exact endpoint provenance,
the planned contrasts, matched ablations, per-pair residual diagnostics, and a benign
agentic replication. A ranking from the current code-choice pilot alone is a useful pilot,
not yet the full claim.
