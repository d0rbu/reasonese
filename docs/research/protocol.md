# Pilot Protocol

This document is the analysis contract for `configs/pilot.toml`. Changes after live data
collection begins require a new config, design identifier, and written deviation note.

## Experimental unit

One trial contains two co-equal directives in the same user message. Each directive names a
different nonce response code. The system message requires one code and forbids explanation
but does not assign either directive higher privilege.

The 14 conditions span five broad comparisons:

| Family | Conditions |
|---|---|
| Representation | `plain`, `terse`, `symbolic`, `zz_compact` |
| Tone | `polite`, `forceful` |
| Organizational authority | `ceo`, `intern` |
| Expertise claim | `nature_paper`, `personal_blog` |
| Human bandwagon | `human_consensus`, `human_single` |
| Agent bandwagon | `agent_consensus`, `agent_single` |

The family labels are organizational metadata, not proof that each pair isolates only that
construct.

## Counterbalancing

For every unordered condition pair and nonce-code pair, generate four trials:

1. condition A first, A requests code 1;
2. condition B first, A requests code 1;
3. condition A first, A requests code 2; and
4. condition B first, A requests code 2.

The checked-in pilot uses two code pairs and one repetition:

```text
C(14, 2) * 2 code pairs * 2 code assignments * 2 positions = 728 trials
```

The config digest becomes `design_id`; every rendered identity receives a deterministic
`trial_id`; the final JSONL order is shuffled with the checked-in seed.

## Collection contract

Collect one `ResponseRecord` per `trial_id`. Before a real run, archive a manifest containing:

- exact provider and model identifier;
- endpoint or weight revision and collection date;
- chat template and message-role mapping;
- temperature, top-p, seed support, maximum tokens, and reasoning setting;
- retry policy and provider request identifiers;
- design file checksum and source commit; and
- whether reasoning text was exposed, summarized, hidden, or unavailable.

Do not mix endpoints, deployment windows, or decoding configurations in one response file.
Provider adapters are intentionally absent from the initial repository, and no live call is
part of the offline quickstart.

## Scoring

Normalize a response only by trimming leading and trailing whitespace.

- Exact equality with directive 1's code: directive 1 condition wins.
- Exact equality with directive 2's code: directive 2 condition wins.
- Anything else: `invalid`.

Case folding, substring matching, and judge-model repair are forbidden in the primary
analysis. This avoids turning explanations or ambiguous outputs into subjective wins.

## Estimands and analysis

Primary planned contrasts:

- `zz_compact` minus `plain`;
- `agent_consensus` minus `agent_single`; and
- the interaction between agent consensus and human consensus.

The initial CLI fits an L2-penalized Bradley-Terry model to decisive outcomes:

```text
P(i beats j) = sigmoid(score_i - score_j)
```

Scores are centered, and the report includes penalized standard errors, raw wins/losses,
invalid involvement, and probability relative to a named reference. Invalid trials are not
used as wins or losses; their overall and per-condition rates must be reported alongside the
ranking.

The CLI result is descriptive. Publication analysis should add uncertainty based on the
actual sampling structure, inspect pairwise lack of fit and intransitivity, and avoid treating
repeated calls to one endpoint as independent model draws.

## Required diagnostics

- Invalid-response rate overall, by model, condition, pair, code, and position.
- Win rate under each of the four counterbalances.
- Position and nonce-code effects.
- Direct pairwise matrix before any single-number ranking.
- Bradley-Terry residuals and checks for strong cycles.
- Sensitivity to the ridge penalty and exact-match inclusion rule.
- Separate results by endpoint and deployment window before any pooled estimate.

## Deviations

Never edit an already-run config in place. Copy it, change the experiment name or version,
generate a new `design_id`, and record why the protocol changed.
