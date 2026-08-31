# Experiment Lifecycle

## 1. Proposal

Write the construct, planned contrast, expected direction if any, and disconfirming result.
Separate confirmatory tests from exploratory rankings.

## 2. Design freeze

Validate treatment meaning, render the JSONL, save `design_id`, record the source commit,
and inspect all four counterbalances. Presence of a design is not collection.

## 3. Collection preflight

Freeze endpoint provenance, decoding settings, retry policy, budget, and artifact paths.
Verify the response schema with a non-billable or explicitly authorized minimal call. A
passing preflight is not a completed run.

## 4. Collection

Write raw responses and provider metadata before transformation. Keep endpoint windows
separate, and do not overwrite partial artifacts. Record failures and retries.

## 5. Scoring

Require a complete one-to-one trial mapping. Produce outcomes with exact primary scoring,
inspect invalid responses, and checksum both raw and scored artifacts.

## 6. Analysis

Report direct pairwise counts and nuisance diagnostics before the Bradley-Terry ranking.
Separate synthetic pipeline checks from live model data and planned from exploratory tests.

## 7. Interpretation

Match every claim to the evidence level: surface-form choice, matched treatment effect,
agentic transfer, or mechanistic evidence. Do not skip levels because a ranking is striking.

## 8. Archive

Keep configs, manifests, checksums, code commit, analysis environment, and a deviation log.
Generated bulk artifacts remain outside git unless a deliberate release process approves them.
