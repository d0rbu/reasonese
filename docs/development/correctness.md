# Correctness

Correctness has three layers: artifact integrity, experimental balance, and numerical honesty.

## Artifact boundaries

- TOML loaders reject missing and unknown keys.
- JSONL readers validate exact schema versions and fields.
- Conditions contain exactly one `{target}` placeholder.
- Response codes are distinct uppercase nonce tokens.
- JSON and JSONL writers use atomic replacement.
- A response set must match the design one-to-one; missing, unknown, and duplicate IDs fail.

`Probability` is a phantom type so invalid rates cannot enter the simulator outside `[0, 1]`.
Dataclass constructors enforce invariants again for programmatic callers.

## Experimental invariants

For every condition pair, code pair, and repetition:

- each condition appears first twice;
- each condition requests each nonce code twice; and
- all four combinations have distinct trial IDs.

Tests assert these properties on the checked-in 728-trial pilot, not only a small fixture.

## Scoring invariants

The scorer trims outer whitespace and then requires exact equality. It never performs
substring matching, case folding, answer extraction, or judge-model repair. Invalid outputs
remain records with both conditions attached so missingness can be audited.

## Numerical invariants

The estimator requires a connected decisive-comparison graph, one model/source per fit, a
positive finite ridge penalty, and pair-consistent winner/loser fields. Scores are centered
after every Newton step. A line search prevents an update from reducing the penalized
objective, and nonconvergence remains visible in the report.

Penalized standard errors are implementation diagnostics, not automatically publication-ready
confidence intervals. Repeated calls to one model endpoint are not independent model samples.
