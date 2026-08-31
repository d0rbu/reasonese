# Architecture

The package implements one directional artifact flow:

```text
experiment TOML -> Trial JSONL -> Response JSONL -> ScoredOutcome JSONL -> ranking JSON
                         ^
                         +-- synthetic demo only
```

| Module | Responsibility |
|---|---|
| `reasonese.config` | Strict experiment and simulator TOML loaders; design digest. |
| `reasonese.schemas` | Versioned dataclasses and record invariants. |
| `reasonese.design` | Complete pair construction, counterbalancing, rendering, and seeded shuffle. |
| `reasonese.simulation` | Explicitly synthetic Bradley-Terry response sampler. |
| `reasonese.scoring` | One-to-one completeness checks and exact response-code scoring. |
| `reasonese.bradley_terry` | Connected-graph validation and penalized ranking fit. |
| `reasonese.io` | Strict JSONL readers and atomic JSON/JSONL replacement. |
| `reasonese.cli` | Thin orchestration for the four offline stages. |

## Boundaries

The package has no provider SDK and performs no network calls. Real responses enter through
the documented `ResponseRecord` boundary. This keeps collection authorization, provider
cost, retry behavior, and endpoint-specific metadata from being hidden inside analysis code.

The simulator shares the response schema but stamps `source="synthetic"`. Downstream reports
preserve this source, preventing the demo from masquerading as live evidence.

## Estimator

For each decisive observation where condition `i` beats `j`, the design row contains `+1`
for `i` and `-1` for `j`. Newton updates maximize the logistic log likelihood with an L2
penalty. Centering selects a readable score origin; the penalty makes separated data finite.

The one-dimensional model may fit poorly when preferences are cyclic or interaction-heavy.
Consumers should inspect the pairwise matrix and residuals before treating the ordering as a
sufficient summary.
