# Data Schema

All pipeline records are UTF-8 JSON objects, one per line, with `schema_version = 1`.
Readers reject missing and unknown fields.

## Trial

| Field | Meaning |
|---|---|
| `design_id` | Digest of the complete experiment inputs. |
| `trial_id` | Digest of the rendered identity and counterbalance. |
| `experiment` | Experiment identifier. |
| `pair_id` | Stable unordered condition-pair label. |
| `code_pair_id` | Stable nonce-code pair label. |
| `repetition` | Zero-based complete-design repetition. |
| `first_condition`, `second_condition` | Display-order condition identifiers. |
| `first_target`, `second_target` | Distinct requested nonce codes. |
| `system_prompt`, `user_prompt` | Exact rendered messages sent to a collector. |

## ResponseRecord

```json
{
  "schema_version": 1,
  "trial_id": "...",
  "model_id": "versioned_endpoint_identifier",
  "source": "external",
  "response_text": "KITE"
}
```

`model_id` is a non-empty exact endpoint or weight identifier; `source` is a lowercase
pipeline identifier. `response_text` is raw and may be empty. One file must contain exactly
one model and source for ranking. Provider request metadata and
decoding settings belong in an adjacent run manifest until a versioned manifest schema is
implemented.

The simulator uses the same record with `source="synthetic"`.

## ScoredOutcome

Every outcome preserves trial, model, source, and both condition identifiers.

- `status="decisive"`: `winner`, `loser`, and `matched_target` are present.
- `status="invalid"`: those three fields are JSON `null`.

The scorer never discards invalid records. The estimator verifies that decisive winner and
loser exactly match the trial's condition pair.

## Ranking report

The JSON report records method, provenance, reference condition, trial counts, ridge,
convergence, iterations, and an ordered condition list. Each condition includes centered log
strength, penalized standard error, win probability versus reference, raw wins/losses, and
the number of invalid trials in which it appeared.
