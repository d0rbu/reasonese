# Plan output

`reasonese-plan` writes one JSON object per line with exactly the four experimental axes:

```json
{
  "author": "Inkling Small",
  "channel": "README.md",
  "framing": "reasonese-persuasive",
  "instruction": "Write a Python program that counts the unique words in a text file."
}
```

These are unrendered condition specifications, not generated prompts or model outputs.

## Generated-message cache

The message cache is YAML under a top-level `messages` list. Each record contains the full
four-axis `input`, its generated `content`, and the unmodified OpenRouter `response`. A
user-authored message has a null response because its content comes from the selected manual
variant file. Records are keyed by all four input coordinates. Manual files remain authoritative;
changed manual text replaces its cached message and invalidates traces containing the old text.

## Message-QA cache

The message-QA cache is YAML under a top-level `message_qa` list. Each record contains the full
four-axis input, exact audited content, strict `complies` boolean, concise issue list, and raw Luna
response. A verdict is reusable only for the same datapoint and exact content. A compliant message
must have no issues; a noncompliant message must have at least one. False verdicts remain cached
for diagnosis and block inference rather than causing an automatic reroll.

## Conversation-trace cache

The trace cache is YAML under a top-level `traces` list. Each record contains the complete
`matchup`, the materialized ordered `conversation`, zero or more `tool_steps`, and the unmodified
final assistant `response`. Each tool step retains the raw assistant response that requested the
function call plus every local tool result sent back. This intentionally retains returned
`reasoning`, `reasoning_details`, token usage, web citations, and other provider fields. Records
are keyed by the complete matchup, including assistant and input order.

## Judgment cache

The judgment cache is YAML under a top-level `judgments` list. Each record contains:

- the complete `matchup`;
- a `trace_fingerprint` over the matchup, delivered conversation, local tool steps, and final
  assistant response; and
- one ordered verdict per input, with the four-axis input, exact `completed` boolean, and
  unmodified raw judge response.

The fingerprint prevents reuse after a trace changes. The verdicts are independent rather
than one-hot: `[true, true]`, `[false, false]`, and mixed outcomes are all valid.

## Collected study

`reasonese-collect-data --output DIRECTORY` writes:

- `study.yaml`: the exact assistant, rollout count, and input cells;
- `generated_messages.yaml`: shared materialized instruction cache;
- `message_qa.yaml`: exact message-compliance verdicts and raw QA responses;
- `collection.sqlite3`: `traces` and `judgments` tables keyed by stable trial ID, with complete
  records stored as JSON text; and
- `observations.jsonl`: one flat row per cell verdict.

Each observation contains the cell ID and five coordinates, permutation, rollout, one-based
position, completion boolean, trace fingerprint, and available assistant/judge response IDs.
This is the input to downstream analysis. Cached study traces are reused only while their
user-authored contents still match the selected manual files.

`reasonese-collect-studies --output DIRECTORY` places shared `generated_messages.yaml` and
`message_qa.yaml` files directly under `DIRECTORY`. Every repeated `--study PATH` is collected
under `DIRECTORY/PATH_STEM/` with the same `study.yaml`, `collection.sqlite3`, and
`observations.jsonl` layout above. This lets identical cells reuse the same authored message and
QA verdict while keeping rollout traces and judgments isolated by study.

`reasonese-sample-studies --output PATH` writes one YAML mapping whose `studies` list contains the
selected assistant, rollout count, and input pair for every sampled study. With `p` pairings per
assistant, `a` assistants, and `r` rollouts per permutation, it contains `pa` studies and plans
`2par` assistant trials. The same `p` input pairs are used for each assistant.

When `reasonese-collect-studies --suite PATH` is used, each study is collected below
`DIRECTORY/<study fingerprint>/`. Each child contains `study.yaml` and `observations.jsonl`; all
trace and judgment rows are deduplicated into `DIRECTORY/collection.sqlite3`. The suite root also
receives one combined `observations.jsonl` ready for analysis.

## Analysis directory

`reasonese-analyze --output DIRECTORY` writes:

- `ranking.csv`: the total Bradley–Terry ordering, penalized standard errors, clustered
  bootstrap intervals, and raw completion rates;
- `axis_summary.csv` and `axis_comparisons.csv`: margins and pairwise contrasts for instruction,
  framing, channel, author, and assistant;
- `position_summary.csv`, `cell_position_effects.csv`, and `axis_position_effects.csv`;
- `order_sensitivity.csv`: position-rate ranges and position/outcome correlations;
- `regularization_sensitivity.csv`: scores and ranks under 0.1×, 1×, and 10× L2 penalties;
- `diagnostics.json`: trial integrity, comparison connectivity, per-cell position balance, and
  rank-stability summaries; and
- `report.md`: a readable total ordering, axis table, strongest order effects, and caveats.

Within a trial, completed-versus-incomplete yields a win; equal completion verdicts yield a
0.5 tie. This retains all-true and all-false trials. Bootstrap sampling is clustered by trial.
