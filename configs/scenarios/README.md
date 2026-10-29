# Retired authored histories

The eleven remaining synthetic README histories were retired when their tasks were
redesigned. All twelve current pairs in `../scenario_pairs.yaml` now have native Pi
tasks in `../pi_transcripts.json`. The original template files remain in Git history.

Use `../pi_remaining_review.json` to prepare eleven Kimi base transcripts following
only the first instruction. The existing templating system will add opposing
requirements and framing variants later. There are no
active YAML templates here: passing this directory to `--scenarios` is an error,
not a way to collect native transcripts. The template renderer remains supported
for separately reviewed definitions and the synthetic test fixture.

See [the task catalog](../../docs/research/remaining-scenario-pilots.md) and
[collection instructions](../../docs/reference/pi-transcripts.md).
