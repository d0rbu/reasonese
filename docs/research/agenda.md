# Research agenda

## Current step

The current step turns the two sides of a mutually exclusive instruction pair into ordered
two-instruction conversations, collects assistant traces, and independently judges completion of
every input. The assistant remains matchup metadata rather than an entry axis, and instruction is
a blocking factor rather than a treatment axis. Study orchestration balances each cell over every
possible position and supports repeated rollouts. Analysis provides penalized within-component
cell rankings, framing, channel, and author margins, a cross-validated feature lasso over the
same comparisons, per-pair exclusivity counts, and explicit position and robustness diagnostics.

Luna message QA is the hard authoring gate. Probes are optional diagnostics, off for the pilot;
the operational contract is in the [README](../../README.md#current-foundation).

This foundation specifies:

- which values currently belong to each axis;
- how simple base instructions are configured;
- how candidate instruction pairs are audited against the bank criteria before collection;
- how every axis combination is enumerated; and
- how those combinations are serialized without ambiguity;
- how model authors generate framed messages;
- how each exact materialized message receives an independent compliance audit;
- how channel treatments become an ordered conversation; and
- how generated messages and raw responses are cached; and
- how one strict completion boolean is collected for each instruction; and
- how both orderings and repeated rollouts become analysis-ready observation rows; and
- how rankings, axis contrasts, order effects, and design diagnostics are reported;
- why instruction is a blocking factor rather than a treatment axis, given that no comparison
  crosses a pair boundary; and
- how an exhaustive within-pair pairing population can be replaced by a seeded axis-stratified,
  degree-aware connected design that still covers every selected cell and counterbalances order.

## Deferred decisions

The repository specifies one initial aggregation and analysis contract but committed source
does not contain an empirical corpus or statistical result. Future work may add hierarchical
models, multiplicity-aware inference, or explicit position-adjusted rankings after empirical
sample sizes and study topology are known.

Pooled axis margins average per-component cell scores. The feature lasso is their model-based
counterpart: for each evaluation assistant independently, it absorbs every instruction pair in a
side offset and fits framing, channel, author, all three two-way interactions, and their three-way
interaction under an L1 penalty. Position is handled by the symmetric two-permutation collection
design rather than a model term. A pooled effect across instruction pairs therefore has to survive
a cross-validated penalty rather than average over cell scores. The lasso selects and shrinks; it
does not test. Standard errors for selected contrasts, whether from a post-selection refit or from
stability selection over trial resamples, remain deferred, and a pooled effect can still be driven
by a few pairs, so per-pair spread is worth inspecting alongside both the margin and the lasso.

## Construct questions for later work

- How should the eight framing treatments be audited for fidelity and distinctness?
- Which protocol should be used to write and review the manual user-authored variants?
- How well do message-QA verdicts agree with blinded human semantic-equivalence ratings?
- How should model revisions and provider routing be recorded alongside stable display names?
- How much does synthetic file-read history itself influence channel comparisons?
- Should future designs weight rare axis-difference strata more heavily than their prevalence in
  the eligible pairing population?

Those questions should be resolved before interpreting future model behavior.

## Free-route interpretation and pilot size

Free and paid routes share canonical cell identity, but their configurations and behavior are not
verified equivalent. Before reporting pooled results, compare completion rates on matched cells
with separately authorized free/paid collections; retain failures as well as successful trials.
Use separate output/cache directories or cache reuse can prevent the second route from running.
Comparing author routes requires independently generated author messages; comparing assistant
routes can hold the authored text fixed. The comparison itself is deferred. If behavior differs,
free routes are development-only and pooling is inappropriate. Do not combine duplicate trial IDs
from independent runs into one analysis file; analyze the matched runs separately.

With the current 24-pair bank, selecting all seven authors (including user) and six assistants
would require 3,745,440 exhaustive trials or 207,360 sampled trials at 720 pairings per pair
and one rollout per ordering. The default selects only Nemotron 3.5 Lightning and Gemma 4 31B
for both authoring and assistant evaluation, using registered `:free` routes. This gives 48
conditions per instruction, 96 cells and 1,280 eligible pairings per pair, with a minimum of
95 edges for connectivity. The unchanged 720-pairing default gives mean degree 15 and
69,120 trials across the bank and two assistants. These are design counts, not cost estimates
or evidence that any provider can serve the full pilot; QA, judgments, and search can still incur
charges and require `--allow-paid` for uncached collection.

For a four-pair pilot at 216 comparisons per pair with these two authors and two assistants,
collect 3,456 trials (both orders, one rollout). This pilot is smaller than the bank-wide default.
