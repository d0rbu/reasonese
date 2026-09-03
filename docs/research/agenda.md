# Research agenda

## Current step

The current step turns pairs of four-axis datapoints into ordered two-instruction conversations,
collects assistant traces, and independently judges completion of every input. The assistant
remains matchup metadata rather than an entry axis. Study orchestration balances each cell
over every possible position and supports repeated rollouts. Analysis provides penalized
cell rankings, axis margins, and explicit position and robustness diagnostics.

This foundation specifies:

- which values currently belong to each axis;
- how simple base instructions are configured;
- how every axis combination is enumerated; and
- how those combinations are serialized without ambiguity;
- how model authors generate framed messages;
- how each exact materialized message receives an independent compliance audit;
- how channel treatments become an ordered conversation; and
- how generated messages and raw responses are cached; and
- how one strict completion boolean is collected for each instruction; and
- how both orderings and repeated rollouts become analysis-ready observation rows; and
- how rankings, axis contrasts, order effects, and design diagnostics are reported; and
- how an exhaustive pairing population can be replaced by a seeded axis-stratified,
  degree-aware connected design that still covers every selected cell and counterbalances order.

## Deferred decisions

The repository specifies one initial aggregation and analysis contract but committed source
does not contain an empirical corpus or statistical result. Future work may add hierarchical
models, multiplicity-aware inference, or explicit position-adjusted rankings after empirical
sample sizes and study topology are known.

## Construct questions for later work

- How should the six framing treatments be audited for fidelity and distinctness?
- Which protocol should be used to write and review the manual user-authored variants?
- How well do message-QA verdicts agree with blinded human semantic-equivalence ratings?
- How should model revisions and provider routing be recorded alongside stable display names?
- How much does synthetic file-read history itself influence channel comparisons?
- Should future designs weight rare axis-difference strata more heavily than their prevalence in
  the eligible pairing population?

Those questions should be resolved before interpreting future model behavior.
