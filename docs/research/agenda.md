# Research agenda

## Current step

The current step turns four-axis datapoints into ordered multi-instruction conversations,
collects assistant traces, and independently judges completion of every input. The assistant
remains matchup metadata rather than an entry axis. Study orchestration balances each cell
over every possible position and supports repeated rollouts.

This foundation specifies:

- which values currently belong to each axis;
- how simple base instructions are configured;
- how every axis combination is enumerated; and
- how those combinations are serialized without ambiguity;
- how model authors generate framed messages;
- how channel treatments become an ordered conversation; and
- how generated messages and raw responses are cached; and
- how one strict completion boolean is collected for each instruction; and
- how permutations and repeated rollouts become analysis-ready observation rows.

## Deferred decisions

Later work must specify aggregation and statistical analysis. The repository contains
execution and judging mechanisms, but committed source does not contain an empirical corpus or
statistical result.

## Construct questions for later work

- How should the six framing treatments be audited for fidelity and distinctness?
- Which protocol should be used to write and review the manual user-authored variants?
- How should generated text be audited for semantic equivalence to its base instruction?
- How should model revisions and provider routing be recorded alongside stable display names?
- How much does synthetic file-read history itself influence channel comparisons?

Those questions should be resolved before interpreting future model behavior.
