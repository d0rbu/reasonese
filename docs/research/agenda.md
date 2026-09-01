# Research agenda

## Current step

The current step turns four-axis datapoints into ordered multi-instruction conversations and
collects assistant traces. The assistant remains matchup metadata rather than an entry axis.

This foundation specifies:

- which values currently belong to each axis;
- how simple base instructions are configured;
- how every axis combination is enumerated; and
- how those combinations are serialized without ambiguity;
- how model authors generate framed messages;
- how channel treatments become an ordered conversation; and
- how generated messages and raw responses are cached.

## Deferred decisions

Later work must specify judging and analysis. The repository contains an execution mechanism,
but committed source does not contain an empirical corpus or statistical result.

## Construct questions for later work

- How should the six framing treatments be audited for fidelity and distinctness?
- How should user-authored variants be elicited without conflating author with framing?
- How should generated text be audited for semantic equivalence to its base instruction?
- How should model revisions and provider routing be recorded alongside stable display names?
- How much does synthetic file-read history itself influence channel comparisons?

Those questions should be resolved before interpreting future model behavior.
