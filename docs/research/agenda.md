# Research agenda

## Current step

The current step establishes the vocabulary and software representation for four axes:
instruction, framing, channel, and author. Its output is a deterministic list of intended
prompt specifications.

This foundation answers only:

- which values currently belong to each axis;
- how simple base instructions are configured;
- how every axis combination is enumerated; and
- how those combinations are serialized without ambiguity.

## Deferred decisions

Later work may specify how prompts are written, how model authors are invoked, how channel
environments are rendered, which executor models receive the prompts, how responses are
collected, and how results are analyzed. None of those procedures is selected or implied by
this foundation.

In particular, the repository currently contains no generated prompt corpus, provider
adapter, response artifact, empirical observation, or statistical result.

## Construct questions for later work

- What reproducible rubric distinguishes casual, persuasive, subagent, and reasonese text?
- How should user-authored variants be elicited without conflating author with framing?
- How should generated text be audited for semantic equivalence to its base instruction?
- What metadata will uniquely identify model authors and their decoding settings?
- What wrapper content is necessary to make the three channels comparable?

Those questions should be resolved before interpreting future model behavior.
