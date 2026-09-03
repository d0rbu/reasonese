# Instruction bank

A study compares two cells inside one conversation, and a trial is informative only when the
assistant clearly completes one instruction and not the other. The instruction bank therefore
consists of **pairs**, not a flat list. Each pair shares an agentic scenario and is written so
that no single response can satisfy both instructions.

## Criteria

Every pair in [`configs/instruction_pairs.yaml`](../../configs/instruction_pairs.yaml) is meant
to satisfy:

| Criterion | Meaning | How it is checked |
|---|---|---|
| Feasible | An ordinary capable assistant can complete each instruction alone with the sandbox tools: standard-library Python, network-free bash, an initially empty workspace, and server-side web search. | LLM audit |
| Agentic | At least one instruction in the pair needs running code or searching rather than answering from memory. Process and source-policy pairs deliberately forbid tools on one side. | LLM audit |
| Right difficulty | Not trivial and not impossible. The audit rates 1 to 5 and the accepted band is 2 to 4. | LLM audit |
| Mutually exclusive | No single response, even one using tools, can fully satisfy both instructions. | Construction plus LLM audit |
| Diverse | Pairs spread across skills and conflict types, without near-duplicate wording across pairs. | Coverage table and lexical-overlap diagnostics |
| Sensible | A person reading the pair agrees it is a fair, natural conflict. | Spot-check section of the report |

The audit is an LLM quality-control judgment, not proof. Exclusivity in particular is asserted by
construction and audited by a model; it is confirmed only when collected trials show both
instructions being completed rarely.

## Conflict taxonomy

| Conflict | Mechanism of exclusivity |
|---|---|
| `output format` | Each instruction requires the entire reply to take an incompatible shape, such as a bare number versus a table. |
| `process` | One requires executing code; the other forbids all tool use. |
| `deliverable` | One requires creating a workspace file; the other forbids creating any file. |
| `tool choice` | Each requires one tool and forbids the other. |
| `source policy` | One requires web search with a citation; the other forbids tools and requires an explicit knowledge-cutoff caveat. |
| `language` | Each requires the entire response in a different language. |
| `length` | A hard maximum and a hard minimum that cannot both hold. |
| `scope` | A single paragraph with a fixed citation count versus a paragraph-free list of many sources. |
| `content` | Contradictory substantive constraints, such as ascending only versus descending only, or exactly one recursive versus exactly one iterative implementation. |
| `register` | An entire response free of technical terms versus one that must contain them. |

Skills name the capability each pair exercises: `python`, `bash`, `web search`, or
`python and web search`.

## Curation flow

```text
candidate pairs -> deterministic diagnostics -> cached LLM audit -> report -> manual spot check
```

`reasonese-curate-instructions` loads the bank, computes coverage and cross-pair lexical overlap,
audits any pair whose exact texts have no cached audit in one judge batch, and writes
`report.md` with an audit table, the overlap table, and every pair's full text for spot-checking.
It exits nonzero when any audited pair fails a criterion. `--no-batch` sends synchronous requests,
which is faster when only a few edited pairs need re-auditing. `--scaffold-user-prompts` creates the
placeholder `prompts/user` directories the manual author needs for every new instruction.

## Open steps

- Run a small empirical pass in which each pair is collected under plain framings and both
  orderings, and drop pairs whose both-completed or neither-completed rate is high. That is the
  real test of exclusivity and of whether the pair is contested at all.
- Write the six manual framings for each instruction that the `user` author will use.
- Include web-search annotations in the judge evidence so that instructions to use or avoid web search are verifiable from the trace rather than from cited URLs alone. This touches judging.py and should follow upstream PR #7.
- Generate additional candidates with a model author from the taxonomy once the audit and the
  empirical pass agree on the hand-written seed.
