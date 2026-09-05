# Analysis plan (proposal)

This document proposes the primary analysis for the pair-restricted design. It is a plan for
discussion, not a description of implemented code, and no empirical result is reported here. The
committed analysis in `reasonese/analysis.py` remains the per-cell Bradley-Terry ranking described
in [`../reference/output.md`](../reference/output.md).

## Question and named contrasts

The question is whether the framing of an instruction, especially the two reasonese framings,
changes the probability that the assistant completes it, holding the task, the channel, the author,
and the competing instruction fixed. Six contrasts are named in advance. Everything else is
exploratory.

| # | Contrast | What it isolates |
|---|---|---|
| 1 | `reasonese-normal` vs `normal` | The headline reasonese effect. |
| 2 | `reasonese-normal` vs `casual` | Trace-likeness beyond terseness, since the casual brief also asks for shorthand. |
| 3 | register × intent | Whether persuasion works differently in reasonese than in plain prose. |
| 4 | register × channel | Whether reasonese is followed differently as a system prompt, a user message, or a README. |
| 5 | self-authored minus same-family other-authored, under reasonese | The role-confusion prediction: a model follows its own trace style more. |
| 6 | `user` vs model authors on `normal`, `casual`, `persuasive` | The only framings where the manual and model authors overlap. |

Framing is recoded for these contrasts as register (plain or reasonese) crossed with intent
(neutral or persuasive), plus `casual` and `subagent` as two further levels. That recoding has the
same degrees of freedom as the six-level factor and changes nothing about the fit; it only makes
contrasts 3 and 4 single parameters.

## Why the per-cell Bradley-Terry cannot be the primary analysis

- Once comparisons are restricted to the two sides of a designed pair, the comparison graph has one
  connected component per pair. Per-cell strengths are then only comparable inside a pair, and no
  single number answers the question above.
- Each trial yields two independent completion verdicts, so it has four outcomes: first only,
  second only, both, neither. The current comparison builder codes both and neither as a 0.5 tie.
  Those are different events. Both-completed means the pair was not exclusive in that trial;
  neither-completed means the assistant refused, failed, or ignored both.
- A Bradley-Terry fit on decisive trials conditions on an outcome that the treatment may itself
  change. If a reasonese framing makes the assistant attempt both instructions, dropping or
  neutralising those trials removes part of the effect under study.
- The judge answers an absolute question per instruction. The natural unit is therefore the judged
  instruction slot, not the contest.

## Primary model: per-slot completion

Each trial contributes two rows, one per judged instruction. The outcome is that slot's completion
boolean. The linear predictor contains:

- a pair term for the side of the pair the slot belongs to, which absorbs task difficulty and is
  identified only as a within-pair difference, which is all that is needed;
- the slot's own framing, channel, and author;
- the competing slot's framing, channel, and author;
- position (first or second in the conversation);
- framing × channel in the base model, because contrast 4 is a prediction of the motivating
  role-confusion account and a main-effects model would average it away;
- assistant main effects, assistant × framing, and author × assistant, with a self-match indicator
  (author equals assistant) and a family-match indicator (both Qwen or both Inkling).

The fit is a penalized logistic regression by Newton's method, a generalisation of the existing
per-cell solver, with sum-to-zero coding and a small L2 penalty used only as a numerical
stabiliser. The existing regularisation-sensitivity table carries over. Only numpy is needed.

Two identifiability notes fix the shape of the fit:

- The fit must pool assistants. An author's rewrite is the same text whichever assistant reads it,
  so within one assistant the self-match indicator is that author's dummy and contrast 5 does not
  exist. Pooled, it is the diagonal of the author × assistant table. The family indicator separates
  "own text" from "own family's style"; contrast 5 is the diagonal minus the same-family
  off-diagonal, and the four per-assistant diagonal deviations are reported alongside it.
- The `user` author writes only three framings. Author × framing is therefore estimable only on
  those three, and contrast 6 is restricted to them. The `user` author is also one writer, so
  contrast 6 is confounded with that writer's style and should be described that way.

The Bradley-Terry ranking stays as a secondary view. On decisive trials the same design matrix
reduces to a covariate Bradley-Terry, which is the symmetric special case, and the existing ranking
output becomes a derived table. The tie coding for that view is reported both ways, 0.5 and
excluded, as a sensitivity check.

## Inference

Trials are not independent. Every trial that uses a cell reuses that cell's single generated text,
rollouts repeat within a trial, and the pairs are the unit over which any claim about instructions
in general must generalise. Intervals are therefore clustered by pair.

- With 24 clusters, percentile resampling of pairs under-covers. The plan is a cluster-robust
  sandwich covariance with a small-sample correction and a t reference with 23 degrees of freedom,
  checked against a wild cluster bootstrap. Clustering by generated message is reported as a
  sensitivity.
- The between-pair spread of contrast 1, from 24 per-pair slopes, is reported as the statistic for
  whether the reasonese effect generalises across tasks. With 24 pairs a fitted variance component
  would be descriptive at best, so it is not the headline.
- The six contrasts are pre-registered; no multiplicity adjustment is applied within them, and
  exploratory results are labelled as such.

## Robustness and heterogeneity

The bank's taxonomy is too sparse for per-type interactions: `output format` has six pairs,
`content` three, `register` one, and every other conflict type two. Skill is confounded with
conflict type. Only coarse contrasts are planned:

- shape conflicts (`output format`, `length`, `scope`, `register`, `language`; 13 pairs) vs tool
  conflicts (`process`, `deliverable`, `tool choice`, `source policy`; 8 pairs);
- `python` vs `web search` pairs;
- an exploratory within-pair asymmetry: whether reasonese favours the terser side of a pair.

The four-way outcome table per pair, with both-completed and neither-completed rates, is the
exclusivity diagnostic and the drop rule for a pilot. A latent mixture over pairs is not planned;
with 24 pairs it is indistinguishable from continuous shrinkage. A mixture over assistant behaviour
("engaged" versus "ignoring") is already represented by keeping the neither-completed rows in the
primary model rather than coding them as ties.

## Collection decisions this raises

These are design choices rather than analysis choices and need agreement before collection.

- **Judge noise.** The judge runs once per instruction at temperature 0.7. One false positive on
  the losing slot turns a decisive trial into both-completed. Judging at temperature 0, or two or
  three judgments with a majority vote, matters more than any tie-coding choice.
- **One text per cell.** Each cell has exactly one generated rewrite reused across all its trials.
  Two or three rewrites per cell would separate the framing effect from that one rewrite's quirks.
- **Pair identity on observation rows.** Rows carry the instruction text but not the pair it belongs
  to. Pair-level clustering needs that field.
- **Position of system prompts.** The conversation preserves matchup order, so a system-prompt cell
  in second position is placed after the user message. That is a legitimate treatment but an unusual
  chat layout, and the existing position diagnostics should be read with it in mind.
- **Identifiability check before the main run.** After a pilot fixes the tie rate, compute the rank
  of the planned design's information matrix and predicted standard-error ratios for the six
  contrasts under the planned sample. This is a check, not an allocation optimiser; stratum quotas in
  the sampler are defined by which axes differ and cannot target level-specific contrasts.

## Prior work

The methods are standard; the novel element is the substantive contrast 5, self- versus
other-authored trace-like text on benign instruction following.

- Paired comparisons with covariates: Springall (1973); Cattelan (2012), *Statistical Science*;
  Turner and Firth (2012), `BradleyTerry2`, *Journal of Statistical Software*; LMArena style
  control (2024); Ameli et al. (2025), ICLR, arXiv 2412.18407.
- Ties: Rao and Kupper (1967); Davidson (1970).
- Few-cluster inference: Cameron, Gelbach and Miller (2008); MacKinnon and Webb (2018); Miller
  (2024), arXiv 2411.00640, on clustering evaluation error bars by question.
- Stimuli as a sampled factor: Judd, Westfall and Kenny (2012).
- Motivation: Ye, Cui and Hadfield-Menell, arXiv 2603.12277, on prompt injection as role
  confusion; Panickssery et al. (2024) on models favouring their own generations in judging.

## Implementation sketch

A design-matrix builder with sum-to-zero coding, a general penalized logistic Newton solver over a
dense matrix, contrast reporting, and a cluster-robust covariance. Roughly a few hundred lines plus
tests to hold the coverage gate. It depends on pair identity being available on observation rows.
