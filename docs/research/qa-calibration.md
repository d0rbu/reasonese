# Message-QA calibration

On September 13, 2026, a Nemotron-only pilot stopped at author-message QA. Manual inspection
found both substantive rewrite failures and questionable objections to compressed wording.
This calibration concerns the author-message gate, not assistant response judgments or research
trial outcomes. The instruction pairs, framing definitions, channels, model routes, sampling,
and generation settings were unchanged.

## Selection

The selected QA clarification checks task meaning and framing independently. It allows
conventional shorthand without relaxing required actions, tool identity, quantities, or output
constraints. It also distinguishes an instruction specifying an answer table from an actual
answer table. The implementation remains one system prompt with the existing strict boolean
and issues schema; there is no new acceptance heuristic or retry policy.

| QA wording | Agreement with assigned labels |
|---|---:|
| Original | 93/95 |
| Selected clarification (v4) | 95/95 |
| Shorter clarification (v5) | 94/95 |

For example, the original rejected a rewrite requiring `print a JSON object` because it did not
repeat the word `single`. Removing the selected prompt's explicit explanation of singular output
specifications reproduced that rejection in v5. The original also rejected `print integer` in a
fresh compressed instruction whose base required a single integer. Negative controls changed
these requirements to multiple outputs, different formats, or permitted forbidden tools.

## Iteration and limitations

The first 32 task-preservation cases and 15 framing/output-role cases did not reveal all the
problems later seen in actual author outputs. Three early rubric revisions introduced errors
without improving the initial set. The original also passed 16 initial held-out cases. We
initially retained it, then reopened calibration when live judgments exposed overly literal
objections to conventional shorthand.

Sixteen additional shorthand contrasts and sixteen fresh confirmation cases brought the set to
95 cases. The selected revision passed all of them. The final sixteen cases were written before
judging v4/v5 on them and were not used to revise either candidate. Earlier held-out cases became
regression cases once further iteration resumed; they are not counted as fresh validation twice.

Labels and rationales were assigned by the coding assistant before the corresponding provider
calls. They are not independently collected human annotations. Each reported judgment is one
observation at the existing temperature, and the small differences do not establish statistical
superiority or prove that the checker is always correct. The conclusion is a bounded engineering
choice among the tested wordings. Manual review remains necessary, especially for ambiguous
compression and partial or analysis-like author outputs.

The authoring comparison retains its original QA scores and separately re-audits saved outputs
under the selected rubric. It does not silently mix rubric versions in one pass-rate comparison.
No cached pilot verdict is overwritten merely because the source prompt changed: restarting
requires an archived cache and explicit re-audit. Actual author outputs containing drafting
analysis remain failures; no heuristic strips that text or substitutes an embedded draft.

Exact candidate prompts, labelled cases, request metadata, provider responses, and intermediate
reviews are local ignored artifacts under `out/prompt-calibration-20260913/`. The relevant QA
jobs are `qa-r0` through `qa-r4`; `qa-r3` evaluates v4 and `qa-r4` evaluates the shorter v5 and
fresh confirmation. These diagnostic artifacts are separate from pilot trial caches.
