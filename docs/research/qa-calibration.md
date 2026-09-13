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
answer table, and requested conversational/delegation cues from discussion of rewriting.
The implementation remains one system prompt with the existing strict boolean
and issues schema; there is no new acceptance heuristic or retry policy.

| QA wording | First 95 cases | Additional 10 cue cases |
|---|---:|---:|
| Original | 93/95 | Not run |
| Shorthand clarification (v4) | 95/95 | 8/10 |
| Shorter clarification (v5) | 94/95 | Not run |
| Selected cue clarification (v6) | 95/95 | 10/10 |

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
95 cases. The v4 revision passed all of them. The final sixteen cases were written before
judging v4/v5 on them and were not used to revise either candidate. Earlier held-out cases became
regression cases once further iteration resumed; they are not counted as fresh validation twice.

Subsequent live review found rejections of requested stylistic cues, including a casual `easy`
sign-off and `Thanks, teammate.` in a delegation. Ten labelled cue contrasts, with negative
controls adding an actual token-usage reporting requirement, exposed two false rejections in v4.
A short clarification distinguishing requested cues from rewriting commentary passed all 105
cases. These last cue cases were targeted diagnostics; the earlier 95 served as regressions for
v6, not a new unseen holdout. No cached verdict is selectively replaced merely because a repeat
judgment passes: changing the selected rubric triggers a complete explicit re-audit.

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
jobs are `qa-r0` through `qa-r5`; `qa-r3` evaluates v4, `qa-r4` evaluates the shorter v5 and
fresh confirmation, and `qa-r5` evaluates the cue clarification v6. These diagnostic artifacts
are separate from pilot trial caches.

## Authoring comparison

The completed wording comparison uses 24 coordinates per candidate: four pilot task pairs,
six framings, and evenly rotated destination channels. Each candidate is scored under the same
selected QA rubric, including failures and unusually long drafting outputs.

| Authoring guidance | QA passes |
|---|---:|
| Original brief | 18/24 |
| Verbose preservation checklist (v1) | 16/24 |
| Shorter preservation checklist (v2) | 19/24 |
| Explicit tool names added (v3) | 11/24 |
| Source, recency, exclusivity, and language qualifiers added (v4) | 17/24 |
| Compact preservation paragraph (v5) | 17/24 |

These results do not show a reliable benefit from adding more guidance. The one-case increase
for v2 is below the protocol's practical improvement threshold of two cases and was not
replicated across the later variants. Some outputs omit execution or tool exclusivity; others
add restrictions or return drafting commentary. Manual review distinguishes those substantive
failures from questionable QA objections. This is a prompt-authoring diagnostic, not a score
of the model's ability to execute the research tasks.

Confirmation on opposite pair sides with shifted channels yielded 14/24 QA passes for the
original and 14/23 returned responses for v5. The remaining v5 request exhausted provider
retries with an upstream timeout; it is recorded separately as a provider failure, not silently
removed from the 24 planned requests or classified as a QA rejection.

A 12-case diagnostic moved the unchanged original authoring brief from the user message into
the system message, adding only a user request to return the rewrite. Both it and the matched
original outputs passed 8/12: two original failures became passes and two passes became
failures. This did not resolve the substantive omissions seen in manual review. These
comparisons use known pilot task families, so they are not blind tests of generalization to
unseen tasks. Temporal and provider variability also limit causal interpretation.

We retain the original authoring prompt. This is the practical plateau among the tested
wordings and message placement, not a claim that prompt authoring is solved or no better
prompt exists. No failed trial is discarded, no author output is heuristically repaired, and
no alternative revision's passing output is selectively inserted into the pilot cache.

All 203 returned author responses from 204 planned requests have selected-rubric QA verdicts
validated with the production parser. Local `final-author-comparison.py` checks exact
coordinate matching for the priority comparison and accounts for the exhausted provider
failure separately. The pilot resumes with original authoring, clarified QA, archived
regeneration history, and the existing bounded regeneration gate.

## Pilot environment correction

Manual review of the first word-count traces found that installed `awk` was unavailable
inside the sandbox: `/usr/bin/awk` resolves through `/etc/alternatives`, which was not mounted.
The runtime now mounts that directory read-only when present. A regression test reproduced
the failure before the fix, and both exact saved failing tool calls succeeded after the fix.
Both affected word-count traces, their judgments, and derived observations were archived and
invalidated, then recollected with the corrected runtime. Their original tool errors are
environment defects and are not presented as clean measurements of model behavior. Six
unaffected smoke trials were reused. The corrected smoke passed an offline replay requiring
eight distinct cached trials, eight matching judgment fingerprints, sixteen observations,
and membership in the full pilot suite.

The run-scoped launcher permits up to eight author regenerations after the earlier two-regeneration
allowance exhausted on the Bash rewrite. Every rejected version is archived; acceptance still
requires the unchanged QA criteria. This conditions authored messages on QA acceptance and is
recorded as an operational retry allowance, not a claim of perfect authoring reliability.

Thirty-eight missing pilot author-cache entries were reused from original-prompt calibration.
Full author-request and QA-request equality was checked for each entry, including model route,
messages, generation settings, and verdict schema. All verdicts were retained (26 passing,
12 failing), and no existing pilot entry was overwritten. This reuse does not select passing
outputs from alternative prompt revisions; failed entries still require regeneration.

CI installs Bubblewrap and `mawk` and enables the user namespaces needed by Bubblewrap on the
disposable Ubuntu runner. All 643 tests then passed with no skips and 98.52% coverage. The
runtime's read-only mounts and network isolation remain enabled. These checks validate code
and cache behavior, not future model compliance or provider availability.
