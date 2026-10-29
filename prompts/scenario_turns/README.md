# User-turn source bank

Max can edit `*/investigation/human/{normal,casual,persuasive}.txt` and
`*/implementation/{a,b}/human/{normal,casual,persuasive}.txt`. Each file is one
complete user message. Loading strips outer whitespace once; internal text and
literal braces remain unchanged. File-byte hashes and delivered text are both
bound to the plan. Replace its entire `TODO:` text. There are 108 human slots:
12 tasks × (shared investigation + implementation A + implementation B) × 3 styles.
Start with the implementation slots if the investigation history will stay fixed.

For every slot:

- `source.txt` preserves the original operator-authored user message.
- `authoring-source.txt` is the explicit neutral adaptation used for rewriting.
- `source-provenance.json` records source/checkpoint/adaptation hashes.
- Implementation slots also have `canonical-instruction.txt`, the required policy.
- `kimi/` contains actual Kimi K3.1 Pro rewrites in all three styles, including normal.
- `human/` contains placeholders for a human to author independently.

Use the full assignment in `authoring-source.txt` as the content reference. Keep the
same task, constraints, concrete examples and requested tests. Change the framing,
not the requirement. A and B must remain opposed; a B message must not retain an
A-preferring introduction. Neither style should mention an experiment, model
comparison, or a framing label. `normal` means an ordinary direct request;
`casual` and `persuasive` use the project definitions in
[axes.md](../../docs/research/axes.md). Human messages are attributed to their human
writer, even when a model suggested an edit.

The checked-in Kimi text was written from the **original** source messages and
explicit operator adaptations. It is an authored assignment, not a user message
recorded in a later resample. Fresh native prefixes are separately pinned. Each case’s
`source-binding.json` documents the explicit applicability review and selected
complete descriptor and message/checkpoint hashes; missing or stale bindings fail before inference. Do not assume an old claim about observed work is
true of a new prefix. These adaptations remove obsolete dependency-repair notes;
the exact original text remains inspectable in `source.txt`.

`bank.json` and `variant-audit.json` describe the initial Kimi authoring pass; they
are provenance, not an editable-file cache. Plans hash current file bytes on every
compilation, and live runs require the existing Luna message QA on the exact
selected text. Human TODOs are allowed in offline previews and rejected for live
execution. Editing a file requires recompilation and new QA for the changed text.

See [native Pi experiments](../../docs/reference/native-pi-experiments.md) for
checkpoint semantics, preview commands, QA, and execution. Raw author requests and
responses stay in the ignored evidence bundle, referenced by the handoff receipt.
