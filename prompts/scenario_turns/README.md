# User-turn source bank

Max can edit `*/investigation/human/{normal,casual,persuasive}.txt` and
`*/implementation/{a,b}/human/{normal,casual,persuasive}.txt`. Each file is one
complete user message. Loading strips outer whitespace once; internal text and
literal braces remain unchanged. File-byte hashes and delivered text are both
bound to the plan. Replace its entire `TODO:` text. There are 108 human slots:
12 tasks × (shared investigation + implementation A + implementation B) × 3 styles.
The primary editing task is the 72 implementation messages. The 36 investigation
messages are optional when keeping the source investigation history fixed.

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

The initial Kimi text was written from the **original** source messages and
explicit operator adaptations. Later Kimi rewrites align task details, investigation
stop boundaries and claims about prior work with the selected source; their
provenance records each adaptation and author request. These are authored assignments, not user messages
recorded in the source trajectory. Fresh native prefixes are separately pinned. Each case’s
`source-binding.json` documents the explicit applicability review and selected
complete descriptor and message/checkpoint hashes; missing or stale bindings fail before inference. Do not assume an old claim about observed work is
true of a new prefix. These adaptations remove obsolete dependency-repair notes;
the exact original text remains inspectable in `source.txt`.

The historical `sourceCheckpointSha256` in `source-provenance.json` hashes the
original native-message prefix using `JSON.stringify`; it is not a workspace
archive hash. The selected source's filesystem archive hashes are the
`target.checkpoints` values in `source-binding.json`.

`bank.json` and `variant-audit.json` describe the initial Kimi authoring pass; they
are provenance, not an editable-file cache. Plans hash current file bytes on every
compilation, and live runs require the existing Luna message QA on the exact
selected text. Human TODOs are allowed in offline previews and rejected for live
execution. Editing a file requires recompilation and new QA for the changed text.

See [native Pi experiments](../../docs/reference/native-pi-experiments.md) for
checkpoint semantics, preview commands, QA, and execution. Raw author requests and
responses stay in the ignored evidence bundle, referenced by the handoff receipt.
The [handoff release](https://github.com/d0rbu/reasonese/releases/tag/pi-native-corpus-20261029)
provides downloadable source context, authoring evidence and review reports.
