# Message workshop

A small, server-backed editor for the twelve native Pi tasks. It shows the genuine
transcript, a writing guide, the selected A/B requirement, and
normal/casual/persuasive style guidance. Human writers start with an empty editor;
Kimi examples are not offered as draft text. Investigation variants are optional.
There are 72 primary implementation entries and 36 optional investigation entries.

Run against an existing complete review-site export (the same `data.json` files
and `/kimi-transcripts/catalog.json` used by the
[static viewer](../pi-transcripts/viewer/README.md)):

```bash
uv sync --locked
uv run python tools/pi-authoring/server.py \
  --site out/transcript-site --state out/message-workshop --port 8000
```

This command makes no model requests and serves a read-only preview. To enable the
user-authorized paid submission checks, set `OPENROUTER_API_KEY` in the server's
environment and add `--allow-paid`. The exact route is
`openai/gpt-6.1-sol`, with `reasoning.effort=high`, required structured outputs and
no model fallback. No credentials are sent to the browser. The default limit is
500 new checks per state directory, counting failed checks too; use
`--max-checks N` for a smaller authorized campaign. Two workers run checks, with at
most eight pending jobs. A failed request needs an explicit resubmission; there is
no automatic paid retry. Repeating identical text reuses a completed check.

The server binds only to loopback. Forward that port through the existing
Cloudflare tunnel. The regular `/`, `/kimi-json-recovery/` and `/kimi-transcripts/`
URLs show the workshop. Startup verifies each original `evidence/messages.json`
against the reviewed source binding and catalog, and requires the displayed
messages to equal that native evidence. Other existing site files remain available, including
original transcript JSON, evidence and the original cohort archive. The site tree
is never modified.

Open `https://HOST/kimi-json-recovery/#access=TOKEN`, replacing `TOKEN` with the
contents of `out/message-workshop/access-token`. Share that authoring link
privately with the writer. The fragment is removed from browser history after it
is stored for the tab; it is not sent as a URL to the server. Possession permits
reading drafts and saving/checking/exporting all entries. Regular viewer links
cannot do those things. The token has file mode 0600; keep the state directory and
its SQLite database, requests, responses and drafts private. Rotate access by
stopping the server, removing only `access-token` and restarting with the same
inputs; saved entries survive. Do not place a capability link in Git or a public
release.

## What the judge checks

`configs/pi_authoring_rubrics.json` contains 36 independently written content
rubrics: investigation and implementation A/B for each task. Each is bound to the
neutral authoring source, canonical policy and selected source binding. The server
composes each with the selected style and assignment-integrity checks, yielding
108 message rubrics. The rubric stays behind the scenes. The selected requirement and style brief
are shown beside the editor; the original user turn is highlighted and scrolled
into view in the unchanged transcript.
Changing a bound source requires explicitly reviewing/updating its rubric.

The judge gets the candidate as quoted data, the full reference assignment, its
checklist, and the recorded investigation's user/assistant visible text. It does
not receive the assistant's hidden reasoning, tool output or later implementation.
Investigation assignments have no preceding investigation context. The judge asks
whether the message communicates enough for a competent coding assistant to carry
out the task, accepting paraphrases and context-dependent references. It must not
require word-for-word copies, an implementation recipe, extra work, elegant prose,
forced slang or fixed punctuation/lowercase counts. Style is assessed separately
from task sufficiency. The judge provides feedback and never generates replacement
human text. Automated semantic checks remain fallible; this is submission QA, not
a measurement of later assistant behavior.

Every required check must occur exactly once and pass. An incomplete, malformed,
wrong-model or failed response cannot accept an entry. Drafts and raw responses
survive failures; a failed/rejected revision leaves the last accepted text intact.
Only one distinct revision of a given message can be checked at once, preventing
late jobs from replacing newer accepted text. Different messages can run in
parallel. Accepted text is stored exactly as submitted, including UTF-8 and
whitespace. The existing planner separately strips outer whitespace on delivery.

## Files and integration

All writable state belongs under ignored `out/`:

- `bank/` is a complete copy of the seed variant bank, with accepted human files
  replaced atomically. Seed files in the repository remain unchanged.
- `entries.sqlite3` holds drafts, jobs, author names, verdicts and accepted entries.
- `checks/JOB/request.json`, `response.json` or `error.json` retain exact request
  evidence and parsed provider response. API keys are not included.
- `inputs.json` pins the seed bank, rubrics, source data and judge contract. A
  different input set requires a fresh state directory; restart repairs accepted
  bank files from the database and marks interrupted checks as failed.

Point the existing planner at that bank:

```bash
uv run reasonese-native-plan \
  --corpus out/CORPUS --variants out/message-workshop/bank \
  --pilot-grid --allow-placeholders --output out/workshop-previews
```

This is an offline preview; unfilled human entries remain placeholders. For one
accepted implementation message:

```bash
uv run reasonese-native-turn \
  --source out/CORPUS/langchain-json-recovery/template.json \
  --variants out/message-workshop/bank --turn 2 --slot implementation --side a \
  --author user --framing normal --output out/workshop-replacement.json
```

“Download edits” downloads accepted text as a patch targeting
`prompts/scenario_turns/`. Apply it in a checkout of the original seed tree with
`git apply --check human-messages.patch`, then `git apply human-messages.patch`.
The authenticated `GET /api/export` additionally downloads the entire bank with
submission metadata, authors and judge verdicts. Keep the server state as the full
audit record. No placeholder is marked accepted and no model-written qualification
fixture is installed in the human bank.

Submission does not start a trajectory or a framing experiment. The existing
native runner's independent Luna exact-text QA gate remains required before live
experimental execution; Sol's website checks do not replace that gate. Source
selection notes and the original archive remain inspectable in the viewer.

Run offline checks with `uv run pytest tests/test_pi_authoring.py --no-cov` and
`node --check tools/pi-authoring/app.js`, plus the required repository gate.

## Drafts and navigation

Typing saves locally immediately and syncs to the server after a short pause.
Switching variants flushes queued saves; writes are serialized so a slower older
save cannot overwrite a newer draft. Empty drafts are saved too, without changing
the last accepted text. A failed sync leaves the local draft available and shows
its status. Reloading restores and resyncs that draft. Author names from existing
drafts remain in metadata; the editor has no writer-name field.

The arrows visit every variant in order: A normal/casual/persuasive, then B
normal/casual/persuasive, then the next task. The previous arrow reverses that
order. Optional investigations use their own three-style sequence. The global
sequence ends disable the corresponding arrow. A successful check advances once
if the writer is still viewing that unchanged draft; a failed/rejected check or
an edit/navigation during the check does not move them.
