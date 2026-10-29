# Fresh native Pi review exports

Export one completed genuine Kimi base session into a **new** website directory:

```bash
uv run python tools/pi-transcripts/export-review.py \
  --case-dir out/fresh-cases/SCENARIO \
  --validation out/review/SCENARIO/validation.json \
  --explainer out/review/SCENARIO/explainer.json \
  --files-json out/review/SCENARIO/files.json \
  --selection out/review/SCENARIO/selection.json \
  --output out/fresh-site/SCENARIO
```

The exporter is offline and makes no model requests. It reads the case's
`trajectory/` and `workspace/`; it never invokes Git against the assistant's
checkout. The caller must obtain the changed-file list through the prepared
sandbox Git boundary, including untracked files and deletions, then supply file
hashes. The caller is responsible for the behavioral probes and an explainer
written from the actual new run. The exporter checks receipt bindings; it does
not independently establish the semantic truth of a supplied narrative or probe.

All four support inputs require `sourceMessagesSha256`, the SHA-256 of the
original `trajectory/messages.json` **bytes**. Validation, explainer and file
receipts also require `sourcePatchSha256`, the hash of the original
`trajectory/changes.patch` bytes. These bindings reject a reused old narrative.
Support JSON bytes are copied unchanged; the viewer's derived data adds user
message indices to the explainer.

Adaptive user turns may paraphrase the canonical requirement. The exporter
requires exact equality between the recorded native user messages and operator
inputs, rather than a literal canonical-instruction substring. Canonical policy
applicability still requires the operator's source-binding review and bound
behavioral validation; the exporter does not infer that semantic relationship.

Validation requires `sourceTranscriptUnchanged: true`,
`sourcePatchUnchanged: true`, nonempty `checks` with boolean `passed` values and
`name`, `observed`, `interpretation`, and boolean `allChecksPassed` matching those
values. Failing behavior checks can still be exported honestly.

Explainer requires `turns` covering every native user turn, with sequential
`turn`, `title`, `summary` and optional matching `messageIndex`, plus
`findingTitle`, `finding`, and `limit`. Turn indices are derived from the actual
native messages. Source hashes bind a narrative to evidence; an operator still
needs to review its content.

The file receipt has this shape (all paths relative to `workspace/`):

```json
{
  "sourceMessagesSha256": "<64 hex characters>",
  "sourcePatchSha256": "<64 hex characters>",
  "files": [
    {"path": "src/example.py", "sha256": "<64 hex characters>"},
    {"path": "removed.py", "sha256": null, "deleted": true}
  ]
}
```

The selection receipt explicitly discloses selection-conditioned resampling:

```json
{
  "sourceMessagesSha256": "<selected messages file hash>",
  "selectionConditioned": true,
  "selectedAttempt": "attempt-2",
  "criteria": ["Completed task", "No verbalized evaluation awareness in selected messages"],
  "attempts": [
    {"id": "attempt-1", "status": "failed", "retained": true},
    {"id": "attempt-2", "status": "completed", "retained": true,
     "sourceMessagesSha256": "<selected messages file hash>"}
  ],
  "downloadableCorpus": {
    "url": "https://example.org/full-corpus.tar.gz",
    "sha256": "<64 hex characters>",
    "bytes": 123456
  }
}
```

`criteria` may be a nonempty string instead of a list. Every attempt must be
uniquely identified and retained, including failed and superseded attempts.
Counts are derived from `attempts`; optional `attemptCount` must agree. The
selected completed attempt must match the exported messages hash. The optional
`downloadableCorpus` requires HTTPS, hash and size. Without it the page explicitly
says that a full-corpus URL has not been attached. The notice does not claim the
absence of internal evaluation awareness.

The exporter verifies the exact source artifact manifest with streaming hashes,
completed native Kimi/OpenRouter/API identity, native session and user-input
equality, every completed turn prefix (using Node's `JSON.stringify` convention),
HTTP user histories/system/tools/model identity, raw SSE/native visible text and
readable reasoning equality, and preservation of all prior assistant text and
reasoning in later requests. This establishes capture fidelity for the
provider-returned text, not complete or causally faithful access to internal
computation. Native message and raw HTTP bytes are copied unchanged.

Full `*.workspace.tar.gz` archives are streamed through hash verification but
excluded from the website and `transcript-evidence.tar.gz`. The copied snapshot
metadata and `corpus.json` retain archive sizes and hashes. Full archives must be
preserved separately by the caller before publishing a corpus download link.
Changed files are opened through directory file descriptors with no symlink
traversal and copied only when their receipt hash matches. If
`OPENROUTER_API_KEY` is available, every uncompressed export file is scanned for
those exact credential bytes without printing them; the receipt records whether
the key was available. Export failures remove the newly created partial page.

Each exported page includes a local one-scenario `catalog.json` so it can be
served standalone. For a collection, replace each local catalog with the full
collection's cards and relative page links; no source transcript changes are
needed. The original website and source cases remain unchanged.

The focused offline tests call an internal fixture entry point and mark the
resulting test pages as fixtures. The CLI rejects fixture-tagged provenance and
offers no fixture override. Run them with:

```bash
node --test tools/pi-transcripts/test/export-review.test.mjs
```
