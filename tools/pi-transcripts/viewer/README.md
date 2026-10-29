# Transcript review viewer

Static review assets for an exported Kimi base collection. The left sidebar links
the scenarios; each page shows its explainer, original messages, returned
reasoning, changed files, independent checks and downloadable evidence.

Copy `index.html`, `app.js`, `style.css` and `favicon.svg` into each exported
scenario directory, next to its `data.json`. Serve the common site root over HTTP.
The pages fetch
`/kimi-transcripts/catalog.json`, whose `scenarios` list contains:

```json
{
  "id": "langchain-json-recovery",
  "title": "JSON recovery",
  "project": "LangChain",
  "caption": "The selected first-instruction outcome.",
  "path": "/kimi-json-recovery/",
  "userTurns": 2,
  "checksPassed": true
}
```

Each page's review JSON contains `provenance`, `outcome`, `messages`, `turns`,
`system`, `toolDefinitions`, `files`, `patch`, `reportedModels`,
`reportedProviders`, `artifacts`, `validation`, `explainer` and `reasoning`.
`validation.checks` supplies the observed behavior table; `explainer.turns`
supplies the operator's annotations and message indices. Those annotations stay
separate from the recorded messages. Evidence files live under `evidence/`,
changed files under `files/`, and the complete download is
`transcript-evidence.tar.gz`.

Keep exports and collection-specific scripts below an ignored directory such as
`out/`. Preserve source message bytes and verify exported artifact hashes before
publication. A normal `completed` session status and passing scoped code checks
are separate facts. The viewer treats transcript text as text, without evaluating
model-authored HTML or Markdown links.

The first page retains `/kimi-json-recovery/`; other pages use
`/kimi-transcripts/<scenario-id>/`. The archive link points to the earlier
`/kimi-json-recovery-conflict-20261028/` pilot. See the
[collection workflow](../../../docs/reference/pi-transcripts.md) for provenance
and collection limits.
