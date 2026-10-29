# Full user-turn variants

`author-variants.mjs` prepares editable human files and separately calls Kimi to
rewrite complete operator turns. These are authoring calls, not experimental
assistant continuations. The selected six formats are human and Kimi authors,
each with `normal`, `casual`, and `persuasive` framing.

The source JSON holds `model`, `thinking`, `framings`, workload bounds, and a
`scenarios` list. Each scenario supplies `id`, source provenance, an
`investigation` record, and `implementation.a` / `.b` records. A record contains:

- `source`: the actual prior operator turn, preserved for provenance;
- `prompt`: the explicitly adapted full user turn to rewrite;
- `canonicalInstruction`: its selected requirement, or null for investigation;
- `sourceCheckpointSha256` and `adaptation` notes.

A B implementation source is a counterfactual adaptation, not a recorded user
turn. Its adapted prompt replaces A's canonical requirement and removes A-specific
prefatory conclusions. Neutral adaptations also omit prior environment-repair
narration, private configuration variables and dependency-failure anecdotes.
The actual source turn remains unchanged in `source.txt`.

Prepare without provider access:

```bash
node tools/pi-transcripts/author-variants.mjs scaffold sources.json out/turn-variants
node tools/pi-transcripts/author-variants.mjs check sources.json out/turn-variants
```

After explicit authorization for paid Kimi authoring:

```bash
node tools/pi-transcripts/author-variants.mjs generate sources.json out/turn-variants --allow-paid
```

The API key is read from `OPENROUTER_API_KEY` and never written into evidence.
The helper verifies the public model catalog, then sends exactly
`moonshotai/kimi-k3.1-pro`, high reasoning, with at most 8,192 output tokens per
request, 180 seconds per request, 108 requests, and three concurrent author calls.
No fallback model, automatic retry, QA judge, or experimental assistant trial is
invoked. Failed and incomplete attempts are retained; inspecting them and choosing
a new output directory is required for a separate retry. Exact completed tasks
may be verified and reused under the unchanged plan.

The editable bank is:

```text
variants/<scenario>/
  investigation/
  implementation/a/
  implementation/b/
    source.txt
    authoring-source.txt
    canonical-instruction.txt
    source-provenance.json
    human/{normal,casual,persuasive}.txt
    kimi/{normal,casual,persuasive}.txt
```

Human files begin with `TODO:` until a person writes them. Scaffold never
replaces an existing human file. `check` reports a placeholder separately from a
written variant; downstream selection must reject placeholders rather than fall
back to source text. For writing either side, use `authoring-source.txt` and its
canonical requirement. In particular, B's `source.txt` is the historical A turn
and is not the assignment to rewrite.

All three Kimi formats, including normal, are actual model rewrites. The helper
preserves author HTTP request/response bodies, provider metadata, reasoning,
usage, source hashes, adapted-source hashes and output hashes under `authoring/`.
Only surrounding whitespace is trimmed from provider content. Its receipts do
not claim independent semantic/style QA: review authored text before experimental
collection and run the experiment's normal message-QA gate if that setup requires
one. Replacing a source turn requires truncating and regenerating its downstream
assistant history; a variant is not evidence that an old assistant response was
produced after that edited user message.
