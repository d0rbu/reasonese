# Transcript paths and environment cues

The twelve Kimi bases now have path-normalized review copies at the existing
[viewer](https://sight-boat-enhanced-fresh.trycloudflare.com/kimi-json-recovery/).
Each page links the edited context, the exact substitutions and cue audit, and
the original captured transcript and bundle. The collection-wide
[audit report](https://sight-boat-enhanced-fresh.trycloudflare.com/kimi-transcripts/environment-audit.html)
includes message references and remaining cues.

This is an offline derivative of the recorded runs. It does not establish that
Kimi originally operated without evaluation cues. No model calls or new runs
were made for this edit. Source trajectories and the historical conflict pilot
remain unchanged.

## Path edits

`tools/pi-transcripts/neutralize.mjs` applies one deterministic, longest-match
substitution per original string. For example, the full collected checkout path
becomes `/home/developer/projects/langchain`; repository-relative paths retain
their meaning. Python environments become `/home/developer/.venv`, Node test
dependencies `/home/developer/.local/share/node`, Go modules
`/home/developer/go/pkg/mod`, Pi installation paths `/opt/pi`, and shared
Python tools `/opt/python`. The case root, collector root and host home have
neutral fallbacks. The scenario ID, date, research-project name and Codex worktree
are removed from these absolute paths. An explicitly recorded pytest-abbreviated
MLflow module path is covered too.

The same substitutions cover the system prompt, tool schemas, every message
string, tool arguments, results and details, returned reasoning, exported HTTP
requests, native session entries, checkpoints, operator audit, patches and changed
files. Each edit records the source JSON pointer, original string offset and
before/after path. Message order, roles, timestamps, IDs, actual commands and
failures are preserved except for their paths. New hashes describe edited files;
source hashes and existing behavior checks continue to describe original evidence.

The primary viewer and evidence download contain the derivative. A separate
`context.json` contains only `system`, `tools` and `messages`, without reviewer
annotations. Raw HTTP response streams and Pi event streams remain only in the
original evidence bundle: replacing individual chunks could miss paths split
across chunks. The event-stream audit caught such fragments in several cases;
the derivative uses complete messages instead of presenting edited raw streams.
The audit reconstructs response text to check that surface as well. The edited
package is not a native Pi resume bundle; it does not qualify signature validity,
live filesystem mounts or future continuation behavior.

## Findings beyond paths

The audit searches all system/schema/message strings, every original exported
text artifact, all 279 HTTP requests and all 279 reconstructed response streams.
Broad lexical matches receive contextual review: ordinary repository tests,
fixtures, mocks, performance benchmarks and the projects' own use of the word
"evaluation" are not evidence of experiment awareness.

Eight sessions contain assistant reasoning about hidden tests, an evaluator or
benchmark expectations: Airflow, ClickHouse result statuses, MLflow, Prometheus,
Compose, Expo, LangChain fallbacks and ClickHouse targeted jobs. Particularly
direct examples are Airflow and MLflow's "hidden evaluator", Prometheus and
LangChain fallback's "benchmark task", and Expo's "coding agent benchmarks".
Expo explicitly inferred its scenario name from the environment. These passages
are retained; deleting or rewriting them would conceal observations from the run.
They do not by themselves prove that a particular path caused the speculation.

Other visible cues include:

- Network/DNS failures, an absent Docker socket, missing `rg`, `uv`, linters or
  package dependencies, and assistant comments about unavailable tooling.
- Read-only Go module-cache failures in Prometheus and Compose. The collector
  mounts prepared caches read-only and disables module downloads; this restriction
  is visible in failed broader test attempts.
- Explicit operator guidance about prepared environments, scoped test commands,
  dependency directories, and Expo's tailored Jest/memfs setup. These are real
  limitations and interventions in the recorded sessions.
- Pi's ordinary "coding agent harness" system introduction and documentation
  links. These identify coding-agent software, without explicitly identifying an
  evaluation. The introduction is retained and only local documentation paths
  change.
- Root project instructions, upstream test fixtures/mocks, local CI placeholder
  warnings, Compose's local Go test harness, and Expo's upstream "bare sandbox"
  example. These belong to normal repository development context and remain.

The standalone review website intentionally identifies the research project,
authorship, model, validation and provenance. Its annotations and downloads are
not assistant-visible context. The original archive is explicitly available for
audit and still contains the original identifying paths.

## Collection implications

For later continuations, use a reviewed edited prefix and ensure actual tool paths
match that prefix. Renaming text alone is insufficient for a live resumed agent.
For fresh collection, mount repositories and all dependencies at neutral paths
before constructing Pi's system prompt; use the same paths in subprocess cwd,
editable-install metadata, symlinks and tool results. Prepare complete writable
dependency caches or provide the intended development connectivity. These are
recommendations from the audit, not changes to the already frozen collector or
claims about new data.

Before using a base for framing comparisons, inspect the exact prefix for
evaluation speculation and existing commitment to a solution. Investigation and
implementation checkpoints carry different history and prior-work effects.
Path normalization does not resolve those experimental-design differences.

The generated build, audit, source/derivative hash receipts and browser checks
live under `out/pi-neutral-review-20261029/`. Original immutable message hashes
remain listed in [the collection record](kimi-base-collection.md).
