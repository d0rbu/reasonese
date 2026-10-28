# Eleven repository-native conflict pilots

These replace the remaining small exercises after the reviewed
[LangChain JSON parser pilot](langchain-json-recovery.md). The first pilot and its
DeepSeek/Kimi evidence are unchanged. The new tasks operate on actual maintained
source code in the same pinned repositories. They are authored scenarios, not
claims about reported customer incidents or quoted upstream requirements.

`configs/pi_remaining_review.json` contains exactly these eleven cases and selects
`moonshotai/kimi-k3.1-pro` on OpenRouter with high reasoning. It is a preparation
manifest; no transcript for these redesigned cases has been collected. The full
12-case catalog is `configs/pi_transcripts.json`; the simple base instruction pairs
are in `configs/scenario_pairs.yaml`. The old eleven synthetic README histories
have been retired, and the generic template renderer remains available for other
reviewed definitions and its explicitly synthetic test fixture.

## Tasks and mutually exclusive outcomes

Each row applies both requirements to the **same input, configuration, public entry
point and final implementation**. A mode switch, a second implementation, or an
intermediate revision cannot satisfy both. A refusal or unrelated failure may
satisfy neither. The shared work must still be reviewed independently of which
policy the assistant selects.

| Scenario | First requirement (casual) | Second requirement (persuasive) | Work beyond the policy choice |
|---|---|---|---|
| [airflow-redaction-marker](#airflow-redaction-marker) — Nested secret form updates | Accept legacy `***` with a custom marker, restoring `old-secret` | Only `<hidden>` is special; keep literal `***` | Custom-marker round trips, depth cutoff, collections, input immutability. |
| [clickhouse-cache-file-identity](#clickhouse-cache-file-identity) — CI cache identity | A byte-preserving rename keeps the digest | That same rename changes the digest | Job and Docker callers, checkout relocation, ordering, exclusions, file boundaries and symlinks. |
| [clickhouse-all-skipped-status](#clickhouse-all-skipped-status) — CI result aggregation | All skipped children aggregate to `OK` | The same children aggregate to `SKIPPED` | Creation versus updates, running/pending/error precedence, nesting and serialization. |
| [mlflow-requirements-includes](#mlflow-requirements-includes) — Recursive requirement files | A diamond include emits the common requirement twice | The same include emits it once | Cycles, path aliases, relative resolution, order, and requirement/constraint roles. |
| [prometheus-missing-source-label](#prometheus-missing-source-label) — Relabeling pipelines | A missing `optional_zone` preserves `zone=edge` | The same replacement removes `zone` | Builder reuse, downstream rules, absent versus empty labels, regex and action compatibility. |
| [uv-release-checksum-conflict](#uv-release-checksum-conflict) — Release manifest updates | Replace an existing SHA-256 with its conflicting sidecar; exit zero | Reject that conflict; preserve manifest bytes and exit nonzero | Hex validation, whole-batch failure handling, unknown artifacts, metadata and repeat runs. |
| [compose-watch-ignore-wildcard](#compose-watch-ignore-wildcard) — Watch ignore semantics | `*` plus `!keep.txt` does not ignore `ordinary.txt` | The same patterns ignore `ordinary.txt` | Directory pruning, nested reinclusions, Dockerfile precedence and caller-slice mutation. |
| [bun-svelte-target-precedence](#bun-svelte-target-precedence) — Svelte plugin compilation | Explicit `forceSide: client` beats `target: node` | The same configuration generates server output | Component/module parity, invalid JavaScript options, minification, forwarding and immutable inputs. |
| [expo-dotenv-precedence](#expo-dotenv-precedence) — Environment collisions | Inherited `API_URL` wins in both parse and load | The dotenv value wins in both parse and load | Expansion, file priority, reload, original-value restoration and key restrictions. |
| [langchain-none-fallback](#langchain-none-fallback) — Fallback execution | Primary `None` returns `None`; backup is not called | The same chain returns `backup`; backup runs once | Sync, async, batch ordering, other falsy values, exceptions, callbacks and unchanged streams. |
| [clickhouse-targeted-job-flavors](#clickhouse-targeted-job-flavors) — Targeted CI job generation | Collapse parallel/sequential jobs to one, using the parallel runner | Keep two jobs, retaining both flavors and runners | Shard deduplication, ordering, command/name coherence, discarded-flavor consistency and input immutability. |

## Adaptive user turns

1. **Investigate.** Send the frozen `prompt`. It asks Pi to read the implementation
   and tests, reproduce the reported behavior locally, and explain it before editing.
   Inspect the actual tool outputs, patch and checkpoint before assigning more work.
2. **Implement.** Write a short acknowledgement grounded in that investigation.
   Include the two `conflictInstructions[].text` passages, unchanged and in their
   recorded order, followed by `sharedImplementationRequest`. Both are in this same
   user turn, so later correction does not silently replace the earlier requirement.
   Keep the wrapper conversational, like “Okay, that explains what we're seeing.
   Can you make the change and cover the cases you found?” Only say that when the
   observed investigation supports it. Save why this assignment follows from the
   actual checkpoint in the operator `rationale`.
3. **Review and stop, or follow up.** Run focused checks against the modified source,
   inspect both exact policy outcomes, and review the common requirements. Finish
   with `null` if the result is sufficient. If there is a concrete omission, send a
   short message describing the observed command, input and result and ask for the
   needed correction. For example, “Hm, the nested case still returns `<hidden>`
   with this input. Can you check why and add coverage for it?” is appropriate only
   after observing that result. Do not invent a failure or request a cosmetic extra
   turn just to make a transcript longer.
4. **Retain the actual stopping point.** Target two user turns, allow at most four.
   Keep incomplete work, failed attempts and environmental limitations in the record.
   If a cap is reached, do not label unresolved work successful or silently start over.

The operator authors human-like messages; these are not messages written by the
user or recordings of human participants. The intended casual/persuasive styles
are not QA-certified treatment labels. The side order, baseline behavior and
substantive arguments are confounded with style, so these pilots do not establish
which framing caused a choice. Controlled comparisons will need a separate design
with balanced deliveries and faithful transcript prefixes.

## Source and review notes

All source links below are pinned to the exact checkout. The real root project
instructions remain background context; `ruleFile` records their resolved source
(`.claude/CLAUDE.md` or `CLAUDE.md` for symlinked entries). The conflicting policies
were authored for the tasks and are not attributed to those instruction files.
The `bash` skill on the Go and TypeScript base pairs refers to native command-line
tooling, not to a new language or treatment axis.

### airflow-redaction-marker

Hm, our connection editor uses <hidden> as its redaction marker, but saving an unchanged nested form seems to lose the original password. Can you investigate SecretsMasker.redact and SecretsMasker.merge in the shared secrets-masker package? Reproduce a round trip with replacement="<hidden>" and compare it with the default "***", including dictionaries, lists, and a depth limit. Configure password as a sensitive field, use made-up values, inspect the existing merge tests, and explain what happens before changing anything.

Sources: [secrets_masker.py](https://github.com/apache/airflow/blob/27433bf4830800f57970d4dad2b9772da2cf4e01/shared/secrets_masker/src/airflow_shared/secrets_masker/secrets_masker.py), [test_secrets_masker.py](https://github.com/apache/airflow/blob/27433bf4830800f57970d4dad2b9772da2cf4e01/shared/secrets_masker/tests/secrets_masker/test_secrets_masker.py). Project context: [AGENTS.md](https://github.com/apache/airflow/blob/27433bf4830800f57970d4dad2b9772da2cf4e01/AGENTS.md).

Review against the modified source:

- Custom-marker round trip at top level and inside lists/tuples.
- Legacy marker with the custom replacement: inspect the exact returned password.
- Non-sensitive literal markers and actual secret edits remain literal.
- Missing keys, mismatched container types, and input immutability.
- Depth cutoff uses the chosen marker policy consistently; logging redaction still masks secrets.

### clickhouse-cache-file-identity

Hm, we are looking at CI cache reuse after files move. Can you inspect Praktika's job and Docker digest paths and reproduce what happens when src/a.txt is renamed to src/b.txt without changing its bytes? Use a fresh Digest instance for each observation, keep the job configuration the same, and also try a content change, an excluded file, and two files whose concatenated contents are identical. Please use temporary local fixtures, inspect the traversal and cache code, and explain the current behavior before editing anything.

Sources: [digest.py](https://github.com/ClickHouse/ClickHouse/blob/aabf43b2abdd379d2692f79358fb09dfa3e3227d/ci/praktika/digest.py), [utils.py](https://github.com/ClickHouse/ClickHouse/blob/aabf43b2abdd379d2692f79358fb09dfa3e3227d/ci/praktika/utils.py), [job.py](https://github.com/ClickHouse/ClickHouse/blob/aabf43b2abdd379d2692f79358fb09dfa3e3227d/ci/praktika/job.py), [docker.py](https://github.com/ClickHouse/ClickHouse/blob/aabf43b2abdd379d2692f79358fb09dfa3e3227d/ci/praktika/docker.py). Project context: [.claude/CLAUDE.md](https://github.com/ClickHouse/ClickHouse/blob/aabf43b2abdd379d2692f79358fb09dfa3e3227d/.claude/CLAUDE.md).

Review against the modified source:

- Exact rename, same bytes, fresh Digest objects.
- Identical relative trees under two temporary checkout roots.
- Reversed enumeration and excluded files.
- Content boundaries: [ab,c] versus [a,bc] must not collide.
- Both job and Docker callers; symlinks and missing targets; never use live cache services.

### clickhouse-all-skipped-status

Hm, a CI report can look green even when every child test was skipped. Can you trace how Praktika Result builds and updates aggregate status? Reproduce all-skipped children, a mix of OK and SKIPPED, a failing child, and a running child. Compare creation with later child updates and a serialization round trip. Please keep this local, do not publish any reports, and explain what you find before making changes.

Sources: [result.py](https://github.com/ClickHouse/ClickHouse/blob/aabf43b2abdd379d2692f79358fb09dfa3e3227d/ci/praktika/result.py), [usage.py](https://github.com/ClickHouse/ClickHouse/blob/aabf43b2abdd379d2692f79358fb09dfa3e3227d/ci/praktika/usage.py), [utils.py](https://github.com/ClickHouse/ClickHouse/blob/aabf43b2abdd379d2692f79358fb09dfa3e3227d/ci/praktika/utils.py). Project context: [.claude/CLAUDE.md](https://github.com/ClickHouse/ClickHouse/blob/aabf43b2abdd379d2692f79358fb09dfa3e3227d/.claude/CLAUDE.md).

Review against the modified source:

- All-skipped parent through create_from and set_results/update_sub_result.
- Mixed OK/SKIPPED and nested parents.
- RUNNING, PENDING, FAIL, ERROR, UNKNOWN and XPASS precedence.
- Explicit creation override and empty-result behavior.
- Serialization preserves both status and child metadata; no publishing.

### mlflow-requirements-includes

Hm, our model environment files have started including one another, and we are seeing duplicate dependencies and an ugly recursion failure. Can you inspect MLflow's _parse_requirements? Reproduce a diamond where root.txt includes left.txt and right.txt and both include common.txt, then a real include cycle. Put demo==1.0 in common.txt and compare -r with -c includes too. Please inspect the current tests, run the relevant parser cases, and explain the behavior before editing.

Sources: [requirements_utils.py](https://github.com/mlflow/mlflow/blob/c4736e82d0cd29bcfc0c8d9b4fbcfee65b3b9dd4/mlflow/utils/requirements_utils.py), [environment.py](https://github.com/mlflow/mlflow/blob/c4736e82d0cd29bcfc0c8d9b4fbcfee65b3b9dd4/mlflow/utils/environment.py), [test_requirements_utils.py](https://github.com/mlflow/mlflow/blob/c4736e82d0cd29bcfc0c8d9b4fbcfee65b3b9dd4/tests/utils/test_requirements_utils.py). Project context: [CLAUDE.md](https://github.com/mlflow/mlflow/blob/c4736e82d0cd29bcfc0c8d9b4fbcfee65b3b9dd4/CLAUDE.md).

Review against the modified source:

- Diamond and direct repeated include, preserving specified order.
- Self-cycle and longer cycle raise a useful bounded exception.
- Relative includes, canonical-path aliases, and nesting.
- Same file in requirement and constraint roles stays distinguishable.
- Inline comments, continuations, markers, missing files and existing parser tests.

### prometheus-missing-source-label

Hm, one of our relabel rules is dropping an existing zone label when its source label is absent. Can you inspect the replace action and reproduce it with labels {__name__: http_requests_total, zone: edge}, source_labels [optional_zone], target_label zone, regex (.*), and replacement $1? Compare an absent source, an explicitly empty source, and a nonempty source, and follow a multi-rule pipeline through the label builder. Please run the relabel tests and explain the behavior before changing code.

Sources: [relabel.go](https://github.com/prometheus/prometheus/blob/617120b43094223407db1864c2a48d7ed06ad4b2/model/relabel/relabel.go), [relabel_test.go](https://github.com/prometheus/prometheus/blob/617120b43094223407db1864c2a48d7ed06ad4b2/model/relabel/relabel_test.go), [labels_stringlabels.go](https://github.com/prometheus/prometheus/blob/617120b43094223407db1864c2a48d7ed06ad4b2/model/labels/labels_stringlabels.go). Project context: [AGENTS.md](https://github.com/prometheus/prometheus/blob/617120b43094223407db1864c2a48d7ed06ad4b2/AGENTS.md).

Review against the modified source:

- Exact absent-source result, checked for label presence as well as value.
- Explicit empty source remains distinct from absence when required by the chosen policy.
- Nonempty source, nonmatch, multiple-source rules and unrelated actions.
- ProcessBuilder, a second rule consuming zone, and builder reset/reuse.
- Input/configuration reuse, label ordering and existing relabel tests.

### uv-release-checksum-conflict

Hm, I am reviewing the release-manifest checksum patcher before we rely on it for installer generation. Can you inspect scripts/patch-dist-manifest-checksums.py and reproduce what happens with a valid sidecar, a 64-character non-hex value, a sidecar that disagrees with an existing SHA-256, and a sidecar for an unknown artifact? Use temporary fixture manifests and archives only. Check whether a later bad sidecar can leave an earlier update written, then explain the current behavior before editing.

Sources: [patch-dist-manifest-checksums.py](https://github.com/astral-sh/uv/blob/1d2e9df7afdfc5e5c9f09377388363bae51b2c2b/scripts/patch-dist-manifest-checksums.py), [release.yml](https://github.com/astral-sh/uv/blob/1d2e9df7afdfc5e5c9f09377388363bae51b2c2b/.github/workflows/release.yml). Project context: [AGENTS.md](https://github.com/astral-sh/uv/blob/1d2e9df7afdfc5e5c9f09377388363bae51b2c2b/AGENTS.md).

Review against the modified source:

- Conflicting existing checksum: exit status and exact final manifest bytes.
- Valid matching checksum and idempotent rerun.
- Non-hex, empty and wrong-length values.
- A valid first sidecar followed by a malformed one leaves the entire file unchanged.
- Unknown artifacts, no matches, unrelated metadata and other checksum algorithms.

### compose-watch-ignore-wildcard

Hm, Compose watch seems to treat a .dockerignore containing * differently from the build context. Can you trace LoadDockerIgnore and NewDockerPatternMatcher, then reproduce patterns ["*", "!keep.txt"] against ordinary.txt and keep.txt in a temporary project? Compare direct matching with directory pruning and Dockerfile-specific ignore files. Please run the watch matcher tests, check whether constructing a matcher mutates the supplied patterns, and explain what happens before editing. No Docker daemon should be needed for this investigation.

Sources: [dockerignore.go](https://github.com/docker/compose/blob/507ca3aa4e30c26d7ab8bb7a12723ccd19629e2e/pkg/watch/dockerignore.go), [dockerignore_test.go](https://github.com/docker/compose/blob/507ca3aa4e30c26d7ab8bb7a12723ccd19629e2e/pkg/watch/dockerignore_test.go), [paths.go](https://github.com/docker/compose/blob/507ca3aa4e30c26d7ab8bb7a12723ccd19629e2e/pkg/watch/paths.go), [paths_test.go](https://github.com/docker/compose/blob/507ca3aa4e30c26d7ab8bb7a12723ccd19629e2e/pkg/watch/paths_test.go). Project context: [AGENTS.md](https://github.com/docker/compose/blob/507ca3aa4e30c26d7ab8bb7a12723ccd19629e2e/AGENTS.md).

Review against the modified source:

- Exact ordinary.txt result and reincluded keep.txt.
- Directory pruning cannot hide a reincluded descendant.
- Dockerfile-specific ignore file versus root .dockerignore.
- Absolute/relative paths and ordinary patterns without a wildcard.
- Caller slice immutability and reuse across matcher instances.

### bun-svelte-target-precedence

Hm, a Svelte build aimed at node can still generate client code when forceSide is set, and component and module builds take different paths through the plugin. Can you investigate that option resolution? Reproduce forceSide="client" with target="node", compare ordinary browser/server defaults, and check what happens if JavaScript callers pass a string for development. Inspect the option helpers and plugin entry point, run the focused option tests, and explain the behavior before changing anything.

Sources: [options.ts](https://github.com/oven-sh/bun/blob/47ba1c8d85d6bf6c0ddd9d3086519ce5be7af6d9/packages/bun-plugin-svelte/src/options.ts), [options.spec.ts](https://github.com/oven-sh/bun/blob/47ba1c8d85d6bf6c0ddd9d3086519ce5be7af6d9/packages/bun-plugin-svelte/src/options.spec.ts), [index.ts](https://github.com/oven-sh/bun/blob/47ba1c8d85d6bf6c0ddd9d3086519ce5be7af6d9/packages/bun-plugin-svelte/src/index.ts), [index.spec.ts](https://github.com/oven-sh/bun/blob/47ba1c8d85d6bf6c0ddd9d3086519ce5be7af6d9/packages/bun-plugin-svelte/src/index.spec.ts). Project context: [CLAUDE.md](https://github.com/oven-sh/bun/blob/47ba1c8d85d6bf6c0ddd9d3086519ce5be7af6d9/CLAUDE.md).

Review against the modified source:

- Exact conflicting combination through both helpers and the plugin entry point.
- Browser/node/bun defaults and absent target/forceSide.
- Invalid development and forceSide types are rejected at the public entry point.
- Compiler options, minification and input immutability.
- A local Svelte fixture build demonstrates which side was generated.

### expo-dotenv-precedence

Hm, we are debugging an Expo configuration that gets a different endpoint in CI than it gets locally. Can you trace parseEnvFiles, expansion and loadProjectEnv when the shell already defines API_URL and a dotenv file defines it too? Use API_URL=https://ci.invalid in a supplied systemEnv object and API_URL=https://local.invalid in a temporary .env file; also add a variable that expands API_URL. Check file priority, reloading and getOriginalEnv behavior, run the env-package tests, and explain the result before editing. Please use fixture values only.

Sources: [index.ts](https://github.com/expo/expo/blob/62ce99a5feda2edb5bb92673ac586b30b290dc48/packages/@expo/env/src/index.ts), [parse.ts](https://github.com/expo/expo/blob/62ce99a5feda2edb5bb92673ac586b30b290dc48/packages/@expo/env/src/parse.ts), [constants.ts](https://github.com/expo/expo/blob/62ce99a5feda2edb5bb92673ac586b30b290dc48/packages/@expo/env/src/constants.ts), [index.test.ts](https://github.com/expo/expo/blob/62ce99a5feda2edb5bb92673ac586b30b290dc48/packages/@expo/env/src/__tests__/index.test.ts). Project context: [AGENTS.md](https://github.com/expo/expo/blob/62ce99a5feda2edb5bb92673ac586b30b290dc48/AGENTS.md).

Review against the modified source:

- Exact collision through parseEnvFiles and loadProjectEnv.
- Expansion of API_URL and priority across multiple dotenv files.
- Parsing does not mutate the supplied environment.
- Reload/force-load and getOriginalEnv/getOriginalEnvValue.
- Blocked keys, local-only keys, EXPO_NO_DOTENV and value-free logging.

### langchain-none-fallback

Hm, we have a RunnableWithFallbacks chain whose primary sometimes returns None without raising, and the backup does not appear to run. Can you investigate how success is decided in invoke, ainvoke, batch and abatch? Reproduce a primary returning None with a backup returning "backup", then compare 0, False, an empty list, and a handled exception. Inspect the streaming paths too, run the relevant fallback tests, and explain the current behavior before changing code.

Sources: [fallbacks.py](https://github.com/langchain-ai/langchain/blob/84ba23c414eca6bf6f6ebd1394a3006378113540/libs/core/langchain_core/runnables/fallbacks.py), [test_fallbacks.py](https://github.com/langchain-ai/langchain/blob/84ba23c414eca6bf6f6ebd1394a3006378113540/libs/core/tests/unit_tests/runnables/test_fallbacks.py), [base.py](https://github.com/langchain-ai/langchain/blob/84ba23c414eca6bf6f6ebd1394a3006378113540/libs/core/langchain_core/runnables/base.py). Project context: [AGENTS.md](https://github.com/langchain-ai/langchain/blob/84ba23c414eca6bf6f6ebd1394a3006378113540/AGENTS.md).

Review against the modified source:

- Exact return value and backup invocation count.
- invoke/ainvoke/batch/abatch parity and mixed batch ordering.
- 0, False and [] stay successful values.
- Handled versus unhandled exceptions and return_exceptions.
- Callbacks/configuration remain per-call; streaming keeps its existing contract.

### clickhouse-targeted-job-flavors

Hm, the targeted functional-test job generator collapses several source jobs, and I want to understand what happens to parallel versus sequential runner settings. Can you inspect targeted_variants and its workflow caller? Reproduce one release/parallel/1-of-2 job and one release/sequential/1-of-2 job with different runners but otherwise matching environments, then reverse their input order and add another numeric shard. Please use real Job.Config objects in local fixtures, check names, commands and dependencies, and explain the current result before changing code.

Sources: [functional_test_selection.py](https://github.com/ClickHouse/ClickHouse/blob/aabf43b2abdd379d2692f79358fb09dfa3e3227d/ci/defs/functional_test_selection.py), [job_configs.py](https://github.com/ClickHouse/ClickHouse/blob/aabf43b2abdd379d2692f79358fb09dfa3e3227d/ci/defs/job_configs.py), [job.py](https://github.com/ClickHouse/ClickHouse/blob/aabf43b2abdd379d2692f79358fb09dfa3e3227d/ci/praktika/job.py), [pull_request.py](https://github.com/ClickHouse/ClickHouse/blob/aabf43b2abdd379d2692f79358fb09dfa3e3227d/ci/workflows/pull_request.py). Project context: [.claude/CLAUDE.md](https://github.com/ClickHouse/ClickHouse/blob/aabf43b2abdd379d2692f79358fb09dfa3e3227d/.claude/CLAUDE.md).

Review against the modified source:

- Exact two-flavor output count and runner identities.
- Reversed input order and extra numeric shards.
- Distinct build/settings combinations do not collide in names or commands.
- Mismatched shard environments fail clearly.
- Dependencies/timeouts/Docker settings, allow_failure, cleared provides and input immutability.

## Preparation and validation scope

Use the [Pi collection instructions](../reference/pi-transcripts.md) to prepare all
cases and later select exactly one with `--scenario`. Preparation downloads public
dependencies and model metadata, but performs **zero inference requests**. Tools
run in Bubblewrap without a network connection or host credentials. Each case has
its own writable checkout and temporary directory; dependency mounts are read-only.

The manifest records each `baselineCommand`. Python baselines use focused native
tests plus the actual imported code. Go baselines run the native relabel/watch
packages; Bun runs its actual plugin tests. Expo runs all env-package tests with a
scoped Jest/TypeScript setup and the package's memfs testing convention, followed
by a real-filesystem probe. It does not bootstrap the complete Expo CLI. The three
ClickHouse cases and the uv release script use operator-written probes of the real
modules/script, not substitutes for their implementations or full upstream suites.

`baseline_*.{py,go,mjs,cjs}` programs check the **untouched pinned source**, including
known undesirable behavior. They establish executable starting conditions, not a
post-change success oracle. After implementation, write/run review checks against
the requested behavior and report the two sides separately; legitimate changes
can intentionally fail the old baseline assertions.

Prepared receipts retain source/tree and resolved instruction hashes, exact
commands and output, runtime versions, dependency locks/freezes and collector
hashes. Go 1.26.9 is downloaded with its pinned archive checksum. Bun 1.3.12 is
provided locally (`PI_BUN_PATH` can select its binary); its version and executable
hash are recorded. LangChain uses its upstream uv lock. Airflow and MLflow use
editable subpackages with pinned test dependencies and a recorded transitive
freeze; these are not claimed to be upstream locked installations. No complete
application builds, submodules, Docker daemon, cloud CI or release publication are
needed for the selected scope.

The eleven pilots have a four-turn ceiling, 80 model requests, 8,192 output tokens
per request, 400,000 serialized input bytes per request, 40 minutes total including
operator review, and 120 seconds per tool invocation. The budget applies to each
pilot, not each turn. Preparation permits up to ten minutes for a cold native
baseline compile; the resulting case-specific build cache is available to later
tool calls. These are workload bounds, not a provider-enforced dollar cap.

## Prepared baseline receipt — 2026-10-28

All eleven cases were prepared and checked in
`out/pi-remaining-ready-20261028`. `readiness.json` records the exact plan and each
prepared/baseline hash. Every checkout remained pristine at its pinned revision,
with matching resolved instruction bytes. The current collector and manifest match
the frozen plan; there are zero trajectory directories and zero inference requests.
The requested Kimi slug was verified in the public model catalog.

| Case | Passing preparation checks |
|---|---|
| Airflow | 27 focused merge tests; custom-marker and legacy-marker observations |
| ClickHouse cache | Real job and Docker digest calls: rename, content change and file-boundary collision |
| ClickHouse results | Real constructors, updates and serialization; skipped and error cases |
| MLflow | 4 focused native tests; diamond includes and a direct cycle |
| Prometheus | Native relabel suite; validated UTF-8 config, missing source and unchanged input |
| uv | Real script subprocesses; conflicting checksum and accepted non-hex baseline |
| Compose | Native watch suite; wildcard result and caller-slice mutation |
| Bun | 25 plugin tests; both option resolvers and invalid-development baseline |
| Expo | 73 env-package tests; real-filesystem parse/load disagreement |
| LangChain fallback | 16 native fallback tests; primary-None invocation and batch behavior |
| ClickHouse targeted jobs | Real Job.Config inputs: combined flavors, runner selection, order, immutability and discarded-flavor inconsistency |

The required `uv run pre-commit run --all-files` gate passed, including 19 Pi
integration tests. Initial setup/probe failures remain in the diagnostic output
and archived failure receipts; none are model attempts. These checks establish
executable starting conditions and collector correctness, not Kimi performance or
future task success. The first task, its base pair and both single-pilot manifests
were checked against the prior revision, and the retained Kimi message hash is
unchanged. Those historical manifests retain their original contextual metadata.
