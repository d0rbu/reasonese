# Activation role probes

Role probes are an additional, model-specific measurement for instruction QA. They use a
target model's hidden activations to estimate the native conversational role represented by
each content token. The implementation follows the controlled construction in
[Ye et al. (2026)](https://arxiv.org/abs/2603.12277) and was cross-checked against the
[authors' released code](https://github.com/role-confusion/prompt-injection-as-role-confusion).

## Scientific status

The training and artifact code is an instrument, not evidence that any particular model has a
usable role direction. A probe is eligible for instruction QA only after it passes both validity
checks from Appendix G:

1. held-out accuracy on neutral text rendered under native role wrappers; and
2. zero-shot role identification on untouched, model-native conversational traces that were not
   used to train the probe.

Acceptance thresholds belong to a frozen project protocol. The paper does not prescribe those
numerical cutoffs, so reports must identify them as project choices. Synthetic unit tests
establish implementation behavior only. They do not qualify a model probe.

Local open-weight activations also do not establish exact parity with an OpenRouter deployment
unless the hosted checkpoint, quantization, tokenizer, and template are all independently shown
to match. Probe reports must preserve that limitation.

### Current Nemotron result (September 13, 2026)

The expanded Nemotron run used 250 neutral documents, a 1,024-token cap, and candidate
layers 13, 20, and 26. Twenty-two of 24 layer/lambda candidates completed; layer 26 failed
with explicit convergence errors at lambda `0.01` and `100`. Development selection chose
layer 13 and lambda `0.001`. On the held-out neutral split, token accuracy was 90.3856%, but
reasoning recall was 87.0986% and assistant/final-output recall was 79.6390%. The latter is
below the unchanged 85% per-role gate, so neutral validation failed.

On 12 untouched native test conversations, reasoning recall was 22.8668%, final-output
recall was 100%, and document-macro accuracy was 56.7335%. AUC of conversation-average
segment scores was 1.0, and the calibration threshold accepted 11 of 12 reasoning segments
while rejecting all 12 final segments. Those ranking and threshold results do not compensate
for the failed native per-role and document-macro gates.

A separate calibration-only control measured mean unconditional `P(reasoning)` under three
renderings of the same 12 intact conversations:

| Rendering | Reasoning segment | Final segment |
| --- | ---: | ---: |
| Native roles | 0.335528 | 0.00000494696 |
| Entire conversation under a tool role | 0.530172 | 0.0372404 |
| Raw text without role tags | 0.449161 | 0.0182800 |

The reasoning-versus-final distinction persisted without the correct native wrappers. The
control preserves source text and source order, but wrapper changes also alter absolute token
positions and causal context, so it does not isolate prose style. It is descriptive only:
intact conversation segments are not authored instructions, and the result does not establish
instruction-style indistinguishability or change any gate.

A second post hoc, calibration-only diagnostic found that reasoning recall fell from 95.18%
over content-token indices `[0, 64)` across 12 conversations to 3.87% over `[512, 1024)`
across the 10 conversations long enough to contribute. For the first three fixed calibration
documents with at least 768 reasoning tokens, the `[512, 768)` window was then replayed as
the only reasoning text, with the original user prompt and final output retained. All 256 token
IDs matched exactly in each pair. Five-class reasoning recall changed from 0.390625% to
94.921875%, 0% to 47.265625%, and 6.640625% to 94.140625%. This is direct evidence that
the probe's classification is context-sensitive in these examples. Removing the preceding 512
reasoning tokens also changed absolute token positions, so the intervention does not isolate
context, position, or prose style. It does not justify excluding later reasoning tokens as a
remedy, and it changed no fit, gate, qualification, or artifact status.

A third calibration-only intervention removed the exact user turn while retaining the native empty
system envelope and assistant/reasoning wrappers. All 5,545 reasoning and final content tokens in
the same three fixed conversations were preserved exactly. Removing the user turn did not rescue
reasoning recognition: per-conversation recall remained at or below 25.81% with the original probe
and 31.45% with the development-selected standardized probe, while final-output recall remained
100% in every condition. This intervention changes preceding context and absolute positions
together, so three calibration examples do not show that user context never matters. It read no
test activation rows and changed no fit, selection, gate, qualification, or artifact status.

A separate post hoc standardization diagnostic fit scaling statistics on the same 150 neutral
training documents and evaluated all eight layer-13 lambda candidates on the same 50-document
development split. All eight fits converged, and seven met the unchanged numeric neutral criteria
on development data. Development-only selection chose lambda `100`, with 92.2731% token accuracy
and 85.0063% minimum per-role recall. A separately frozen screen then applied only that candidate
to the existing 12 native calibration conversations: reasoning recall was 31.4968%, final-output
recall was 100%, and document-macro accuracy was 66.6015%. The diagnostic did not rescore neutral or
native test data, performed no refit, and changed no gate; the calibration result was not treated as
sufficient to proceed to qualification and produced no qualified model.

The pinned upstream source at revision `ec333c40fd43fe991e1ebf66765051b6d7e35784`
sets `SKIP_FIRST_N = 32` for nested-reasoning models and filters the rows produced through
`label_nemotron3_content_roles` to `token_in_seg_ix >= SKIP_FIRST_N` before neutral probe fitting
and development evaluation. Native projection retains full content segments. The exact notebook
and `utils/role_assignments.py` hashes are recorded in the
`paper-source-audit/receipt.json` and `paper-source-audit/role-assignments-receipt.json` evidence.
The current trainer does not apply that first-32-content-token filter. The standardization
diagnostic deliberately left this difference unchanged so that it isolated scaling; this is a
documented method difference, not evidence that the filter caused either validation failure.

The run therefore produced no QA-eligible Nemotron probe, and the pilot remains paused. The
earlier diagnostic and expanded runs are not a paired intervention, so their difference does
not identify a cause or show that the larger setup corrected the failure. Gemma evidence is
also incomplete at two of 24 native dialogues; no further provider calls are planned without
user direction.

## Controlled training data

Each neutral base document is rendered five times with the exact target checkpoint's native role
templates: `system`, `user`, `tool`, `reasoning`, and `assistant`. The role labels vary while the
content is held constant. Reasoning models commonly nest reasoning and final output inside one
assistant envelope. In that case, assistant content follows variable-length closed reasoning
filler, and other roles receive position-matched filler. Filler comes from a dedicated document
pool disjoint from every target document. Role tags and filler tokens are excluded from fitting;
the disjoint pool also prevents held-out target content from influencing training activations
through attention.

`ActivationDataset` verifies the controls before training:

- every neutral document contains every configured role;
- ordered content token IDs are identical across all role copies;
- relative content indices are contiguous;
- absolute sequence positions match across the role copies;
- filler document IDs are consistent across each target's role copies and disjoint from targets;
- every activation is finite and has the recorded dtype, layer count, and hidden size; and
- the metadata says that only content tokens remain.

Base document IDs, rather than rendered sequence IDs or token rows, are split into train,
validation, and test sets. All five copies of one document therefore remain in one split. This is
stricter than splitting role variants independently, which would leak the same underlying content
across the holdout boundary.

The qualification-scale neutral protocol uses 250 target documents with up to 1,024 retained
content tokens per role copy; boundary-straddling tokens are masked, and exact retained counts are
recorded. It preregisters three candidate layers for each model: 13, 20, and 26 for
Nemotron; 15, 23, and 30 for Gemma. The earlier 60-document, single-layer protocol remains accepted
only as a diagnostic run and cannot satisfy the expanded protocol's document or token counts.

The provenance includes exact model and tokenizer revisions, a digest of the resolved weight
manifest, the native template digest, model and stored-activation dtypes, the model-specific
activation site, a runtime digest covering the software and kernel path, the neutral corpus
digest, layers, masking counts, and extraction protocol. A probe for one model pipeline cannot
be qualified with conversation activations from another. Projection also requires the exact
trained layer and stored dtype instead of accepting an unidentified two-dimensional array.
The qualification protocol separately binds the exact normalized target and filler-source
digests, its own file digest, and the fresh native-prompt partition digest.

## Classifier and selection

The classifier is L2-regularized multinomial logistic regression. Its class space contains all
five roles, avoiding a forced reasoning-versus-final decision for text that the model represents
as user, system, or tool content. Layer and regularization are selected jointly on the grouped
neutral validation split using the preregistered layers and Appendix-G lambda grid. Accuracy is
primary, negative log likelihood breaks ties, followed by the larger penalty and then the
shallower layer. The selected pair alone is refit on train plus validation documents and evaluated
once on held-out neutral documents. The artifact retains every candidate's development metrics,
the selected pair, and the exact training configuration; its fingerprint covers all of them. The
test set and real conversations never select a layer or penalty.

Metrics include token accuracy and negative log likelihood, per-role recall,
document-macro accuracy, per-role document-macro accuracy, and the full confusion matrix. The
document metrics prevent a handful of long documents from hiding failures elsewhere.

The expanded protocol uses explicit cuML QN in FP32 with `penalty_normalized=True`, `C=1/lambda`,
`max_iter=5000`, `linesearch_max_iter=100`, `lbfgs_memory=5`, and `tol=1e-3`. The protocol stores a
canonical runtime record covering the solver arguments, package versions, CUDA device/runtime,
matrix layout, and hashes of the fitted implementation files. Training refuses to run when the
live runtime differs. The older diagnostic protocol uses explicit sklearn LBFGS in FP64 and records
its runtime separately; results from the two numerical backends are not represented as bitwise
equivalent.

The cuML tolerance was frozen before any expanded development or test scores were read. At the
released backend's `1e-4` default, train-only Nemotron layer-13 fits at lambda `1e-4` and `0.1`
reported line-search failures; increasing L-BFGS memory did not resolve the failure, and a bounded
FP64 reference run did not finish. With `tol=1e-3`, the preregistered lambda `0.1` fit converged in
1,791 iterations, and an independent FP64 calculation on the complete training split confirmed
that its gradient infinity norm (`0.0000994677`) was below the cuML convergence bound
(`0.0001122102`). This train-only numerical check changed no layer, lambda, split, or acceptance
gate.

Candidate-failure handling was amended post hoc during the development stage, rather than being
part of the original preregistration. Sixteen development candidates at layers 13 and 20 had
finished when the layer-26, lambda-`1e-4` fit emitted an explicit cuML line-search failure; no
selected refit, held-out neutral test, or native calibration/test score had been computed. The
amended protocol records `candidate_failure_policy` as `exclude explicit convergence failures`.
During the development grid only, a recognized sklearn convergence warning or one of the pinned
cuML convergence-failure messages excludes that coordinate from selection. The artifact records
the coordinate, exception type, and matched message, while successful candidates retain their
ordinary development metrics. Every configured coordinate must appear exactly once across those
two sets. Unexpected runtime, CUDA, memory, dtype, parameter, and provenance failures still abort;
all candidates failing is fatal; and a convergence failure while refitting the selected candidate
on train plus development is fatal. Failed parameters are never scored.

Keep the GPU fitter in an ignored isolated environment so the base and CPU test environments do
not install RAPIDS. The supported pinned setup is:

```bash
uv venv out/role-probe-research/cuml-env-2606 --python 3.13.2
uv pip install \
  --python out/role-probe-research/cuml-env-2606/bin/python \
  --extra-index-url https://pypi.nvidia.com \
  --index-strategy unsafe-best-match \
  -e . 'numpy==2.4.6' 'pytest>=9.0.0' 'scikit-learn>=1.8.0' \
  'cuml-cu13==26.6.0' 'libcuml-cu13==26.6.0' 'cupy-cuda13x==14.2.0' \
  'rmm-cu13==26.6.0' 'cuda-python==13.0.3' \
  'cuda-toolkit[cublas,cufft,curand,cusolver,cusparse]==13.0.3.0'
```

Run the management command with that environment's Python. The command reconstructs the live
runtime record and fails before fitting unless it exactly matches the record frozen in the
protocol. GPU fitting first builds an FP32 Fortran-order host matrix and copies it to exactly one
device matrix; it refuses to start without two GiB of free device headroom beyond that matrix.
Probe artifacts created before optimizer provenance was added are rejected rather than silently
interpreted as the current format.

The explicit cross-backend numerical check is opt-in because it requires a CUDA device:

```bash
REASONESE_TEST_CUML=1 \
  out/role-probe-research/cuml-env-2606/bin/python -m pytest \
  -q tests/test_role_probes.py -k cuml_qn_matches_sklearn_reference --no-cov
```

The fitted estimator is not serialized. Its coefficients and intercepts are promoted to a uniform,
portable multinomial softmax representation, so applying the trained probe does not require cuML.
Binary sklearn coefficients receive an algebraically equivalent symmetric two-logit conversion,
although production role probes use the five-role space. Tests compare portable probabilities
directly against sklearn; the cuML feasibility check compares objectives, probabilities, and
predictions within explicit numerical tolerances rather than comparing raw coefficients.

## Conversation qualification and scoring

Conversation qualification replays unmodified native reasoning and final-output text through the
same exact local checkpoint and extraction path. It uses 12 calibration conversations and 12
document- and content-disjoint test conversations. The calibration split alone selects a
deterministic threshold on the per-segment mean unconditional reasoning probability. The frozen
test checks pooled AUC with a paired-conversation, 10,000-replicate bootstrap as well as native
role recall and document-macro accuracy. It also applies the unchanged calibration threshold and
requires at least 0.75 reasoning sensitivity and 0.75 final-output specificity. This operating-point
gate prevents rank separation alone from qualifying a threshold that does not transfer. The test
split never selects or adjusts the threshold. Exact
activation-dataset fingerprints and the native-prompt partition digest remain in the fitted
artifact.

The five-class classifier is also evaluated directly on the two observed native roles.
Predictions of `system`, `user`, or `tool` count as errors, while the absence of those roles from
an ordinary conversation is not itself an error. Every qualification conversation must contain
both non-empty reasoning and assistant segments.

Instruction scoring returns the full five-role probability vector for every content token and its
mean. The QA signal corresponding to the paper's CoTness is the unconditional mean probability of
the `reasoning` class. Other role probabilities remain available as controls. Compressed framing
scores are descriptive: compressed instructions do not have a required probe direction.

Probe artifacts are fingerprinted, pickle-free `.npz` files containing portable float64 weights
and JSON metadata. The fingerprint covers the weights, fit and validation metrics, split document
IDs, thresholds, and full provenance. Generated activation exports, fitted probes, and reports
belong below ignored `out/` directories rather than in source control.

## Local extraction

`reasonese-extract-role-activations` extracts one exact, pinned native model adapter at a time.
It loads a filtered BF16 checkpoint only through the highest requested probe site, CPU-offloads
the retained prefix, and microbatches the five position-matched role copies of each document.
Nemotron preloads each whole `NemotronHMamba2Mixer` while it executes because that module reads a
child weight directly when choosing its CUDA path. On a runtime identified as fused, every forward
audits the actual CUDA-kernel and PyTorch-fallback methods and aborts immediately if the fallback is
selected or no retained Mamba layer executes.
The default is all five sequences when Nemotron's pinned fused Mamba kernels are active or for
Gemma, and two sequences when Nemotron falls back to the memory-intensive native PyTorch scan;
`--batch-size` records and overrides it. A dedicated filler pool must follow the target documents
in the corpus JSONL; its
document IDs and text must be disjoint from every target. For example, the 16-document systems
smoke uses:

```bash
uv run reasonese-extract-role-activations \
  --adapter nemotron-3.5-lightning-native-v1 \
  --checkpoint out/role-probe-research/prefix-nemotron \
  --corpus out/role-probe-research/smoke-corpus.jsonl \
  --output out/role-probe-research/smoke-nemotron-activations \
  --layers 26 \
  --documents 16 \
  --filler-documents 16 \
  --activation-dtype float32
```

The checkpoint directory must contain the pinned config, tokenizer, filtered safetensors index
and shards, and `prefix-checkpoint-manifest.json`. The utility rejects a template, architecture,
revision, layer bound, tensor set, or weight identity that does not exactly match the adapter.
It refuses to overwrite an existing artifact. A completed directory includes content-only
activations, numeric row metadata, target and filler-document mappings, file digests, masking
counts, package versions, exact model and template provenance, and runtime duration and peak
memory measurements.

Runtime provenance hashes the exact config and Transformers model implementation, Torch and CUDA
versions, tokenizer package and tokenizer-file bytes, every model-adapter capture field, attention
backend, device capability, numerical flags, CPU-offload/preload policy, dispatch audit, and each
resolved Nemotron kernel module and Hugging Face snapshot revision. Training, qualification, and QA
reject runtime hash mismatches. Nemotron's
optional runtime pins `kernels==0.15.2` and `einops==0.8.2`; changing the tokenizer, adapter capture
site, kernels, or fused path creates a distinct instrument even when model weights are unchanged.

This command creates an activation dataset; it does not qualify a probe. The 16-document C4 run
is a systems smoke and is smaller and less diverse than Appendix G. Local Nemotron BF16 weights
also differ from the NVFP4 checkpoint served by its OpenRouter free route, so results from the
local instrument must not be described as hosted-activation parity.

## Training and native qualification

The frozen workflow has separate training and qualification stages so a held-out neutral result
can be saved without being mistaken for a QA-eligible probe:

The 60-document diagnostic uses scikit-learn L-BFGS with a 2,000-iteration limit and `1e-4`
tolerance. The expanded workflow uses the pinned cuML QN backend from the released analysis and the
explicit `1e-3` tolerance selected by the train-only check above. Solver and tolerance are
implementation details rather than requirements stated in the paper, so the full runtime and
configuration are stored in the probe artifact.
For expanded neutral training, the command first binds the activation manifest's recorded content,
filler, sequence, and seed settings to the protocol's existing construction fields; it refuses a
changed artifact before loading its activation arrays. Native extraction validates the dialogue
partition and every request/response record before importing or loading the large model runtime.

```bash
uv run reasonese-role-probe train \
  --adapter nemotron-3.5-lightning-native-v1 \
  --activations out/role-probe-research/nemotron-neutral-activations \
  --protocol out/role-probe-research/protocol.json \
  --output out/role-probe-research/nemotron-neutral-probe.npz
```

`extract-native` replays exactly one of the 12-conversation frozen partitions. It reads all raw
dialogue files but selects only records assigned to the requested partition, verifies the original
request fields, and stores content-token activations without copying prompt, reasoning, or final
text into the activation artifact. It requires the exact prefix checkpoint and captures every
candidate layer named by the frozen protocol:

```bash
uv run reasonese-role-probe extract-native \
  --adapter nemotron-3.5-lightning-native-v1 \
  --checkpoint out/role-probe-research/prefix-nemotron \
  --dialogue-dir out/role-probe-research \
  --dialogue-glob 'native-0-*.json' \
  --prompt-partitions out/role-probe-research/native-prompt-partitions.json \
  --protocol out/role-probe-research/protocol.json \
  --split calibration \
  --output out/role-probe-research/nemotron-native-calibration
```

After extracting calibration and test partitions independently, `qualify` calibrates on the first,
evaluates the frozen test once, binds both activation fingerprints and the prompt-partition digest,
and writes a new artifact. The output is QA-eligible only when every preregistered neutral and native
gate passes:

```bash
uv run reasonese-role-probe qualify \
  --probe out/role-probe-research/nemotron-neutral-probe.npz \
  --calibration out/role-probe-research/nemotron-native-calibration \
  --test out/role-probe-research/nemotron-native-test \
  --prompt-partitions out/role-probe-research/native-prompt-partitions.json \
  --protocol out/role-probe-research/protocol.json \
  --output out/role-probe-research/nemotron-qualified-probe.npz
```

All three workflow stages refuse an existing output path before loading activations, fitting a
classifier, or loading a model. A changed protocol or signal therefore requires a new artifact
name and cannot silently replace earlier evidence.

## Collection gate

Collection accepts an explicit JSON bundle file through `--role-probes`. Paths are relative to the
bundle file, and every requested assistant must have exactly one matching entry:

```json
{
  "bundles": [
    {
      "assistant": "Nemotron 3.5 Lightning",
      "adapter": "nemotron-3.5-lightning-native-v1",
      "checkpoint": "prefix-nemotron",
      "probe": "nemotron-qualified-probe.npz"
    }
  ]
}
```

Before authoring or any other provider call, the collector loads the probe, requires all neutral
and untouched-native gates to pass, validates the assistant/model/template/site/layer/dtype and the
complete prefix-checkpoint file identity, and fails if any assistant lacks a bundle. Scoring renders
both instruction spans in each of the two exact ordered contexts with thinking enabled and the three
local function declarations. The OpenRouter web-search tool is injected server-side and has no
reproducible local prompt representation; every report records this limitation.

Each ordered context uses one prefix forward for both target spans. Assistants are processed one at
a time and the prefix model is released before the next assistant is loaded. The separate readable
`probe_qa_cache.json` key covers the probe artifact, model/runtime identity, adapter, render/tool
configuration, full context token IDs, target positions and IDs, assistant, order, and position. It
does not store instruction text or hidden reasoning.

```bash
uv run reasonese-collect-studies \
  --suite out/pilot/studies.yaml \
  --output out/pilot \
  --role-probes out/role-probe-research/probe-bundles.json
```
