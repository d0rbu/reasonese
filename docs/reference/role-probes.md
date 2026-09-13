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

The provenance includes exact model and tokenizer revisions, a digest of the resolved weight
manifest, the native template digest, model and stored-activation dtypes, the model-specific
activation site, a runtime digest covering the software and kernel path, the neutral corpus
digest, layers, masking counts, and extraction protocol. A probe for one model pipeline cannot
be qualified with conversation activations from another. Projection also requires the exact
trained layer and stored dtype instead of accepting an unidentified two-dimensional array.

## Classifier and selection

The classifier is L2-regularized multinomial logistic regression. Its class space contains all
five roles, avoiding a forced reasoning-versus-final decision for text that the model represents
as user, system, or tool content. Regularization is selected on the grouped neutral validation
split using the configured Appendix-G lambda grid. Accuracy is primary, negative log likelihood
breaks ties, and the larger penalty is the final deterministic tie-break. The selected model is
refit on train plus validation documents, then evaluated once on held-out neutral documents.

The layer is preregistered in `ProbeTrainingConfig`; the test set and real conversations do not
select it. Metrics include token accuracy and negative log likelihood, per-role recall,
document-macro accuracy, per-role document-macro accuracy, and the full confusion matrix. The
document metrics prevent a handful of long documents from hiding failures elsewhere.

The fitted sklearn estimator is not serialized. Its coefficients and intercepts are converted to
a uniform multinomial softmax representation. Binary sklearn coefficients receive an algebraically
equivalent symmetric two-logit conversion, although production role probes use the five-role
space. Tests compare portable softmax probabilities directly against sklearn for both binary and
multiclass fits.

## Conversation qualification and scoring

Conversation qualification replays unmodified native reasoning and final-output text through the
same exact local checkpoint and extraction path. It uses 12 calibration conversations and 12
document- and content-disjoint test conversations. The calibration split alone selects a
deterministic threshold on the per-segment mean unconditional reasoning probability. The frozen
test checks pooled AUC with a paired-conversation, 10,000-replicate bootstrap as well as native
role recall and document-macro accuracy. The test split never selects the threshold. Exact
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
backend, device capability, numerical flags, and each resolved Nemotron kernel module and Hugging
Face snapshot revision. Training, qualification, and QA reject runtime hash mismatches. Nemotron's
optional runtime pins `kernels==0.15.2` and `einops==0.8.2`; changing the tokenizer, adapter capture
site, kernels, or fused path creates a distinct instrument even when model weights are unchanged.

This command creates an activation dataset; it does not qualify a probe. The 16-document C4 run
is a systems smoke and is smaller and less diverse than Appendix G. Local Nemotron BF16 weights
also differ from the NVFP4 checkpoint served by its OpenRouter free route, so results from the
local instrument must not be described as hosted-activation parity.

## Training and native qualification

The frozen workflow has separate training and qualification stages so a held-out neutral result
can be saved without being mistaken for a QA-eligible probe:

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
text into the activation artifact. It requires the exact prefix checkpoint and pinned layer:

```bash
uv run reasonese-role-probe extract-native \
  --adapter nemotron-3.5-lightning-native-v1 \
  --checkpoint out/role-probe-research/prefix-nemotron \
  --dialogue-dir out/role-probe-research \
  --dialogue-glob 'native-0-*.json' \
  --prompt-partitions out/role-probe-research/native-prompt-partitions.json \
  --protocol out/role-probe-research/protocol.json \
  --split calibration \
  --layer 26 \
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
