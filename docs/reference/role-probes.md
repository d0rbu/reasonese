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

Acceptance thresholds are explicit configuration. The paper does not prescribe numerical
cutoffs, so a run must record its thresholds rather than attributing them to the paper. Synthetic
unit tests establish implementation behavior only. They do not qualify a model probe.

Local open-weight activations also do not establish exact parity with an OpenRouter deployment
unless the hosted checkpoint, quantization, tokenizer, and template are all independently shown
to match. Probe reports must preserve that limitation.

## Controlled training data

Each neutral base document is rendered five times with the exact target checkpoint's native role
templates: `system`, `user`, `tool`, `reasoning`, and `assistant`. The role labels vary while the
content is held constant. Reasoning models commonly nest reasoning and final output inside one
assistant envelope. In that case, assistant content follows variable-length closed reasoning
filler, and other roles receive position-matched filler. Role tags and filler tokens are excluded
from the activations used for fitting.

`ActivationDataset` verifies the controls before training:

- every neutral document contains every configured role;
- ordered content token IDs are identical across all role copies;
- relative content indices are contiguous;
- absolute sequence positions match across the role copies;
- every activation is finite and has the recorded dtype, layer count, and hidden size; and
- the metadata says that only content tokens remain.

Base document IDs, rather than rendered sequence IDs or token rows, are split into train,
validation, and test sets. All five copies of one document therefore remain in one split. This is
stricter than splitting role variants independently, which would leak the same underlying content
across the holdout boundary.

The provenance includes exact model and tokenizer revisions, a digest of the resolved weight
manifest, the native template digest, the model-specific activation site and dtype, the neutral
corpus digest, layers, masking counts, and extraction protocol. A probe for one model pipeline
cannot be qualified with conversation activations from another.

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
same exact local checkpoint and extraction path. It evaluates the five-class classifier on those
two observed roles. Predictions of `system`, `user`, or `tool` count as errors, while the absence
of those roles from an ordinary conversation is not itself an error. Both reasoning and assistant
tokens must be present.

Instruction scoring returns the full five-role probability vector for every content token and its
mean. The QA signal corresponding to the paper's CoTness is the unconditional mean probability of
the `reasoning` class. Other role probabilities remain available as controls. Compressed framing
scores are descriptive: compressed instructions do not have a required probe direction.

Probe artifacts are fingerprinted, pickle-free `.npz` files containing portable float64 weights
and JSON metadata. The fingerprint covers the weights, fit and validation metrics, split document
IDs, thresholds, and full provenance. Generated activation exports, fitted probes, and reports
belong below ignored `out/` directories rather than in source control.
