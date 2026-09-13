"""Train and apply activation-based classifiers of model-native roles.

The probe construction follows Ye et al. (2026): render identical neutral
content under native role wrappers, retain only content-token activations, and
fit an L2 multinomial logistic regression. Model-specific extraction is a
separate boundary; this module validates its paired-content contract, keeps
base documents disjoint across splits, and stores portable NumPy weights.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
from beartype import beartype
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

PAIRED_NEUTRAL = "paired-neutral-role-wrappers"
UNTOUCHED_CONVERSATIONS = "untouched-native-conversations"
CONTENT_TOKENS_ONLY = "content-tokens-only"
_ARTIFACT_FORMAT = "reasonese-activation-role-probe"

type Array = np.ndarray


def _text(value: str, name: str) -> None:
    if not value or value.strip() != value:
        raise ValueError(f"{name} must be non-empty and trimmed")


def _sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@beartype
@dataclass(frozen=True, slots=True)
class ActivationProvenance:
    """Identity and extraction contract for one activation dataset."""

    dataset_kind: str
    model_id: str
    model_revision: str
    weights_sha256: str
    weights_hash_kind: str
    tokenizer_id: str
    tokenizer_revision: str
    chat_template_sha256: str
    native_template_adapter: str
    activation_site: str
    activation_dtype: str
    layer_indices: tuple[int, ...]
    hidden_size: int
    roles: tuple[str, ...]
    source_name: str
    source_sha256: str
    extraction_protocol: str
    content_mask: str
    masked_control_tokens: int
    masked_filler_tokens: int

    def __post_init__(self) -> None:
        for name in (
            "dataset_kind",
            "model_id",
            "model_revision",
            "weights_hash_kind",
            "tokenizer_id",
            "tokenizer_revision",
            "native_template_adapter",
            "activation_site",
            "activation_dtype",
            "source_name",
            "extraction_protocol",
        ):
            _text(cast(str, getattr(self, name)), name)
        for name in ("weights_sha256", "chat_template_sha256", "source_sha256"):
            _sha256(cast(str, getattr(self, name)), name)
        if self.dataset_kind not in {PAIRED_NEUTRAL, UNTOUCHED_CONVERSATIONS}:
            raise ValueError(f"unknown activation dataset kind {self.dataset_kind!r}")
        if self.content_mask != CONTENT_TOKENS_ONLY:
            raise ValueError(f"content_mask must be {CONTENT_TOKENS_ONLY!r}")
        if not self.layer_indices or min(self.layer_indices) < 0:
            raise ValueError("layer_indices must contain non-negative layers")
        if len(set(self.layer_indices)) != len(self.layer_indices):
            raise ValueError("layer_indices must not contain duplicates")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if len(self.roles) < 2 or len(set(self.roles)) != len(self.roles):
            raise ValueError("roles must contain at least two distinct values")
        if not {"reasoning", "assistant"}.issubset(self.roles):
            raise ValueError("role space must include reasoning and assistant")
        for role in self.roles:
            _text(role, "role")
        if self.masked_control_tokens < 0 or self.masked_filler_tokens < 0:
            raise ValueError("masked token counts must be non-negative")


@beartype
@dataclass(frozen=True, slots=True)
class ActivationDataset:
    """Content-token activations with shape ``[tokens, layers, hidden]``."""

    provenance: ActivationProvenance
    activations: Array
    document_ids: Array
    roles: Array
    content_token_index: Array
    content_token_id: Array
    sequence_token_index: Array

    def __post_init__(self) -> None:
        values = np.asarray(self.activations)
        if values.ndim != 3 or values.shape[0] == 0:
            raise ValueError("activations must have non-empty shape [tokens, layers, hidden]")
        if values.shape[1:] != (
            len(self.provenance.layer_indices),
            self.provenance.hidden_size,
        ):
            raise ValueError("activation shape does not match layer and hidden-size provenance")
        if not np.issubdtype(values.dtype, np.floating):
            raise TypeError("activations must use a floating dtype")
        if values.dtype.name != self.provenance.activation_dtype:
            raise ValueError("activation dtype does not match provenance")
        if not np.isfinite(values).all():
            raise ValueError("activations must be finite")
        vectors = {
            "document_ids": self.document_ids,
            "roles": self.roles,
            "content_token_index": self.content_token_index,
            "content_token_id": self.content_token_id,
            "sequence_token_index": self.sequence_token_index,
        }
        for name, vector in vectors.items():
            if np.asarray(vector).ndim != 1 or len(vector) != values.shape[0]:
                raise ValueError(f"{name} must have one value per activation")
        if any(not value or value.strip() != value for value in self.document_ids.tolist()):
            raise ValueError("document_ids must be non-empty and trimmed")
        observed = set(self.roles.tolist())
        required = set(self.provenance.roles)
        if self.provenance.dataset_kind == PAIRED_NEUTRAL and observed != required:
            raise ValueError("neutral activation roles do not match provenance roles")
        if self.provenance.dataset_kind == UNTOUCHED_CONVERSATIONS and not {
            "reasoning",
            "assistant",
        }.issubset(observed):
            raise ValueError("conversation activations require reasoning and assistant tokens")
        if not observed.issubset(required):
            raise ValueError("activation contains a role outside its provenance")
        if np.any(self.content_token_index < 0) or np.any(self.sequence_token_index < 0):
            raise ValueError("token indices must be non-negative")
        if self.provenance.dataset_kind == PAIRED_NEUTRAL:
            _validate_neutral_pairs(self)


def _rows(dataset: ActivationDataset, document: str, role: str) -> Array:
    selected = np.flatnonzero((dataset.document_ids == document) & (dataset.roles == role))
    return selected[np.argsort(dataset.content_token_index[selected], kind="stable")]


def _validate_neutral_pairs(dataset: ActivationDataset) -> None:
    """Verify identical token content and controlled positions across roles."""
    for document in np.unique(dataset.document_ids):
        expected: tuple[Array, Array] | None = None
        for role in dataset.provenance.roles:
            rows = _rows(dataset, str(document), role)
            if rows.size == 0:
                raise ValueError(f"neutral document {document!r} is missing role {role!r}")
            relative = dataset.content_token_index[rows]
            if not np.array_equal(relative, np.arange(rows.size)):
                raise ValueError("neutral content token indices must be contiguous from zero")
            actual = (dataset.content_token_id[rows], dataset.sequence_token_index[rows])
            if expected is None:
                expected = actual
            elif not all(
                np.array_equal(left, right) for left, right in zip(actual, expected, strict=True)
            ):
                raise ValueError(
                    "neutral role copies must contain identical tokens at identical positions"
                )


@beartype
@dataclass(frozen=True, slots=True)
class DocumentSplit:
    """A deterministic and group-disjoint base-document split."""

    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def __post_init__(self) -> None:
        groups = tuple(set(part) for part in (self.train, self.validation, self.test))
        if any(not group for group in groups):
            raise ValueError("every document split must be non-empty")
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError("document splits must be disjoint")

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            _json({"train": self.train, "validation": self.validation, "test": self.test})
        ).hexdigest()


@beartype
def split_documents(
    document_ids: tuple[str, ...],
    *,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> DocumentSplit:
    """Assign all role copies of a base document to the same split."""
    if len(document_ids) < 3 or len(set(document_ids)) != len(document_ids):
        raise ValueError("at least three unique document_ids are required")
    fractions = (train_fraction, validation_fraction, 1 - train_fraction - validation_fraction)
    if any(not 0 < value < 1 for value in fractions):
        raise ValueError("split fractions must be positive and sum to one")
    ordered = tuple(
        sorted(
            document_ids,
            key=lambda value: hashlib.sha256(f"{seed}\0{value}".encode()).digest(),
        )
    )
    raw = np.asarray(fractions) * len(ordered)
    counts = np.maximum(np.floor(raw).astype(int), 1)
    while int(counts.sum()) > len(ordered):
        index = min(
            (item for item in range(3) if counts[item] > 1),
            key=lambda item: (raw[item] - counts[item], item),
        )
        counts[index] -= 1
    while int(counts.sum()) < len(ordered):
        index = max(range(3), key=lambda item: (raw[item] - counts[item], -item))
        counts[index] += 1
    train_count, validation_count = int(counts[0]), int(counts[1])
    return DocumentSplit(
        ordered[:train_count],
        ordered[train_count : train_count + validation_count],
        ordered[train_count + validation_count :],
    )


@beartype
@dataclass(frozen=True, slots=True)
class ProbeTrainingConfig:
    """Preregistered layer, split, search grid, and neutral acceptance gates."""

    layer_index: int
    minimum_neutral_accuracy: float
    minimum_neutral_per_role_accuracy: float
    minimum_neutral_document_accuracy: float
    minimum_neutral_per_role_document_accuracy: float
    lambda_grid: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3)
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    seed: int = 0
    max_iterations: int = 2_000
    tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if self.layer_index < 0:
            raise ValueError("layer_index must be non-negative")
        gates = (
            self.minimum_neutral_accuracy,
            self.minimum_neutral_per_role_accuracy,
            self.minimum_neutral_document_accuracy,
            self.minimum_neutral_per_role_document_accuracy,
        )
        if any(not 0 <= value <= 1 for value in gates):
            raise ValueError("neutral acceptance gates must lie between zero and one")
        if not self.lambda_grid or any(
            value <= 0 or not math.isfinite(value) for value in self.lambda_grid
        ):
            raise ValueError("lambda_grid must contain distinct finite positive values")
        if len(set(self.lambda_grid)) != len(self.lambda_grid):
            raise ValueError("lambda_grid must not contain duplicates")
        if self.max_iterations <= 0 or self.tolerance <= 0:
            raise ValueError("optimizer limits must be positive")


@beartype
@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Token and document-macro classifier diagnostics."""

    accuracy: float
    document_accuracy: float
    negative_log_likelihood: float
    per_role_accuracy: tuple[tuple[str, float], ...]
    per_role_document_accuracy: tuple[tuple[str, float], ...]
    confusion_matrix: tuple[tuple[int, ...], ...]
    token_count: int
    document_count: int

    def __post_init__(self) -> None:
        rates = (self.accuracy, self.document_accuracy) + tuple(
            value for _, value in self.per_role_accuracy + self.per_role_document_accuracy
        )
        if any(not 0 <= value <= 1 for value in rates):
            raise ValueError("classification rates must lie between zero and one")
        if self.token_count <= 0 or self.document_count <= 0:
            raise ValueError("classification counts must be positive")
        if self.negative_log_likelihood < 0 or not math.isfinite(self.negative_log_likelihood):
            raise ValueError("negative_log_likelihood must be finite and non-negative")
        if not self.per_role_accuracy or len(self.per_role_accuracy) != len(
            self.per_role_document_accuracy
        ):
            raise ValueError("per-role metrics must be non-empty and aligned")
        size = len(self.confusion_matrix)
        if size < 2 or any(len(row) != size for row in self.confusion_matrix):
            raise ValueError("confusion_matrix must be square")

    @property
    def minimum_role_accuracy(self) -> float:
        return min(value for _, value in self.per_role_accuracy)

    @property
    def minimum_role_document_accuracy(self) -> float:
        return min(value for _, value in self.per_role_document_accuracy)


@beartype
@dataclass(frozen=True, slots=True)
class QualificationThresholds:
    """Preregistered gates for untouched-conversation generalization."""

    minimum_accuracy: float
    minimum_per_role_accuracy: float
    minimum_document_accuracy: float
    minimum_per_role_document_accuracy: float

    def __post_init__(self) -> None:
        if any(not 0 <= value <= 1 for value in asdict(self).values()):
            raise ValueError("conversation acceptance gates must lie between zero and one")


@beartype
@dataclass(frozen=True, slots=True)
class ConversationQualification:
    """Measured zero-shot performance on untouched native conversations."""

    dataset_fingerprint: str
    source_name: str
    conversation_count: int
    metrics: ClassificationMetrics
    thresholds: QualificationThresholds

    def __post_init__(self) -> None:
        _sha256(self.dataset_fingerprint, "dataset_fingerprint")
        _text(self.source_name, "source_name")
        if self.conversation_count <= 0:
            raise ValueError("conversation_count must be positive")

    @property
    def passed(self) -> bool:
        return (
            self.metrics.accuracy >= self.thresholds.minimum_accuracy
            and self.metrics.minimum_role_accuracy >= self.thresholds.minimum_per_role_accuracy
            and self.metrics.document_accuracy >= self.thresholds.minimum_document_accuracy
            and self.metrics.minimum_role_document_accuracy
            >= self.thresholds.minimum_per_role_document_accuracy
        )


@beartype
@dataclass(frozen=True, slots=True)
class RoleProbe:
    """One predeclared layer's portable role classifier and evidence."""

    provenance: ActivationProvenance
    training: ProbeTrainingConfig
    split: DocumentSplit
    regularization_lambda: float
    coefficients: Array
    intercepts: Array
    validation_metrics: ClassificationMetrics
    neutral_test_metrics: ClassificationMetrics
    qualification: ConversationQualification | None = None

    def __post_init__(self) -> None:
        expected = (len(self.provenance.roles), self.provenance.hidden_size)
        if np.asarray(self.coefficients).shape != expected:
            raise ValueError("probe coefficient shape does not match provenance")
        if np.asarray(self.intercepts).shape != (len(self.provenance.roles),):
            raise ValueError("probe intercept shape does not match provenance")
        if not np.isfinite(self.coefficients).all() or not np.isfinite(self.intercepts).all():
            raise ValueError("probe parameters must be finite")
        if self.training.layer_index not in self.provenance.layer_indices:
            raise ValueError("trained layer is absent from activation provenance")
        if self.regularization_lambda <= 0 or not math.isfinite(self.regularization_lambda):
            raise ValueError("regularization_lambda must be finite and positive")

    @property
    def neutral_valid(self) -> bool:
        metrics = self.neutral_test_metrics
        return (
            metrics.accuracy >= self.training.minimum_neutral_accuracy
            and metrics.minimum_role_accuracy >= self.training.minimum_neutral_per_role_accuracy
            and metrics.document_accuracy >= self.training.minimum_neutral_document_accuracy
            and metrics.minimum_role_document_accuracy
            >= self.training.minimum_neutral_per_role_document_accuracy
        )

    @property
    def qa_eligible(self) -> bool:
        return self.neutral_valid and self.qualification is not None and self.qualification.passed


def _select_rows(dataset: ActivationDataset, documents: tuple[str, ...]) -> Array:
    return np.flatnonzero(np.isin(dataset.document_ids, documents))


def _labels(dataset: ActivationDataset, rows: Array) -> Array:
    indices = {role: index for index, role in enumerate(dataset.provenance.roles)}
    return np.asarray([indices[str(role)] for role in dataset.roles[rows]], dtype=np.int64)


def _softmax(logits: Array) -> Array:
    shifted = np.asarray(logits, dtype=np.float64).copy()
    shifted -= shifted.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _predict_parameters(coefficients: Array, intercepts: Array, activations: Array) -> Array:
    return _softmax(np.asarray(activations, dtype=np.float64) @ coefficients.T + intercepts)


def _predict(probe: RoleProbe, activations: Array) -> Array:
    return _predict_parameters(probe.coefficients, probe.intercepts, activations)


def _metrics(
    probabilities: Array,
    labels: Array,
    roles: tuple[str, ...],
    documents: Array,
    evaluated_roles: tuple[str, ...] | None = None,
) -> ClassificationMetrics:
    measured_roles = roles if evaluated_roles is None else evaluated_roles
    measured_indices = tuple(roles.index(role) for role in measured_roles)
    predicted = np.argmax(probabilities, axis=1)
    confusion = np.zeros((len(roles), len(roles)), dtype=np.int64)
    np.add.at(confusion, (labels, predicted), 1)
    supports = confusion.sum(axis=1)
    if any(supports[index] == 0 for index in measured_indices):
        raise ValueError("evaluation data must contain every measured role")
    correct = predicted == labels
    unique_documents = np.unique(documents)
    per_role = tuple(
        (role, float(confusion[index, index] / supports[index]))
        for role, index in zip(measured_roles, measured_indices, strict=True)
    )
    per_role_document = tuple(
        (
            role,
            float(
                np.mean(
                    [
                        np.mean(correct[(documents == document) & (labels == index)])
                        for document in unique_documents
                        if np.any((documents == document) & (labels == index))
                    ]
                )
            ),
        )
        for role, index in zip(measured_roles, measured_indices, strict=True)
    )
    likelihoods = np.clip(probabilities[np.arange(labels.size), labels], 1e-15, 1.0)
    return ClassificationMetrics(
        accuracy=float(correct.mean()),
        document_accuracy=float(
            np.mean([correct[documents == document].mean() for document in unique_documents])
        ),
        negative_log_likelihood=float(-np.log(likelihoods).mean()),
        per_role_accuracy=per_role,
        per_role_document_accuracy=per_role_document,
        confusion_matrix=tuple(tuple(int(value) for value in row) for row in confusion),
        token_count=int(labels.size),
        document_count=int(unique_documents.size),
    )


def _fit(
    x: Array, y: Array, regularization: float, config: ProbeTrainingConfig
) -> LogisticRegression:
    classifier = LogisticRegression(
        C=1 / regularization,
        solver="lbfgs",
        fit_intercept=True,
        max_iter=config.max_iterations,
        tol=config.tolerance,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            classifier.fit(np.asarray(x, dtype=np.float64), y)
        except ConvergenceWarning as error:
            raise RuntimeError(f"probe failed to converge for lambda={regularization:g}") from error
    return classifier


def _parameters(classifier: LogisticRegression, class_count: int) -> tuple[Array, Array]:
    if not np.array_equal(classifier.classes_, np.arange(class_count)):
        raise ValueError("classifier class order does not match role order")
    coefficients = np.asarray(classifier.coef_, dtype=np.float64)
    intercepts = np.asarray(classifier.intercept_, dtype=np.float64)
    if class_count == 2:
        coefficients = np.concatenate((-coefficients / 2, coefficients / 2))
        intercepts = np.concatenate((-intercepts / 2, intercepts / 2))
    return coefficients, intercepts


@beartype
def train_role_probe(dataset: ActivationDataset, config: ProbeTrainingConfig) -> RoleProbe:
    """Select L2 strength on grouped neutral dev data and test once."""
    if dataset.provenance.dataset_kind != PAIRED_NEUTRAL:
        raise ValueError("training requires paired neutral-role activations")
    try:
        layer_offset = dataset.provenance.layer_indices.index(config.layer_index)
    except ValueError as error:
        raise ValueError("configured layer is absent from activation export") from error
    split = split_documents(
        tuple(str(value) for value in np.unique(dataset.document_ids)),
        seed=config.seed,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
    )
    train_rows = _select_rows(dataset, split.train)
    validation_rows = _select_rows(dataset, split.validation)
    test_rows = _select_rows(dataset, split.test)
    train_labels = _labels(dataset, train_rows)
    validation_labels = _labels(dataset, validation_rows)

    candidates: list[tuple[float, ClassificationMetrics]] = []
    for regularization in config.lambda_grid:
        classifier = _fit(
            dataset.activations[train_rows, layer_offset], train_labels, regularization, config
        )
        coefficients, intercepts = _parameters(classifier, len(dataset.provenance.roles))
        probabilities = _predict_parameters(
            coefficients, intercepts, dataset.activations[validation_rows, layer_offset]
        )
        candidates.append(
            (
                regularization,
                _metrics(
                    probabilities,
                    validation_labels,
                    dataset.provenance.roles,
                    dataset.document_ids[validation_rows],
                ),
            )
        )
    regularization, validation_metrics = max(
        candidates,
        key=lambda item: (item[1].accuracy, -item[1].negative_log_likelihood, item[0]),
    )
    fit_rows = np.concatenate((train_rows, validation_rows))
    classifier = _fit(
        dataset.activations[fit_rows, layer_offset],
        _labels(dataset, fit_rows),
        regularization,
        config,
    )
    coefficients, intercepts = _parameters(classifier, len(dataset.provenance.roles))
    test_probabilities = _predict_parameters(
        coefficients, intercepts, dataset.activations[test_rows, layer_offset]
    )
    return RoleProbe(
        provenance=dataset.provenance,
        training=config,
        split=split,
        regularization_lambda=regularization,
        coefficients=coefficients,
        intercepts=intercepts,
        validation_metrics=validation_metrics,
        neutral_test_metrics=_metrics(
            test_probabilities,
            _labels(dataset, test_rows),
            dataset.provenance.roles,
            dataset.document_ids[test_rows],
        ),
    )


def _same_pipeline(left: ActivationProvenance, right: ActivationProvenance) -> bool:
    fields = (
        "model_id",
        "model_revision",
        "weights_sha256",
        "weights_hash_kind",
        "tokenizer_id",
        "tokenizer_revision",
        "chat_template_sha256",
        "native_template_adapter",
        "activation_site",
        "activation_dtype",
        "layer_indices",
        "hidden_size",
        "roles",
        "content_mask",
    )
    return all(getattr(left, field) == getattr(right, field) for field in fields)


def _array_hash(hasher: Any, name: str, value: Array) -> None:
    array = np.ascontiguousarray(value)
    hasher.update(name.encode())
    hasher.update(array.dtype.str.encode())
    hasher.update(_json(array.shape))
    hasher.update(array.tobytes())


@beartype
def activation_dataset_fingerprint(dataset: ActivationDataset) -> str:
    """Hash exact provenance, activations, labels, and token alignment."""
    hasher = hashlib.sha256(_json(asdict(dataset.provenance)))
    for name in (
        "activations",
        "document_ids",
        "roles",
        "content_token_index",
        "content_token_id",
        "sequence_token_index",
    ):
        _array_hash(hasher, name, cast(Array, getattr(dataset, name)))
    return hasher.hexdigest()


@beartype
def qualify_role_probe(
    probe: RoleProbe,
    conversations: ActivationDataset,
    thresholds: QualificationThresholds,
) -> RoleProbe:
    """Measure and attach untouched reasoning/final conversation validity."""
    if conversations.provenance.dataset_kind != UNTOUCHED_CONVERSATIONS:
        raise ValueError("qualification requires untouched native conversations")
    if not _same_pipeline(probe.provenance, conversations.provenance):
        raise ValueError("conversation activations do not match the probe's exact model pipeline")
    if probe.qualification is not None:
        raise ValueError("probe is already qualified")
    layer_offset = conversations.provenance.layer_indices.index(probe.training.layer_index)
    rows = np.arange(len(conversations.roles))
    qualification = ConversationQualification(
        dataset_fingerprint=activation_dataset_fingerprint(conversations),
        source_name=conversations.provenance.source_name,
        conversation_count=len(set(conversations.document_ids.tolist())),
        metrics=_metrics(
            _predict(probe, conversations.activations[:, layer_offset]),
            _labels(conversations, rows),
            probe.provenance.roles,
            conversations.document_ids,
            ("reasoning", "assistant"),
        ),
        thresholds=thresholds,
    )
    return replace(probe, qualification=qualification)


@beartype
@dataclass(frozen=True, slots=True)
class RoleProjection:
    """Full role probabilities with the paper's unconditional CoTness score."""

    roles: tuple[str, ...]
    token_probabilities: Array
    mean_probabilities: tuple[float, ...]
    predicted_role: str
    mean_reasoning_probability: float


@beartype
def project_role(
    probe: RoleProbe,
    activations: Array,
    *,
    require_qa_eligible: bool = True,
) -> RoleProjection:
    """Project content-token activations from the probe's selected layer."""
    if require_qa_eligible and not probe.qa_eligible:
        raise ValueError("probe lacks passing neutral and conversation validation")
    values = np.asarray(activations)
    if values.ndim != 2 or values.shape[1] != probe.provenance.hidden_size:
        raise ValueError("activations must have shape [tokens, hidden_size]")
    if values.shape[0] == 0 or not np.issubdtype(values.dtype, np.floating):
        raise ValueError("activations must be a non-empty floating array")
    if not np.isfinite(values).all():
        raise ValueError("activations must be finite")
    probabilities = _predict(probe, values)
    means = tuple(float(value) for value in probabilities.mean(axis=0))
    reasoning = probe.provenance.roles.index("reasoning")
    return RoleProjection(
        roles=probe.provenance.roles,
        token_probabilities=probabilities,
        mean_probabilities=means,
        predicted_role=probe.provenance.roles[int(np.argmax(means))],
        mean_reasoning_probability=means[reasoning],
    )


def _json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def _metrics_from(raw: dict[str, Any]) -> ClassificationMetrics:
    return ClassificationMetrics(
        accuracy=raw["accuracy"],
        document_accuracy=raw["document_accuracy"],
        negative_log_likelihood=raw["negative_log_likelihood"],
        per_role_accuracy=tuple(tuple(item) for item in raw["per_role_accuracy"]),
        per_role_document_accuracy=tuple(tuple(item) for item in raw["per_role_document_accuracy"]),
        confusion_matrix=tuple(tuple(row) for row in raw["confusion_matrix"]),
        token_count=raw["token_count"],
        document_count=raw["document_count"],
    )


def _metadata(probe: RoleProbe) -> dict[str, object]:
    data = asdict(probe)
    del data["coefficients"]
    del data["intercepts"]
    data["format"] = _ARTIFACT_FORMAT
    return data


def _artifact_fingerprint(
    metadata: dict[str, object], coefficients: Array, intercepts: Array
) -> str:
    hasher = hashlib.sha256(_json(metadata))
    _array_hash(hasher, "coefficients", coefficients)
    _array_hash(hasher, "intercepts", intercepts)
    return hasher.hexdigest()


@beartype
def save_role_probe(probe: RoleProbe, path: Path) -> None:
    """Atomically write a fingerprinted artifact without pickled code."""
    metadata = _metadata(probe)
    envelope = {
        "probe": metadata,
        "fingerprint": _artifact_fingerprint(metadata, probe.coefficients, probe.intercepts),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                metadata=np.frombuffer(_json(envelope), dtype=np.uint8),
                coefficients=np.asarray(probe.coefficients, dtype=np.float64),
                intercepts=np.asarray(probe.intercepts, dtype=np.float64),
            )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _probe_from(metadata: dict[str, Any], coefficients: Array, intercepts: Array) -> RoleProbe:
    provenance = dict(metadata["provenance"])
    provenance["layer_indices"] = tuple(provenance["layer_indices"])
    provenance["roles"] = tuple(provenance["roles"])
    training = dict(metadata["training"])
    training["lambda_grid"] = tuple(training["lambda_grid"])
    split = {name: tuple(value) for name, value in metadata["split"].items()}
    qualification = metadata["qualification"]
    parsed_qualification = None
    if qualification is not None:
        parsed_qualification = ConversationQualification(
            dataset_fingerprint=qualification["dataset_fingerprint"],
            source_name=qualification["source_name"],
            conversation_count=qualification["conversation_count"],
            metrics=_metrics_from(qualification["metrics"]),
            thresholds=QualificationThresholds(**qualification["thresholds"]),
        )
    return RoleProbe(
        provenance=ActivationProvenance(**provenance),
        training=ProbeTrainingConfig(**training),
        split=DocumentSplit(**split),
        regularization_lambda=metadata["regularization_lambda"],
        coefficients=coefficients,
        intercepts=intercepts,
        validation_metrics=_metrics_from(metadata["validation_metrics"]),
        neutral_test_metrics=_metrics_from(metadata["neutral_test_metrics"]),
        qualification=parsed_qualification,
    )


@beartype
def load_role_probe(path: Path) -> RoleProbe:
    """Load only an artifact whose exact content digest still matches."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"metadata", "coefficients", "intercepts"}:
                raise ValueError("invalid role-probe archive members")
            raw_metadata = np.asarray(archive["metadata"])
            coefficients = np.asarray(archive["coefficients"], dtype=np.float64)
            intercepts = np.asarray(archive["intercepts"], dtype=np.float64)
        if raw_metadata.ndim != 1 or raw_metadata.dtype != np.uint8:
            raise ValueError("role-probe metadata must be bytes")
        envelope = json.loads(raw_metadata.tobytes().decode())
        if not isinstance(envelope, dict) or set(envelope) != {"probe", "fingerprint"}:
            raise ValueError("invalid role-probe metadata envelope")
        metadata = envelope["probe"]
        if not isinstance(metadata, dict) or metadata.get("format") != _ARTIFACT_FORMAT:
            raise ValueError("unsupported role-probe format")
        expected = _artifact_fingerprint(metadata, coefficients, intercepts)
        if envelope["fingerprint"] != expected:
            raise ValueError("role-probe artifact fingerprint does not match its contents")
        del metadata["format"]
        return _probe_from(metadata, coefficients, intercepts)
    except (KeyError, OSError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load role-probe artifact {path}") from error
