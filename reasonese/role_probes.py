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
import zipfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
from beartype import beartype
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from reasonese.probe_statistics import (
    CALIBRATION_SPLIT,
    EXPECTED_CONVERSATIONS,
    TEST_SPLIT,
    NativeQualification,
    PairedBootstrapAuc,
    PairedSegmentScores,
    ThresholdCalibration,
    calibrate_reasoning_threshold,
    qualify_native_test,
)

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
    runtime_sha256: str
    model_dtype: str
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
    filler_pool_kind: str
    filler_source_sha256: str | None
    filler_documents: int

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
            "model_dtype",
            "activation_dtype",
            "source_name",
            "extraction_protocol",
            "filler_pool_kind",
        ):
            _text(cast(str, getattr(self, name)), name)
        for name in (
            "weights_sha256",
            "chat_template_sha256",
            "runtime_sha256",
            "source_sha256",
        ):
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
        if self.dataset_kind == PAIRED_NEUTRAL:
            if self.filler_pool_kind != "dedicated-disjoint-documents":
                raise ValueError("neutral extraction requires a dedicated disjoint filler pool")
            if self.filler_source_sha256 is None:
                raise ValueError("neutral extraction requires filler source provenance")
            _sha256(self.filler_source_sha256, "filler_source_sha256")
            if self.filler_documents <= 0:
                raise ValueError("neutral extraction requires filler documents")
        elif (
            self.filler_pool_kind != "none"
            or self.filler_source_sha256 is not None
            or self.filler_documents != 0
        ):
            raise ValueError("conversation extraction cannot claim a neutral filler pool")


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
    filler_document_ids: Array

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
            "filler_document_ids": self.filler_document_ids,
        }
        for name, vector in vectors.items():
            if np.asarray(vector).ndim != 1 or len(vector) != values.shape[0]:
                raise ValueError(f"{name} must have one value per activation")
        for name in ("document_ids", "roles", "filler_document_ids"):
            if np.asarray(vectors[name]).dtype.kind not in {"U", "S"}:
                raise TypeError(f"{name} must use a string dtype")
        for name in ("content_token_index", "content_token_id", "sequence_token_index"):
            if not np.issubdtype(np.asarray(vectors[name]).dtype, np.integer):
                raise TypeError(f"{name} must use an integer dtype")
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
        else:
            required_conversation_roles = {"reasoning", "assistant"}
            for document in np.unique(self.document_ids):
                observed_for_document = set(self.roles[self.document_ids == document].tolist())
                if not required_conversation_roles.issubset(observed_for_document):
                    raise ValueError(
                        "every untouched conversation must contain reasoning and assistant tokens"
                    )


def _rows(dataset: ActivationDataset, document: str, role: str) -> Array:
    selected = np.flatnonzero((dataset.document_ids == document) & (dataset.roles == role))
    return selected[np.argsort(dataset.content_token_index[selected], kind="stable")]


def _validate_neutral_pairs(dataset: ActivationDataset) -> None:
    """Verify identical token content and controlled positions across roles."""
    targets = set(dataset.document_ids.tolist())
    fillers = set(dataset.filler_document_ids.tolist())
    if "" in fillers or targets & fillers:
        raise ValueError("neutral filler document IDs must be non-empty and disjoint from targets")
    if len(fillers) != dataset.provenance.filler_documents:
        raise ValueError("neutral filler document count does not match provenance")
    content_signatures: dict[tuple[int, ...], str] = {}
    for document in np.unique(dataset.document_ids):
        expected: tuple[Array, Array, Array] | None = None
        for role in dataset.provenance.roles:
            rows = _rows(dataset, str(document), role)
            if rows.size == 0:
                raise ValueError(f"neutral document {document!r} is missing role {role!r}")
            relative = dataset.content_token_index[rows]
            if not np.array_equal(relative, np.arange(rows.size)):
                raise ValueError("neutral content token indices must be contiguous from zero")
            actual = (
                dataset.content_token_id[rows],
                dataset.sequence_token_index[rows],
                dataset.filler_document_ids[rows],
            )
            if expected is None:
                expected = actual
            elif not all(
                np.array_equal(left, right) for left, right in zip(actual, expected, strict=True)
            ):
                raise ValueError(
                    "neutral role copies must contain identical tokens at identical positions"
                )
        assert expected is not None
        signature = tuple(int(value) for value in expected[0])
        duplicate = content_signatures.get(signature)
        if duplicate is not None:
            raise ValueError(
                f"neutral documents {duplicate!r} and {str(document)!r} contain identical tokens"
            )
        content_signatures[signature] = str(document)


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
class NativeProbeQualification:
    """Frozen calibration and untouched native-test evidence for QA use."""

    calibration_dataset_fingerprint: str
    test_dataset_fingerprint: str
    prompt_partition_name: str
    prompt_partition_sha256: str
    calibration_conversation_count: int
    calibration_scores: PairedSegmentScores
    calibration: ThresholdCalibration
    test_metrics: ClassificationMetrics
    test: NativeQualification

    def __post_init__(self) -> None:
        _sha256(self.calibration_dataset_fingerprint, "calibration_dataset_fingerprint")
        _sha256(self.test_dataset_fingerprint, "test_dataset_fingerprint")
        _text(self.prompt_partition_name, "prompt_partition_name")
        _sha256(self.prompt_partition_sha256, "prompt_partition_sha256")
        if self.calibration_dataset_fingerprint == self.test_dataset_fingerprint:
            raise ValueError("calibration and test activation datasets must be distinct")
        if self.calibration_conversation_count != EXPECTED_CONVERSATIONS:
            raise ValueError(f"native calibration requires {EXPECTED_CONVERSATIONS} conversations")
        if self.calibration_scores.split != CALIBRATION_SPLIT:
            raise ValueError("native calibration scores must use the calibration split")
        if self.calibration_scores.conversation_count != self.calibration_conversation_count:
            raise ValueError("native calibration score count does not match its provenance")
        if self.calibration.calibration_fingerprint != self.calibration_scores.fingerprint:
            raise ValueError("frozen threshold does not match the calibration scores")

    @property
    def passed(self) -> bool:
        return self.calibration.usable and self.test.passed


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
    qualification: NativeProbeQualification | None = None

    def __post_init__(self) -> None:
        expected = (len(self.provenance.roles), self.provenance.hidden_size)
        if np.asarray(self.coefficients).shape != expected:
            raise ValueError("probe coefficient shape does not match provenance")
        if np.asarray(self.intercepts).shape != (len(self.provenance.roles),):
            raise ValueError("probe intercept shape does not match provenance")
        if not np.isfinite(self.coefficients).all() or not np.isfinite(self.intercepts).all():
            raise ValueError("probe parameters must be finite")
        if not np.issubdtype(np.asarray(self.coefficients).dtype, np.floating) or not np.issubdtype(
            np.asarray(self.intercepts).dtype, np.floating
        ):
            raise TypeError("probe parameters must use floating dtypes")
        if self.training.layer_index not in self.provenance.layer_indices:
            raise ValueError("trained layer is absent from activation provenance")
        if self.regularization_lambda <= 0 or not math.isfinite(self.regularization_lambda):
            raise ValueError("regularization_lambda must be finite and positive")
        expected_roles = self.provenance.roles
        for name, metrics in (
            ("validation", self.validation_metrics),
            ("neutral test", self.neutral_test_metrics),
        ):
            if (
                tuple(role for role, _ in metrics.per_role_accuracy) != expected_roles
                or tuple(role for role, _ in metrics.per_role_document_accuracy) != expected_roles
            ):
                raise ValueError(f"{name} metrics do not match probe roles")
            if len(metrics.confusion_matrix) != len(expected_roles):
                raise ValueError(f"{name} confusion matrix does not match probe roles")
        if self.qualification is not None:
            measured = ("reasoning", "assistant")
            metrics = self.qualification.test_metrics
            if (
                tuple(role for role, _ in metrics.per_role_accuracy) != measured
                or tuple(role for role, _ in metrics.per_role_document_accuracy) != measured
            ):
                raise ValueError(
                    "conversation qualification metrics must measure reasoning and assistant"
                )
            if len(metrics.confusion_matrix) != len(expected_roles):
                raise ValueError("conversation confusion matrix does not match probe roles")

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
        "runtime_sha256",
        "model_dtype",
        "activation_dtype",
        "layer_indices",
        "hidden_size",
        "roles",
        "content_mask",
        "extraction_protocol",
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
        "filler_document_ids",
    ):
        _array_hash(hasher, name, cast(Array, getattr(dataset, name)))
    return hasher.hexdigest()


def _paired_segment_scores(
    probe: RoleProbe,
    dataset: ActivationDataset,
    split: str,
) -> PairedSegmentScores:
    layer_offset = dataset.provenance.layer_indices.index(probe.training.layer_index)
    probabilities = _predict(probe, dataset.activations[:, layer_offset])
    reasoning_index = probe.provenance.roles.index("reasoning")
    conversation_ids = tuple(str(value) for value in np.unique(dataset.document_ids))

    def scores_for(role: str) -> tuple[float, ...]:
        return tuple(
            float(
                probabilities[
                    (dataset.document_ids == document) & (dataset.roles == role), reasoning_index
                ].mean()
            )
            for document in conversation_ids
        )

    return PairedSegmentScores(
        split=split,
        conversation_ids=conversation_ids,
        reasoning_scores=scores_for("reasoning"),
        final_scores=scores_for("assistant"),
    )


def _conversation_content_signatures(
    dataset: ActivationDataset,
) -> set[tuple[tuple[int, ...], tuple[int, ...]]]:
    signatures: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    for document in np.unique(dataset.document_ids):
        segments: list[tuple[int, ...]] = []
        for role in ("reasoning", "assistant"):
            rows = _rows(dataset, str(document), role)
            segments.append(tuple(int(value) for value in dataset.content_token_id[rows]))
        signature = (segments[0], segments[1])
        if signature in signatures:
            raise ValueError("native conversations contain duplicated measured content")
        signatures.add(signature)
    return signatures


@beartype
def qualify_role_probe(
    probe: RoleProbe,
    calibration_conversations: ActivationDataset,
    test_conversations: ActivationDataset,
    *,
    prompt_partition_name: str,
    prompt_partition_sha256: str,
) -> RoleProbe:
    """Calibrate once and attach disjoint untouched native-test validity."""
    datasets = (calibration_conversations, test_conversations)
    if any(dataset.provenance.dataset_kind != UNTOUCHED_CONVERSATIONS for dataset in datasets):
        raise ValueError("qualification requires untouched native conversations")
    if any(not _same_pipeline(probe.provenance, dataset.provenance) for dataset in datasets):
        raise ValueError("conversation activations do not match the probe's exact model pipeline")
    if probe.qualification is not None:
        raise ValueError("probe is already qualified")
    neutral_documents = set(probe.split.train + probe.split.validation + probe.split.test)
    calibration_documents = set(calibration_conversations.document_ids.tolist())
    test_documents = set(test_conversations.document_ids.tolist())
    if neutral_documents & (calibration_documents | test_documents):
        raise ValueError("qualification conversations must be disjoint from neutral documents")
    if calibration_documents & test_documents:
        raise ValueError("calibration and test conversations must be document-disjoint")
    calibration_content = _conversation_content_signatures(calibration_conversations)
    test_content = _conversation_content_signatures(test_conversations)
    if calibration_content & test_content:
        raise ValueError("calibration and test conversations must be content-disjoint")

    calibration_scores = _paired_segment_scores(probe, calibration_conversations, CALIBRATION_SPLIT)
    test_scores = _paired_segment_scores(probe, test_conversations, TEST_SPLIT)
    if calibration_scores.conversation_count != EXPECTED_CONVERSATIONS:
        raise ValueError(f"native calibration requires {EXPECTED_CONVERSATIONS} conversations")
    layer_offset = test_conversations.provenance.layer_indices.index(probe.training.layer_index)
    rows = np.arange(len(test_conversations.roles))
    test_metrics = _metrics(
        _predict(probe, test_conversations.activations[:, layer_offset]),
        _labels(test_conversations, rows),
        probe.provenance.roles,
        test_conversations.document_ids,
        ("reasoning", "assistant"),
    )
    qualification = NativeProbeQualification(
        calibration_dataset_fingerprint=activation_dataset_fingerprint(calibration_conversations),
        test_dataset_fingerprint=activation_dataset_fingerprint(test_conversations),
        prompt_partition_name=prompt_partition_name,
        prompt_partition_sha256=prompt_partition_sha256,
        calibration_conversation_count=calibration_scores.conversation_count,
        calibration_scores=calibration_scores,
        calibration=calibrate_reasoning_threshold(calibration_scores),
        test_metrics=test_metrics,
        test=qualify_native_test(
            test_scores,
            minimum_role_accuracy=test_metrics.minimum_role_accuracy,
            document_macro_accuracy=test_metrics.document_accuracy,
        ),
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
    provenance: ActivationProvenance,
    layer_index: int,
    require_qa_eligible: bool = True,
) -> RoleProjection:
    """Project content-token activations from the probe's selected layer."""
    if not _same_pipeline(probe.provenance, provenance):
        raise ValueError("projection activations do not match the probe's exact model pipeline")
    if layer_index != probe.training.layer_index:
        raise ValueError("projection layer does not match the trained probe layer")
    if require_qa_eligible and not probe.qa_eligible:
        raise ValueError("probe lacks passing neutral and conversation validation")
    values = np.asarray(activations)
    if values.ndim != 2 or values.shape[1] != probe.provenance.hidden_size:
        raise ValueError("activations must have shape [tokens, hidden_size]")
    if values.shape[0] == 0 or not np.issubdtype(values.dtype, np.floating):
        raise ValueError("activations must be a non-empty floating array")
    if values.dtype.name != provenance.activation_dtype:
        raise ValueError("projection activation dtype does not match provenance")
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
    coefficients = np.asarray(probe.coefficients, dtype=np.float64)
    intercepts = np.asarray(probe.intercepts, dtype=np.float64)
    envelope = {
        "probe": metadata,
        "fingerprint": _artifact_fingerprint(metadata, coefficients, intercepts),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                metadata=np.frombuffer(_json(envelope), dtype=np.uint8),
                coefficients=coefficients,
                intercepts=intercepts,
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
        calibration = qualification["calibration"]
        calibration_scores = qualification["calibration_scores"]
        test = qualification["test"]
        scores = test["scores"]
        bootstrap = test["bootstrap_auc"]
        parsed_qualification = NativeProbeQualification(
            calibration_dataset_fingerprint=qualification["calibration_dataset_fingerprint"],
            test_dataset_fingerprint=qualification["test_dataset_fingerprint"],
            prompt_partition_name=qualification["prompt_partition_name"],
            prompt_partition_sha256=qualification["prompt_partition_sha256"],
            calibration_conversation_count=qualification["calibration_conversation_count"],
            calibration_scores=PairedSegmentScores(
                split=calibration_scores["split"],
                conversation_ids=tuple(calibration_scores["conversation_ids"]),
                reasoning_scores=tuple(calibration_scores["reasoning_scores"]),
                final_scores=tuple(calibration_scores["final_scores"]),
            ),
            calibration=ThresholdCalibration(**calibration),
            test_metrics=_metrics_from(qualification["test_metrics"]),
            test=NativeQualification(
                scores=PairedSegmentScores(
                    split=scores["split"],
                    conversation_ids=tuple(scores["conversation_ids"]),
                    reasoning_scores=tuple(scores["reasoning_scores"]),
                    final_scores=tuple(scores["final_scores"]),
                ),
                minimum_role_accuracy=test["minimum_role_accuracy"],
                document_macro_accuracy=test["document_macro_accuracy"],
                bootstrap_auc=PairedBootstrapAuc(**bootstrap),
            ),
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
    except (
        EOFError,
        KeyError,
        OSError,
        TypeError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
    ) as error:
        raise ValueError(f"could not load role-probe artifact {path}") from error
