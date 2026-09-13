"""Train and apply activation-based classifiers of model-native roles.

The probe construction follows Ye et al. (2026): render identical neutral
content under native role wrappers, retain only content-token activations, and
fit an L2 multinomial logistic regression. Model-specific extraction is a
separate boundary; this module validates its paired-content contract, keeps
base documents disjoint across splits, and stores portable NumPy weights.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import io
import json
import logging
import math
import os
import platform
import time
import warnings
import zipfile
from contextlib import redirect_stdout
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
    MIN_THRESHOLD_FINAL_SPECIFICITY,
    MIN_THRESHOLD_REASONING_SENSITIVITY,
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
_ARTIFACT_FORMAT = "reasonese-activation-role-probe-v2"
logger = logging.getLogger(__name__)

type Array = np.ndarray


def _text(value: str, name: str) -> None:
    if not value or value.strip() != value:
        raise ValueError(f"{name} must be non-empty and trimmed")


def _sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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


def _canonical_json_text(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


@beartype
@dataclass(frozen=True, slots=True)
class ProbeOptimizerConfig:
    """Exact fitting backend, numerical settings, and auditable runtime identity."""

    backend: str
    solver: str
    fit_dtype: str
    max_iterations: int
    tolerance: float
    linesearch_max_iterations: int | None
    lbfgs_memory: int | None
    penalty_normalized: bool | None
    runtime_json: str
    runtime_sha256: str

    def __post_init__(self) -> None:
        if self.backend not in {"sklearn-lbfgs", "cuml-qn"}:
            raise ValueError("unsupported probe optimizer backend")
        expected = (
            ("lbfgs", "float64", None, None, None)
            if self.backend == "sklearn-lbfgs"
            else ("qn", "float32", 100, 5, True)
        )
        actual = (
            self.solver,
            self.fit_dtype,
            self.linesearch_max_iterations,
            self.lbfgs_memory,
            self.penalty_normalized,
        )
        if actual != expected:
            raise ValueError("probe optimizer settings do not match the selected backend")
        if type(self.max_iterations) is not int or self.max_iterations <= 0:
            raise ValueError("optimizer max_iterations must be a positive integer")
        if type(self.tolerance) not in {int, float} or (
            self.tolerance <= 0 or not math.isfinite(self.tolerance)
        ):
            raise ValueError("optimizer tolerance must be finite and positive")
        try:
            runtime = json.loads(self.runtime_json)
        except json.JSONDecodeError as error:
            raise ValueError("optimizer runtime_json is invalid") from error
        if not isinstance(runtime, dict) or self.runtime_json != _canonical_json_text(runtime):
            raise ValueError("optimizer runtime_json must be a canonical JSON object")
        _sha256(self.runtime_sha256, "optimizer runtime_sha256")
        if hashlib.sha256(self.runtime_json.encode()).hexdigest() != self.runtime_sha256:
            raise ValueError("optimizer runtime digest does not match its record")
        expected_solver = (
            {
                "class_weight": None,
                "fit_intercept": True,
                "max_iter": self.max_iterations,
                "penalty": "l2",
                "solver": self.solver,
                "tol": self.tolerance,
            }
            if self.backend == "sklearn-lbfgs"
            else {
                "class_weight": None,
                "fit_intercept": True,
                "lbfgs_memory": self.lbfgs_memory,
                "linesearch_max_iter": self.linesearch_max_iterations,
                "l1_ratio": None,
                "max_iter": self.max_iterations,
                "output_type": "cupy",
                "penalty": "l2",
                "penalty_normalized": self.penalty_normalized,
                "solver": self.solver,
                "tol": self.tolerance,
                "verbose": False,
            }
        )
        expected_matrix = {
            "activation_dtype": self.fit_dtype,
            "array_order": "F" if self.backend == "cuml-qn" else "C",
            "convert_dtype": False if self.backend == "cuml-qn" else None,
            "labels_dtype": "int32" if self.backend == "cuml-qn" else "int64",
        }
        if (
            runtime.get("backend") != self.backend
            or runtime.get("solver") != expected_solver
            or runtime.get("matrix") != expected_matrix
            or runtime.get("regularization_mapping") != "C=1/lambda"
        ):
            raise ValueError("optimizer runtime record does not match its numerical settings")


def _optimizer_config(
    record: dict[str, object],
    *,
    backend: str,
    solver: str,
    fit_dtype: str,
    max_iterations: int,
    tolerance: float,
    linesearch_max_iterations: int | None,
    lbfgs_memory: int | None,
    penalty_normalized: bool | None,
) -> ProbeOptimizerConfig:
    runtime_json = _canonical_json_text(record)
    return ProbeOptimizerConfig(
        backend=backend,
        solver=solver,
        fit_dtype=fit_dtype,
        max_iterations=max_iterations,
        tolerance=tolerance,
        linesearch_max_iterations=linesearch_max_iterations,
        lbfgs_memory=lbfgs_memory,
        penalty_normalized=penalty_normalized,
        runtime_json=runtime_json,
        runtime_sha256=hashlib.sha256(runtime_json.encode()).hexdigest(),
    )


@beartype
def sklearn_optimizer_config(
    *, max_iterations: int = 2_000, tolerance: float = 1e-4
) -> ProbeOptimizerConfig:
    """Identify the exact CPU diagnostic solver without loading GPU dependencies."""
    module = importlib.import_module("sklearn.linear_model._logistic")
    source = getattr(module, "__file__", None)
    if not isinstance(source, str) or not Path(source).is_file():
        raise ValueError("sklearn logistic-regression implementation is not hashable")
    record: dict[str, object] = {
        "backend": "sklearn-lbfgs",
        "format_version": 1,
        "implementation_sha256": {"sklearn_logistic_py": _path_sha256(Path(source))},
        "matrix": {
            "activation_dtype": "float64",
            "array_order": "C",
            "convert_dtype": None,
            "labels_dtype": "int64",
        },
        "packages": {
            "numpy": np.__version__,
            "scikit-learn": importlib.metadata.version("scikit-learn"),
            "scipy": importlib.metadata.version("scipy"),
        },
        "python_version": platform.python_version(),
        "regularization_mapping": "C=1/lambda",
        "solver": {
            "class_weight": None,
            "fit_intercept": True,
            "max_iter": max_iterations,
            "penalty": "l2",
            "solver": "lbfgs",
            "tol": tolerance,
        },
    }
    return _optimizer_config(
        record,
        backend="sklearn-lbfgs",
        solver="lbfgs",
        fit_dtype="float64",
        max_iterations=max_iterations,
        tolerance=tolerance,
        linesearch_max_iterations=None,
        lbfgs_memory=None,
        penalty_normalized=None,
    )


def _distribution_file(distribution: str, relative_path: str) -> Path:
    path = Path(str(importlib.metadata.distribution(distribution).locate_file(relative_path)))
    if not path.is_file():
        raise ValueError(f"optimizer runtime file is missing: {relative_path}")
    return path


@beartype
def cuml_optimizer_config(
    *,
    max_iterations: int = 5_000,
    tolerance: float = 1e-4,
    linesearch_max_iterations: int = 100,
    lbfgs_memory: int = 5,
) -> ProbeOptimizerConfig:
    """Identify the exact explicit cuML QN runtime selected for expanded fitting."""
    try:
        cupy = importlib.import_module("cupy")
        logistic = importlib.import_module("cuml.linear_model.logistic_regression")
        qn = importlib.import_module("cuml.solvers.qn")
    except ImportError as error:
        raise RuntimeError(
            "cuML role-probe fitting requires the pinned isolated GPU environment documented "
            "in docs/reference/role-probes.md"
        ) from error
    logistic_path = getattr(logistic, "__file__", None)
    qn_path = getattr(qn, "__file__", None)
    if not isinstance(logistic_path, str) or not isinstance(qn_path, str):
        raise ValueError("cuML optimizer implementation is not hashable")
    device = int(cupy.cuda.runtime.getDevice())
    properties = cupy.cuda.runtime.getDeviceProperties(device)
    raw_name = properties["name"]
    device_name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
    packages = {
        name: importlib.metadata.version(name)
        for name in (
            "cuda-python",
            "cuda-toolkit",
            "cuml-cu13",
            "cupy-cuda13x",
            "libcuml-cu13",
            "nvidia-cuda-nvrtc",
            "nvidia-cuda-runtime",
            "rmm-cu13",
        )
    }
    packages["numpy"] = np.__version__
    record: dict[str, object] = {
        "backend": "cuml-qn",
        "cuda": {
            "compute_capability": f"{properties['major']}.{properties['minor']}",
            "device_name": device_name,
            "driver_version": int(cupy.cuda.runtime.driverGetVersion()),
            "runtime_version": int(cupy.cuda.runtime.runtimeGetVersion()),
        },
        "format_version": 1,
        "implementation_sha256": {
            "cuml/linear_model/logistic_regression.py": _path_sha256(Path(logistic_path)),
            "cuml/solvers/qn.abi3.so": _path_sha256(Path(qn_path)),
            "libcuml/lib64/libcuml.so": _path_sha256(
                _distribution_file("libcuml-cu13", "libcuml/lib64/libcuml.so")
            ),
        },
        "matrix": {
            "activation_dtype": "float32",
            "array_order": "F",
            "convert_dtype": False,
            "labels_dtype": "int32",
        },
        "packages": packages,
        "python_version": platform.python_version(),
        "regularization_mapping": "C=1/lambda",
        "solver": {
            "class_weight": None,
            "fit_intercept": True,
            "lbfgs_memory": lbfgs_memory,
            "linesearch_max_iter": linesearch_max_iterations,
            "l1_ratio": None,
            "max_iter": max_iterations,
            "output_type": "cupy",
            "penalty": "l2",
            "penalty_normalized": True,
            "solver": "qn",
            "tol": tolerance,
            "verbose": False,
        },
    }
    return _optimizer_config(
        record,
        backend="cuml-qn",
        solver="qn",
        fit_dtype="float32",
        max_iterations=max_iterations,
        tolerance=tolerance,
        linesearch_max_iterations=linesearch_max_iterations,
        lbfgs_memory=lbfgs_memory,
        penalty_normalized=True,
    )


@beartype
def optimizer_config_from_runtime(record: dict[str, Any]) -> ProbeOptimizerConfig:
    """Build a validated optimizer config from one frozen protocol runtime record."""
    try:
        backend = record["backend"]
        solver = record["solver"]
        matrix = record["matrix"]
        assert isinstance(solver, dict) and isinstance(matrix, dict)
        return _optimizer_config(
            record,
            backend=backend,
            solver=solver["solver"],
            fit_dtype=matrix["activation_dtype"],
            max_iterations=solver["max_iter"],
            tolerance=solver["tol"],
            linesearch_max_iterations=solver.get("linesearch_max_iter"),
            lbfgs_memory=solver.get("lbfgs_memory"),
            penalty_normalized=solver.get("penalty_normalized"),
        )
    except (AssertionError, KeyError, TypeError) as error:
        raise ValueError("invalid optimizer runtime record") from error


def _validate_optimizer_runtime(config: ProbeOptimizerConfig) -> None:
    if config.backend == "sklearn-lbfgs":
        actual = sklearn_optimizer_config(
            max_iterations=config.max_iterations, tolerance=config.tolerance
        )
    else:
        assert config.linesearch_max_iterations is not None
        assert config.lbfgs_memory is not None
        actual = cuml_optimizer_config(
            max_iterations=config.max_iterations,
            tolerance=config.tolerance,
            linesearch_max_iterations=config.linesearch_max_iterations,
            lbfgs_memory=config.lbfgs_memory,
        )
    if actual.runtime_sha256 != config.runtime_sha256:
        raise ValueError("live optimizer runtime does not match the frozen training protocol")


@beartype
@dataclass(frozen=True, slots=True)
class ProbeTrainingConfig:
    """Preregistered layers, split, search grid, and neutral acceptance gates."""

    layer_indices: tuple[int, ...]
    minimum_neutral_accuracy: float
    minimum_neutral_per_role_accuracy: float
    minimum_neutral_document_accuracy: float
    minimum_neutral_per_role_document_accuracy: float
    optimizer: ProbeOptimizerConfig
    expected_document_count: int | None = None
    maximum_content_tokens_per_document: int | None = None
    protocol_sha256: str | None = None
    native_prompt_partitions_sha256: str | None = None
    neutral_target_source_sha256: str | None = None
    neutral_filler_source_sha256: str | None = None
    lambda_grid: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3)
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    seed: int = 0

    def __post_init__(self) -> None:
        if (
            not self.layer_indices
            or any(type(layer) is not int for layer in self.layer_indices)
            or min(self.layer_indices) < 0
            or len(set(self.layer_indices)) != len(self.layer_indices)
            or self.layer_indices != tuple(sorted(self.layer_indices))
        ):
            raise ValueError("layer_indices must contain increasing distinct non-negative layers")
        if self.expected_document_count is not None and (
            type(self.expected_document_count) is not int or self.expected_document_count < 3
        ):
            raise ValueError("expected_document_count must be an integer of at least three")
        if self.maximum_content_tokens_per_document is not None and (
            type(self.maximum_content_tokens_per_document) is not int
            or self.maximum_content_tokens_per_document <= 1
        ):
            raise ValueError("maximum_content_tokens_per_document must be an integer above one")
        if (self.neutral_target_source_sha256 is None) != (
            self.neutral_filler_source_sha256 is None
        ):
            raise ValueError("neutral target and filler source digests must be provided together")
        for name in (
            "protocol_sha256",
            "native_prompt_partitions_sha256",
            "neutral_target_source_sha256",
            "neutral_filler_source_sha256",
        ):
            value = cast(str | None, getattr(self, name))
            if value is not None:
                _sha256(value, name)
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
        if self.lambda_grid != tuple(sorted(self.lambda_grid)):
            raise ValueError("lambda_grid must be increasing")
        fractions = (
            self.train_fraction,
            self.validation_fraction,
            1 - self.train_fraction - self.validation_fraction,
        )
        if any(not math.isfinite(value) or value <= 0 for value in fractions):
            raise ValueError("training split fractions must be finite and positive")
        if type(self.seed) is not int:
            raise ValueError("training seed must be an integer")


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
        accuracy_roles = tuple(role for role, _ in self.per_role_accuracy)
        document_roles = tuple(role for role, _ in self.per_role_document_accuracy)
        if (
            not self.per_role_accuracy
            or accuracy_roles != document_roles
            or len(set(accuracy_roles)) != len(accuracy_roles)
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
class ProbeCandidateMetrics:
    """Development-only evidence for one layer and regularization candidate."""

    layer_index: int
    regularization_lambda: float
    metrics: ClassificationMetrics

    def __post_init__(self) -> None:
        if self.layer_index < 0:
            raise ValueError("candidate layer_index must be non-negative")
        if self.regularization_lambda <= 0 or not math.isfinite(self.regularization_lambda):
            raise ValueError("candidate regularization_lambda must be finite and positive")


class ProbeConvergenceError(RuntimeError):
    """A recognized optimizer convergence failure for one candidate fit."""


@beartype
@dataclass(frozen=True, slots=True)
class ProbeCandidateFailure:
    """Development-grid coordinate excluded after a recognized convergence failure."""

    layer_index: int
    regularization_lambda: float
    error_type: str
    message: str

    def __post_init__(self) -> None:
        if type(self.layer_index) is not int or self.layer_index < 0:
            raise ValueError("failed candidate layer_index must be a non-negative integer")
        if self.regularization_lambda <= 0 or not math.isfinite(self.regularization_lambda):
            raise ValueError("failed candidate regularization_lambda must be finite and positive")
        if self.error_type != ProbeConvergenceError.__name__:
            raise ValueError("failed candidate must record ProbeConvergenceError")
        _text(self.message, "failed candidate message")


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
        if self.test_metrics.document_count != self.test.scores.conversation_count:
            raise ValueError("native test metric count does not match its segment scores")
        if self.test.minimum_role_accuracy != self.test_metrics.minimum_role_accuracy:
            raise ValueError("native role gate does not match the test metrics")
        if self.test.document_macro_accuracy != self.test_metrics.document_accuracy:
            raise ValueError("native document gate does not match the test metrics")

    @property
    def threshold_reasoning_sensitivity(self) -> float | None:
        threshold = self.calibration.threshold
        if threshold is None:
            return None
        return float(np.mean(np.asarray(self.test.scores.reasoning_scores) >= threshold))

    @property
    def threshold_final_specificity(self) -> float | None:
        threshold = self.calibration.threshold
        if threshold is None:
            return None
        return float(np.mean(np.asarray(self.test.scores.final_scores) < threshold))

    @property
    def passed(self) -> bool:
        sensitivity = self.threshold_reasoning_sensitivity
        specificity = self.threshold_final_specificity
        return bool(
            self.calibration.usable
            and self.test.passed
            and sensitivity is not None
            and sensitivity >= MIN_THRESHOLD_REASONING_SENSITIVITY
            and specificity is not None
            and specificity >= MIN_THRESHOLD_FINAL_SPECIFICITY
        )


@beartype
@dataclass(frozen=True, slots=True)
class RoleProbe:
    """One development-selected layer's portable role classifier and evidence."""

    provenance: ActivationProvenance
    training: ProbeTrainingConfig
    split: DocumentSplit
    selected_layer_index: int
    regularization_lambda: float
    development_candidates: tuple[ProbeCandidateMetrics, ...]
    neutral_content_signatures: tuple[str, ...]
    coefficients: Array
    intercepts: Array
    validation_metrics: ClassificationMetrics
    neutral_test_metrics: ClassificationMetrics
    qualification: NativeProbeQualification | None = None
    failed_candidates: tuple[ProbeCandidateFailure, ...] = ()

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
        if any(layer not in self.provenance.layer_indices for layer in self.training.layer_indices):
            raise ValueError("candidate layer is absent from activation provenance")
        if self.selected_layer_index not in self.training.layer_indices:
            raise ValueError("selected layer is absent from the preregistered candidates")
        if self.regularization_lambda <= 0 or not math.isfinite(self.regularization_lambda):
            raise ValueError("regularization_lambda must be finite and positive")
        expected_candidates = tuple(
            (layer, regularization)
            for layer in self.training.layer_indices
            for regularization in self.training.lambda_grid
        )
        successful_coordinates = tuple(
            (candidate.layer_index, candidate.regularization_lambda)
            for candidate in self.development_candidates
        )
        failed_coordinates = tuple(
            (candidate.layer_index, candidate.regularization_lambda)
            for candidate in self.failed_candidates
        )
        observed_coordinates = successful_coordinates + failed_coordinates
        if len(observed_coordinates) != len(expected_candidates) or set(
            observed_coordinates
        ) != set(expected_candidates):
            raise ValueError("candidate evidence must cover the complete grid exactly once")
        if not successful_coordinates:
            raise ValueError("candidate evidence must contain at least one successful fit")
        successful_set = set(successful_coordinates)
        failed_set = set(failed_coordinates)
        if successful_coordinates != tuple(
            coordinate for coordinate in expected_candidates if coordinate in successful_set
        ):
            raise ValueError("successful candidate evidence must follow grid order")
        if failed_coordinates != tuple(
            coordinate for coordinate in expected_candidates if coordinate in failed_set
        ):
            raise ValueError("failed candidate evidence must follow grid order")
        expected_documents = len(self.split.train + self.split.validation + self.split.test)
        if (
            len(self.neutral_content_signatures) != expected_documents
            or len(set(self.neutral_content_signatures)) != expected_documents
        ):
            raise ValueError(
                "probe must retain one distinct neutral content signature per document"
            )
        for signature in self.neutral_content_signatures:
            _sha256(signature, "neutral_content_signature")
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
        for candidate in self.development_candidates:
            metrics = candidate.metrics
            if (
                tuple(role for role, _ in metrics.per_role_accuracy) != expected_roles
                or tuple(role for role, _ in metrics.per_role_document_accuracy) != expected_roles
                or len(metrics.confusion_matrix) != len(expected_roles)
            ):
                raise ValueError("candidate metrics do not match probe roles")
            if metrics.document_count != len(self.split.validation):
                raise ValueError("candidate metrics do not match the development split")
        selected = next(
            (
                candidate
                for candidate in self.development_candidates
                if candidate.layer_index == self.selected_layer_index
                and candidate.regularization_lambda == self.regularization_lambda
            ),
            None,
        )
        if selected is None or selected.metrics != self.validation_metrics:
            raise ValueError("selected candidate must match the stored validation metrics")
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


def _fit(x: Array, y: Array, regularization: float, config: ProbeTrainingConfig) -> Any:
    optimizer = config.optimizer
    if optimizer.backend == "sklearn-lbfgs":
        classifier = LogisticRegression(
            C=1 / regularization,
            penalty="l2",
            solver="lbfgs",
            fit_intercept=True,
            class_weight=None,
            max_iter=optimizer.max_iterations,
            tol=optimizer.tolerance,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            try:
                classifier.fit(
                    np.asarray(x, dtype=np.float64, order="C"),
                    np.asarray(y, dtype=np.int64),
                )
            except ConvergenceWarning as error:
                warning = " ".join(str(error).split())
                raise ProbeConvergenceError(
                    f"sklearn LBFGS failed to converge for lambda={regularization:g}: {warning}"
                ) from error
        return classifier

    cupy = importlib.import_module("cupy")
    cuml_linear_model = importlib.import_module("cuml.linear_model")
    assert optimizer.linesearch_max_iterations is not None
    assert optimizer.lbfgs_memory is not None
    host_x = np.asfortranarray(x, dtype=np.float32)
    if not host_x.flags.f_contiguous:
        raise RuntimeError("cuML probe matrix must be a single Fortran-order allocation")
    cupy.get_default_memory_pool().free_all_blocks()
    free_bytes, _ = cupy.cuda.runtime.memGetInfo()
    required_bytes = host_x.nbytes + 2 * 1024**3
    if free_bytes < required_bytes:
        raise MemoryError(
            "cuML probe fit lacks device headroom for one Fortran-order activation matrix"
        )
    device_x = cupy.empty(host_x.shape, dtype=np.float32, order="F")
    if (
        device_x.shape != host_x.shape
        or device_x.nbytes != host_x.nbytes
        or not device_x.flags.f_contiguous
        or device_x.dtype.name != "float32"
    ):
        raise RuntimeError("cuML probe matrix lost its FP32 Fortran-order contract")
    # CuPy 14.2's asarray path stages pageable input through an equally large pinned buffer.
    # This synchronous raw copy keeps host_x alive and avoids that second host allocation.
    device_x.data.copy_from_host(host_x.ctypes.data, host_x.nbytes)
    device_y = cupy.asarray(np.asarray(y, dtype=np.int32))
    if device_y.dtype.name != "int32":
        raise RuntimeError("cuML probe labels lost their int32 contract")
    classifier = cuml_linear_model.LogisticRegression(
        penalty="l2",
        tol=optimizer.tolerance,
        C=1 / regularization,
        fit_intercept=True,
        class_weight=None,
        max_iter=optimizer.max_iterations,
        linesearch_max_iter=optimizer.linesearch_max_iterations,
        l1_ratio=None,
        solver="qn",
        lbfgs_memory=optimizer.lbfgs_memory,
        penalty_normalized=True,
        verbose=False,
        output_type="cupy",
    )
    solver_output = io.StringIO()
    try:
        with redirect_stdout(solver_output):
            classifier.fit(device_x, device_y, sample_weight=None, convert_dtype=False)
    finally:
        output = solver_output.getvalue()
        if output:
            logger.info("cuML QN output for lambda=%g:\n%s", regularization, output.rstrip())
    failure_markers = (
        "l-bfgs line search failed (code ",
        "l-bfgs error fx=",
        "l-bfgs stopped, because the line search failed to advance",
        "l-bfgs: max iterations reached",
        "maximum iterations reached before solver is converged",
    )
    matched_failure = next(
        (
            line.strip()
            for line in output.splitlines()
            if any(marker in line.lower() for marker in failure_markers)
        ),
        None,
    )
    if matched_failure is not None:
        raise ProbeConvergenceError(
            f"cuML QN reported a fitting failure for lambda={regularization:g}: "
            f"{matched_failure}"
        )
    return classifier


def _host_array(value: Any) -> Array:
    getter = getattr(value, "get", None)
    return np.asarray(getter() if callable(getter) else value)


def _parameters(classifier: Any, class_count: int) -> tuple[Array, Array]:
    if not np.array_equal(_host_array(classifier.classes_), np.arange(class_count)):
        raise ValueError("classifier class order does not match role order")
    coefficients = np.asarray(_host_array(classifier.coef_), dtype=np.float64)
    intercepts = np.asarray(_host_array(classifier.intercept_), dtype=np.float64)
    if class_count == 2:
        coefficients = np.concatenate((-coefficients / 2, coefficients / 2))
        intercepts = np.concatenate((-intercepts / 2, intercepts / 2))
    return coefficients, intercepts


def _fit_parameters(
    x: Array,
    y: Array,
    regularization: float,
    config: ProbeTrainingConfig,
    class_count: int,
) -> tuple[Array, Array, list[int]]:
    classifier: Any | None = None
    try:
        classifier = _fit(x, y, regularization, config)
        coefficients, intercepts = _parameters(classifier, class_count)
        iterations = [int(value) for value in _host_array(classifier.n_iter_).reshape(-1)]
        if not iterations or any(
            value < 0 or value > config.optimizer.max_iterations for value in iterations
        ):
            raise RuntimeError("probe optimizer returned an invalid iteration count")
        if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(intercepts)):
            raise RuntimeError("probe optimizer returned non-finite parameters")
        return coefficients, intercepts, iterations
    finally:
        del classifier
        if config.optimizer.backend == "cuml-qn":
            cupy = importlib.import_module("cupy")
            cupy.get_default_memory_pool().free_all_blocks()


def _materialize_activation_matrix(
    dataset: ActivationDataset,
    rows: Array,
    layer_offset: int,
    *,
    dtype: Any,
    order: str,
) -> Array:
    selected = dataset.activations[rows, layer_offset]
    if order == "F":
        return np.asfortranarray(selected, dtype=dtype)
    if order == "C":
        return np.ascontiguousarray(selected, dtype=dtype)
    raise ValueError("activation matrix order must be C or F")


@beartype
def train_role_probe(dataset: ActivationDataset, config: ProbeTrainingConfig) -> RoleProbe:
    """Select layer and L2 strength on grouped neutral dev data and test once."""
    if dataset.provenance.dataset_kind != PAIRED_NEUTRAL:
        raise ValueError("training requires paired neutral-role activations")
    _validate_optimizer_runtime(config.optimizer)
    if any(layer not in dataset.provenance.layer_indices for layer in config.layer_indices):
        raise ValueError("configured layer is absent from activation export")
    documents = tuple(str(value) for value in np.unique(dataset.document_ids))
    if (
        config.expected_document_count is not None
        and len(documents) != config.expected_document_count
    ):
        raise ValueError("neutral document count does not match the training protocol")
    if config.maximum_content_tokens_per_document is not None:
        allowed_counts = {
            config.maximum_content_tokens_per_document - 1,
            config.maximum_content_tokens_per_document,
        }
        for document in documents:
            role_rows = _rows(dataset, document, dataset.provenance.roles[0])
            if len(role_rows) not in allowed_counts:
                raise ValueError("neutral content-token count does not match the training protocol")
    split = split_documents(
        documents,
        seed=config.seed,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
    )
    train_rows = _select_rows(dataset, split.train)
    validation_rows = _select_rows(dataset, split.validation)
    test_rows = _select_rows(dataset, split.test)
    train_labels = _labels(dataset, train_rows)
    validation_labels = _labels(dataset, validation_rows)
    fit_dtype = np.float32 if config.optimizer.backend == "cuml-qn" else np.float64
    fit_order = "F" if config.optimizer.backend == "cuml-qn" else "C"

    candidates: list[ProbeCandidateMetrics] = []
    failed_candidates: list[ProbeCandidateFailure] = []
    for layer_index in config.layer_indices:
        layer_offset = dataset.provenance.layer_indices.index(layer_index)
        train_matrix = _materialize_activation_matrix(
            dataset, train_rows, layer_offset, dtype=fit_dtype, order=fit_order
        )
        validation_matrix = _materialize_activation_matrix(
            dataset, validation_rows, layer_offset, dtype=np.float64, order="C"
        )
        for regularization in config.lambda_grid:
            started = time.monotonic()
            logger.info(
                "Fitting role probe candidate layer=%d lambda=%g", layer_index, regularization
            )
            try:
                coefficients, intercepts, iterations = _fit_parameters(
                    train_matrix,
                    train_labels,
                    regularization,
                    config,
                    len(dataset.provenance.roles),
                )
            except ProbeConvergenceError as error:
                logger.exception(
                    "Excluding role probe candidate layer=%d lambda=%g after convergence failure",
                    layer_index,
                    regularization,
                )
                failed_candidates.append(
                    ProbeCandidateFailure(
                        layer_index,
                        regularization,
                        type(error).__name__,
                        str(error),
                    )
                )
                continue
            probabilities = _predict_parameters(
                coefficients, intercepts, validation_matrix
            )
            if not np.all(np.isfinite(probabilities)):
                raise RuntimeError("probe candidate produced non-finite development probabilities")
            metrics = _metrics(
                probabilities,
                validation_labels,
                dataset.provenance.roles,
                dataset.document_ids[validation_rows],
            )
            candidates.append(ProbeCandidateMetrics(layer_index, regularization, metrics))
            logger.info(
                "Finished role probe candidate layer=%d lambda=%g iterations=%s "
                "dev_accuracy=%.6f dev_nll=%.6f elapsed_seconds=%.3f",
                layer_index,
                regularization,
                iterations,
                metrics.accuracy,
                metrics.negative_log_likelihood,
                time.monotonic() - started,
            )
        del train_matrix, validation_matrix
    if not candidates:
        raise RuntimeError("all role probe candidates failed to converge")
    selected = max(
        candidates,
        key=lambda candidate: (
            candidate.metrics.accuracy,
            -candidate.metrics.negative_log_likelihood,
            candidate.regularization_lambda,
            -candidate.layer_index,
        ),
    )
    layer_index = selected.layer_index
    regularization = selected.regularization_lambda
    validation_metrics = selected.metrics
    layer_offset = dataset.provenance.layer_indices.index(layer_index)
    fit_rows = np.concatenate((train_rows, validation_rows))
    started = time.monotonic()
    logger.info(
        "Refitting selected role probe layer=%d lambda=%g on train+development",
        layer_index,
        regularization,
    )
    fit_matrix = _materialize_activation_matrix(
        dataset, fit_rows, layer_offset, dtype=fit_dtype, order=fit_order
    )
    coefficients, intercepts, iterations = _fit_parameters(
        fit_matrix,
        _labels(dataset, fit_rows),
        regularization,
        config,
        len(dataset.provenance.roles),
    )
    del fit_matrix
    logger.info(
        "Finished selected role probe refit layer=%d lambda=%g iterations=%s elapsed_seconds=%.3f",
        layer_index,
        regularization,
        iterations,
        time.monotonic() - started,
    )
    test_probabilities = _predict_parameters(
        coefficients, intercepts, dataset.activations[test_rows, layer_offset]
    )
    if not np.all(np.isfinite(test_probabilities)):
        raise RuntimeError("probe produced non-finite held-out probabilities")
    return RoleProbe(
        provenance=dataset.provenance,
        training=config,
        split=split,
        selected_layer_index=layer_index,
        regularization_lambda=regularization,
        development_candidates=tuple(candidates),
        neutral_content_signatures=_neutral_content_signatures(dataset),
        coefficients=coefficients,
        intercepts=intercepts,
        validation_metrics=validation_metrics,
        neutral_test_metrics=_metrics(
            test_probabilities,
            _labels(dataset, test_rows),
            dataset.provenance.roles,
            dataset.document_ids[test_rows],
        ),
        failed_candidates=tuple(failed_candidates),
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
    layer_offset = dataset.provenance.layer_indices.index(probe.selected_layer_index)
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


def _content_signature(token_ids: Array) -> str:
    return hashlib.sha256(_json([int(value) for value in token_ids])).hexdigest()


def _neutral_content_signatures(dataset: ActivationDataset) -> tuple[str, ...]:
    role = dataset.provenance.roles[0]
    return tuple(
        sorted(
            _content_signature(dataset.content_token_id[_rows(dataset, str(document), role)])
            for document in np.unique(dataset.document_ids)
        )
    )


def _conversation_content_signatures(dataset: ActivationDataset) -> set[str]:
    signatures: set[str] = set()
    for document in np.unique(dataset.document_ids):
        for role in ("reasoning", "assistant"):
            rows = _rows(dataset, str(document), role)
            signature = _content_signature(dataset.content_token_id[rows])
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
    neutral_content = set(probe.neutral_content_signatures)
    if neutral_content & (calibration_content | test_content):
        raise ValueError("native and neutral measured content must be disjoint")
    if calibration_content & test_content:
        raise ValueError("calibration and test conversation segments must be content-disjoint")

    calibration_scores = _paired_segment_scores(probe, calibration_conversations, CALIBRATION_SPLIT)
    test_scores = _paired_segment_scores(probe, test_conversations, TEST_SPLIT)
    if calibration_scores.conversation_count != EXPECTED_CONVERSATIONS:
        raise ValueError(f"native calibration requires {EXPECTED_CONVERSATIONS} conversations")
    layer_offset = test_conversations.provenance.layer_indices.index(probe.selected_layer_index)
    rows = np.arange(len(test_conversations.roles))
    test_metrics = _metrics(
        _predict(probe, test_conversations.activations[:, layer_offset]),
        _labels(test_conversations, rows),
        probe.provenance.roles,
        test_conversations.document_ids,
        ("reasoning", "assistant"),
    )
    calibration = calibrate_reasoning_threshold(calibration_scores)
    qualification = NativeProbeQualification(
        calibration_dataset_fingerprint=activation_dataset_fingerprint(calibration_conversations),
        test_dataset_fingerprint=activation_dataset_fingerprint(test_conversations),
        prompt_partition_name=prompt_partition_name,
        prompt_partition_sha256=prompt_partition_sha256,
        calibration_conversation_count=calibration_scores.conversation_count,
        calibration_scores=calibration_scores,
        calibration=calibration,
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
    if layer_index != probe.selected_layer_index:
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
    training["layer_indices"] = tuple(training["layer_indices"])
    training["lambda_grid"] = tuple(training["lambda_grid"])
    development_candidates = tuple(
        ProbeCandidateMetrics(
            layer_index=candidate["layer_index"],
            regularization_lambda=candidate["regularization_lambda"],
            metrics=_metrics_from(candidate["metrics"]),
        )
        for candidate in metadata["development_candidates"]
    )
    failed_candidates = tuple(
        ProbeCandidateFailure(**candidate)
        for candidate in metadata.get("failed_candidates", ())
    )
    neutral_content_signatures = tuple(metadata["neutral_content_signatures"])
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
    optimizer = training.get("optimizer")
    if not isinstance(optimizer, dict):
        raise ValueError("probe artifact lacks optimizer provenance")
    training["optimizer"] = ProbeOptimizerConfig(**optimizer)
    return RoleProbe(
        provenance=ActivationProvenance(**provenance),
        training=ProbeTrainingConfig(**training),
        split=DocumentSplit(**split),
        selected_layer_index=metadata["selected_layer_index"],
        regularization_lambda=metadata["regularization_lambda"],
        development_candidates=development_candidates,
        neutral_content_signatures=neutral_content_signatures,
        coefficients=coefficients,
        intercepts=intercepts,
        validation_metrics=_metrics_from(metadata["validation_metrics"]),
        neutral_test_metrics=_metrics_from(metadata["neutral_test_metrics"]),
        failed_candidates=failed_candidates,
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
