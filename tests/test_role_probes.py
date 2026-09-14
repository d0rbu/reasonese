"""Activation-role probe training, qualification, and artifact contracts."""

from __future__ import annotations

import ctypes
import json
import os
from dataclasses import replace
from math import inf, nan
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

import reasonese.role_probes as role_probes
from reasonese.role_probes import (
    ActivationDataset,
    ActivationProvenance,
    ClassificationMetrics,
    ProbeTrainingConfig,
    RoleProbe,
    SavedProbeAdoption,
    _fit_parameters,
    _parameters,
    _predict_parameters,
    activation_dataset_fingerprint,
    adopt_saved_probe_parameters,
    load_role_probe,
    optimizer_config_from_runtime,
    project_role,
    qualify_role_probe,
    save_role_probe,
    split_documents,
    train_role_probe,
)

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_ROLES = ("reasoning", "assistant")


def _provenance(
    *,
    kind: str = "paired-neutral-role-wrappers",
    roles: tuple[str, ...] = _ROLES,
    layers: tuple[int, ...] = (3, 7),
    model_revision: str = "model-commit-a",
    filler_documents: int = 20,
) -> ActivationProvenance:
    return ActivationProvenance(
        dataset_kind=kind,
        model_id="nvidia/test-model",
        model_revision=model_revision,
        weights_sha256=_SHA_A,
        weights_hash_kind="hf_lfs_manifest_sha256",
        tokenizer_id="nvidia/test-model",
        tokenizer_revision="tokenizer-commit-a",
        chat_template_sha256=_SHA_B,
        native_template_adapter="test-native-template",
        activation_site="normalized_pre_mixer",
        runtime_sha256="f" * 64,
        model_dtype="bfloat16",
        activation_dtype="float32",
        layer_indices=layers,
        hidden_size=5,
        roles=roles,
        source_name="synthetic neutral documents",
        source_sha256="c" * 64,
        extraction_protocol="role-probe-extraction-v1",
        content_mask="content-tokens-only",
        masked_control_tokens=48,
        masked_filler_tokens=24,
        filler_pool_kind=(
            "dedicated-disjoint-documents" if kind == "paired-neutral-role-wrappers" else "none"
        ),
        filler_source_sha256="d" * 64 if kind == "paired-neutral-role-wrappers" else None,
        filler_documents=filler_documents if kind == "paired-neutral-role-wrappers" else 0,
    )


def _dataset(
    *,
    kind: str = "paired-neutral-role-wrappers",
    roles: tuple[str, ...] = _ROLES,
    document_count: int = 20,
    layers: tuple[int, ...] = (3, 7),
    model_revision: str = "model-commit-a",
    document_prefix: str | None = None,
    content_token_offset: int = 0,
) -> ActivationDataset:
    activations: list[np.ndarray] = []
    document_ids: list[str] = []
    labels: list[str] = []
    content_positions: list[int] = []
    content_token_ids: list[int] = []
    sequence_positions: list[int] = []
    filler_document_ids: list[str] = []
    document_prefix = document_prefix or (
        "document" if kind == "paired-neutral-role-wrappers" else "conversation"
    )
    for document_index in range(document_count):
        for role_index, role in enumerate(roles):
            for token_index in range(4):
                # Both layers contain document/content variation, but the role
                # direction is deliberately dominant and transfers to fresh docs.
                common = np.asarray(
                    [
                        (document_index % 3) * 0.02,
                        token_index * 0.01,
                        (document_index % 2) * 0.01,
                        0.0,
                        0.0,
                    ],
                    dtype=np.float32,
                )
                per_layer = []
                for layer_offset, _ in enumerate(layers):
                    vector = common.copy()
                    vector[role_index] += 3.0 + layer_offset
                    vector[-1] = layer_offset * 0.1
                    per_layer.append(vector)
                activations.append(np.stack(per_layer))
                document_ids.append(f"{document_prefix}-{document_index:03d}")
                labels.append(role)
                content_positions.append(token_index)
                content_token_ids.append(
                    content_token_offset + 100 + document_index * 4 + token_index
                )
                sequence_positions.append(32 + token_index)
                filler_document_ids.append(
                    f"filler-{document_index:03d}" if kind == "paired-neutral-role-wrappers" else ""
                )
    return ActivationDataset(
        provenance=_provenance(
            kind=kind,
            roles=roles,
            layers=layers,
            model_revision=model_revision,
            filler_documents=document_count,
        ),
        activations=np.stack(activations),
        document_ids=np.asarray(document_ids, dtype=np.str_),
        roles=np.asarray(labels, dtype=np.str_),
        content_token_index=np.asarray(content_positions, dtype=np.int64),
        content_token_id=np.asarray(content_token_ids, dtype=np.int64),
        sequence_token_index=np.asarray(sequence_positions, dtype=np.int64),
        filler_document_ids=np.asarray(filler_document_ids, dtype=np.str_),
    )


def _training_config(selected_layer: int = 7) -> ProbeTrainingConfig:
    return ProbeTrainingConfig(
        layer_indices=(selected_layer,),
        minimum_neutral_accuracy=0.95,
        minimum_neutral_per_role_accuracy=0.95,
        minimum_neutral_document_accuracy=0.95,
        minimum_neutral_per_role_document_accuracy=0.95,
        lambda_grid=(0.01, 0.1),
        train_fraction=0.7,
        validation_fraction=0.15,
        seed=19,
        optimizer=role_probes.sklearn_optimizer_config(max_iterations=500),
    )


def test_neutral_source_digests_are_bound_together() -> None:
    config = _training_config()
    with pytest.raises(ValueError, match="provided together"):
        replace(config, neutral_target_source_sha256="a" * 64)
    with pytest.raises(ValueError, match="provided together"):
        replace(config, neutral_filler_source_sha256="b" * 64)
    assert (
        replace(
            config,
            neutral_target_source_sha256="a" * 64,
            neutral_filler_source_sha256="b" * 64,
        ).neutral_target_source_sha256
        == "a" * 64
    )


def _multinomial_objective(
    coefficients: np.ndarray,
    intercepts: np.ndarray,
    activations: np.ndarray,
    labels: np.ndarray,
    regularization: float,
) -> float:
    probabilities = _predict_parameters(coefficients, intercepts, activations)
    likelihoods = np.clip(probabilities[np.arange(len(labels)), labels], 1e-15, 1.0)
    penalty = regularization * float(np.square(coefficients).sum()) / (2 * len(labels))
    return float(-np.log(likelihoods).mean() + penalty)


def _cuml_optimizer_config(*, tolerance: float = 1e-4) -> role_probes.ProbeOptimizerConfig:
    record = {
        "backend": "cuml-qn",
        "format_version": 1,
        "implementation_sha256": {},
        "matrix": {
            "activation_dtype": "float32",
            "array_order": "F",
            "convert_dtype": False,
            "labels_dtype": "int32",
        },
        "packages": {},
        "python_version": "test",
        "regularization_mapping": "C=1/lambda",
        "solver": {
            "class_weight": None,
            "fit_intercept": True,
            "lbfgs_memory": 5,
            "linesearch_max_iter": 100,
            "l1_ratio": None,
            "max_iter": 5_000,
            "output_type": "cupy",
            "penalty": "l2",
            "penalty_normalized": True,
            "solver": "qn",
            "tol": tolerance,
            "verbose": False,
        },
    }
    return optimizer_config_from_runtime(record)


def _qualify(
    bundle: RoleProbe,
    *,
    calibration: ActivationDataset | None = None,
    test: ActivationDataset | None = None,
) -> RoleProbe:
    return qualify_role_probe(
        bundle,
        calibration
        or _dataset(
            kind="untouched-native-conversations",
            document_count=12,
            document_prefix="calibration",
            content_token_offset=5_000,
        ),
        test
        or _dataset(
            kind="untouched-native-conversations",
            document_count=12,
            document_prefix="test",
            content_token_offset=10_000,
        ),
        prompt_partition_name="native-prompt-partitions.json",
        prompt_partition_sha256="e" * 64,
    )


def test_neutral_data_requires_exact_paired_content_and_positions() -> None:
    dataset = _dataset()
    assert dataset.activations.shape == (160, 2, 5)

    bad_token_ids = dataset.content_token_id.copy()
    bad_token_ids[4] += 1
    with pytest.raises(ValueError, match="identical tokens"):
        replace(dataset, content_token_id=bad_token_ids)

    bad_sequence_positions = dataset.sequence_token_index.copy()
    bad_sequence_positions[4] += 1
    with pytest.raises(ValueError, match="identical positions"):
        replace(dataset, sequence_token_index=bad_sequence_positions)

    keep = dataset.roles != "assistant"
    with pytest.raises(ValueError, match="roles do not match provenance"):
        replace(
            dataset,
            activations=dataset.activations[keep],
            document_ids=dataset.document_ids[keep],
            roles=dataset.roles[keep],
            content_token_index=dataset.content_token_index[keep],
            content_token_id=dataset.content_token_id[keep],
            sequence_token_index=dataset.sequence_token_index[keep],
            filler_document_ids=dataset.filler_document_ids[keep],
        )


def test_activation_shape_dtype_and_provenance_are_fail_closed() -> None:
    dataset = _dataset()
    with pytest.raises(ValueError, match="hidden-size"):
        replace(dataset, activations=dataset.activations[:, :, :-1])
    with pytest.raises(TypeError, match="floating dtype"):
        replace(dataset, activations=dataset.activations.astype(np.int64))
    invalid = dataset.activations.copy()
    invalid[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        replace(dataset, activations=invalid)
    with pytest.raises(ValueError, match="SHA-256"):
        replace(dataset.provenance, weights_sha256="not-a-digest")
    with pytest.raises(ValueError, match="content_mask"):
        replace(dataset.provenance, content_mask="includes-tags")
    overlapping_filler = dataset.document_ids.copy()
    with pytest.raises(ValueError, match="disjoint"):
        replace(dataset, filler_document_ids=overlapping_filler)
    with pytest.raises(TypeError, match="integer dtype"):
        replace(dataset, content_token_index=dataset.content_token_index.astype(np.float64))
    with pytest.raises(TypeError, match="string dtype"):
        replace(dataset, roles=dataset.roles.astype(object))


def test_document_groups_cannot_hide_duplicate_content_under_distinct_ids() -> None:
    dataset = _dataset()
    duplicated = dataset.content_token_id.copy()
    first = dataset.document_ids == "document-000"
    second = dataset.document_ids == "document-001"
    duplicated[second] = duplicated[first]
    with pytest.raises(ValueError, match="contain identical tokens"):
        replace(dataset, content_token_id=duplicated)


def test_document_split_is_stable_complete_and_group_disjoint() -> None:
    document_ids = tuple(f"doc-{index}" for index in range(20))
    first = split_documents(document_ids, seed=7, train_fraction=0.7, validation_fraction=0.15)
    second = split_documents(
        tuple(reversed(document_ids)), seed=7, train_fraction=0.7, validation_fraction=0.15
    )
    assert first == second
    assert (len(first.train), len(first.validation), len(first.test)) == (14, 3, 3)
    assert set(first.train) | set(first.validation) | set(first.test) == set(document_ids)
    assert not (set(first.train) & set(first.validation))
    assert len(first.fingerprint) == 64
    with pytest.raises(ValueError, match="unique"):
        split_documents(("a", "a", "b"), seed=0, train_fraction=0.6, validation_fraction=0.2)


def test_training_selects_only_on_grouped_neutral_development_data() -> None:
    dataset = _dataset()
    bundle = train_role_probe(dataset, _training_config())
    assert bundle.selected_layer_index == 7
    assert bundle.training.optimizer.tolerance == 1e-4
    assert bundle.regularization_lambda in {0.01, 0.1}
    assert bundle.neutral_valid
    assert not bundle.qa_eligible
    assert bundle.validation_metrics.accuracy == 1.0
    assert bundle.neutral_test_metrics.accuracy == 1.0
    assert bundle.neutral_test_metrics.document_accuracy == 1.0
    assert bundle.neutral_test_metrics.document_count == len(bundle.split.test)
    for split_ids in (bundle.split.train, bundle.split.validation, bundle.split.test):
        rows = np.isin(dataset.document_ids, split_ids)
        assert set(dataset.roles[rows]) == set(_ROLES)


def test_saved_parameters_are_adopted_without_refitting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = _dataset()
    reference = train_role_probe(dataset, _training_config())
    source_probe = tmp_path / "reference.npz"
    save_role_probe(reference, source_probe)
    adoption = SavedProbeAdoption(
        method=role_probes.SAVED_STANDARDIZED_PARAMETERS,
        fit_document_count=len(reference.split.train),
        source_probe_sha256=role_probes._path_sha256(source_probe),
        source_report_sha256="1" * 64,
        source_parameters_sha256="2" * 64,
        source_activation_manifest_sha256="3" * 64,
        training_mean_sha256="4" * 64,
        training_scale_sha256="5" * 64,
    )
    config = replace(
        reference.training,
        qualification_policy=role_probes.SEGMENT_MEAN_REASONING_PROBABILITY,
    )
    monkeypatch.setattr(
        role_probes,
        "_fit_parameters",
        lambda *_args, **_kwargs: pytest.fail("adoption must not fit"),
    )
    adopted = adopt_saved_probe_parameters(
        reference,
        dataset,
        config,
        reference.development_candidates,
        reference.coefficients,
        reference.intercepts,
        adoption,
    )
    assert adopted.saved_adoption == adoption
    assert adopted.selected_layer_index == reference.selected_layer_index
    assert adopted.regularization_lambda == reference.regularization_lambda
    assert adopted.neutral_test_metrics == reference.neutral_test_metrics

    output = tmp_path / "adopted.npz"
    save_role_probe(adopted, output)
    assert load_role_probe(output).saved_adoption == adoption


@pytest.mark.parametrize(
    "changes",
    [
        {"train_fraction": 0.6},
        {"validation_fraction": 0.2},
        {"seed": 1},
        {"expected_document_count": 21},
        {"maximum_content_tokens_per_document": 127},
        {
            "neutral_target_source_sha256": "a" * 64,
            "neutral_filler_source_sha256": "b" * 64,
        },
        {"optimizer": role_probes.sklearn_optimizer_config(max_iterations=501)},
    ],
)
def test_saved_parameter_adoption_rejects_reference_training_drift_before_test_scoring(
    monkeypatch: pytest.MonkeyPatch, changes: dict[str, object]
) -> None:
    dataset = _dataset()
    reference = train_role_probe(dataset, _training_config())
    config = replace(
        reference.training,
        qualification_policy=role_probes.SEGMENT_MEAN_REASONING_PROBABILITY,
        **changes,
    )
    monkeypatch.setattr(
        role_probes,
        "_predict_parameters",
        lambda *_args, **_kwargs: pytest.fail("TEST must not be scored after protocol drift"),
    )
    with pytest.raises(ValueError, match="immutable reference training settings"):
        adopt_saved_probe_parameters(
            reference,
            dataset,
            config,
            reference.development_candidates,
            reference.coefficients,
            reference.intercepts,
            SavedProbeAdoption(
                method=role_probes.SAVED_STANDARDIZED_PARAMETERS,
                fit_document_count=len(reference.split.train),
                source_probe_sha256="0" * 64,
                source_report_sha256="1" * 64,
                source_parameters_sha256="2" * 64,
                source_activation_manifest_sha256="3" * 64,
                training_mean_sha256="4" * 64,
                training_scale_sha256="5" * 64,
            ),
        )


def test_segment_policy_keeps_token_metrics_diagnostic() -> None:
    dataset = _dataset()
    legacy = _qualify(train_role_probe(dataset, _training_config()))
    assert legacy.qualification is not None
    metrics = legacy.qualification.test_metrics
    poor_roles = tuple((role, 0.25) for role, _ in metrics.per_role_accuracy)
    poor_metrics = replace(
        metrics,
        accuracy=0.25,
        document_accuracy=0.25,
        per_role_accuracy=poor_roles,
        per_role_document_accuracy=poor_roles,
    )
    poor_test = replace(
        legacy.qualification.test,
        minimum_role_accuracy=0.25,
        document_macro_accuracy=0.25,
    )
    qualification = replace(
        legacy.qualification,
        test_metrics=poor_metrics,
        test=poor_test,
    )
    assert not qualification.passed
    assert qualification.segment_passed
    assert not replace(legacy, qualification=qualification).qa_eligible
    segment = replace(
        legacy,
        training=replace(
            legacy.training,
            qualification_policy=role_probes.SEGMENT_MEAN_REASONING_PROBABILITY,
        ),
        qualification=qualification,
    )
    assert segment.qa_eligible


def test_training_jointly_selects_layer_and_lambda_without_reading_test_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset(layers=(3, 7))
    config = replace(_training_config(), layer_indices=(3, 7))
    metric_documents: list[tuple[str, ...]] = []
    materializations: list[tuple[int, int, str, str]] = []
    fit_matrix_ids: list[int] = []
    original_metrics = role_probes._metrics
    original_materialize = role_probes._materialize_activation_matrix
    original_fit_parameters = role_probes._fit_parameters

    def recording_metrics(
        probabilities: np.ndarray,
        labels: np.ndarray,
        roles: tuple[str, ...],
        documents: np.ndarray,
        evaluated_roles: tuple[str, ...] | None = None,
    ) -> ClassificationMetrics:
        metric_documents.append(tuple(sorted(set(documents.tolist()))))
        return original_metrics(probabilities, labels, roles, documents, evaluated_roles)

    monkeypatch.setattr(role_probes, "_metrics", recording_metrics)

    def recording_materialize(
        source: ActivationDataset,
        rows: np.ndarray,
        layer_offset: int,
        *,
        dtype: object,
        order: str,
    ) -> np.ndarray:
        result = original_materialize(source, rows, layer_offset, dtype=dtype, order=order)
        np.testing.assert_array_equal(result, source.activations[rows, layer_offset])
        materializations.append((layer_offset, len(rows), result.dtype.name, order))
        return result

    def recording_fit(
        x: np.ndarray,
        y: np.ndarray,
        regularization: float,
        training: ProbeTrainingConfig,
        class_count: int,
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        fit_matrix_ids.append(id(x))
        return original_fit_parameters(x, y, regularization, training, class_count)

    monkeypatch.setattr(role_probes, "_materialize_activation_matrix", recording_materialize)
    monkeypatch.setattr(role_probes, "_fit_parameters", recording_fit)
    bundle = train_role_probe(dataset, config)

    assert bundle.selected_layer_index == 7
    assert len(bundle.development_candidates) == 4
    assert bundle.failed_candidates == ()
    assert {
        (candidate.layer_index, candidate.regularization_lambda)
        for candidate in bundle.development_candidates
    } == {(3, 0.01), (3, 0.1), (7, 0.01), (7, 0.1)}
    assert all(documents == metric_documents[0] for documents in metric_documents[:-1])
    assert set(metric_documents[0]) == set(bundle.split.validation)
    assert set(metric_documents[-1]) == set(bundle.split.test)
    assert len(metric_documents) == len(bundle.development_candidates) + 1
    assert materializations == [
        (0, len(bundle.split.train) * len(_ROLES) * 4, "float64", "C"),
        (0, len(bundle.split.validation) * len(_ROLES) * 4, "float64", "C"),
        (1, len(bundle.split.train) * len(_ROLES) * 4, "float64", "C"),
        (1, len(bundle.split.validation) * len(_ROLES) * 4, "float64", "C"),
        (
            1,
            (len(bundle.split.train) + len(bundle.split.validation)) * len(_ROLES) * 4,
            "float64",
            "C",
        ),
    ]
    assert fit_matrix_ids[0] == fit_matrix_ids[1]
    assert fit_matrix_ids[2] == fit_matrix_ids[3]


def test_candidate_grid_records_only_explicit_convergence_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = _dataset(layers=(3, 7))
    config = replace(_training_config(), layer_indices=(3, 7))
    original_fit = role_probes._fit_parameters
    calls = 0

    def controlled_fit(
        x: np.ndarray,
        y: np.ndarray,
        regularization: float,
        training: ProbeTrainingConfig,
        class_count: int,
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        nonlocal calls
        calls += 1
        if calls in {1, 4}:
            raise role_probes.ProbeConvergenceError(f"recognized failure {calls}")
        return original_fit(x, y, regularization, training, class_count)

    monkeypatch.setattr(role_probes, "_fit_parameters", controlled_fit)
    bundle = train_role_probe(dataset, config)
    assert calls == 5
    assert tuple(
        (candidate.layer_index, candidate.regularization_lambda)
        for candidate in bundle.development_candidates
    ) == ((3, 0.1), (7, 0.01))
    assert tuple(
        (failure.layer_index, failure.regularization_lambda, failure.error_type)
        for failure in bundle.failed_candidates
    ) == (
        (3, 0.01, "ProbeConvergenceError"),
        (7, 0.1, "ProbeConvergenceError"),
    )

    path = tmp_path / "probe-with-failed-candidates.npz"
    save_role_probe(bundle, path)
    loaded = load_role_probe(path)
    assert loaded.failed_candidates == bundle.failed_candidates
    qualified = _qualify(loaded)
    assert qualified.qa_eligible
    assert qualified.failed_candidates == bundle.failed_candidates

    with pytest.raises(ValueError, match="complete grid exactly once"):
        replace(bundle, failed_candidates=bundle.failed_candidates[:1])
    with pytest.raises(ValueError, match="complete grid exactly once"):
        replace(
            bundle,
            failed_candidates=(bundle.failed_candidates[0],) + bundle.failed_candidates,
        )
    with pytest.raises(ValueError, match="failed candidate evidence must follow grid order"):
        replace(bundle, failed_candidates=tuple(reversed(bundle.failed_candidates)))


def test_all_candidate_convergence_failures_are_fatal_and_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    calls = 0

    def fail(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise role_probes.ProbeConvergenceError("recognized convergence failure")

    monkeypatch.setattr(role_probes, "_fit_parameters", fail)
    with caplog.at_level("ERROR"), pytest.raises(RuntimeError, match="all role probe candidates"):
        train_role_probe(_dataset(layers=(3, 7)), replace(_training_config(), layer_indices=(3, 7)))
    assert calls == 4
    assert (
        sum(message.startswith("Excluding role probe candidate") for message in caplog.messages)
        == 4
    )


def test_selected_refit_convergence_failure_remains_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_fit = role_probes._fit_parameters
    calls = 0

    def fail_refit(
        x: np.ndarray,
        y: np.ndarray,
        regularization: float,
        training: ProbeTrainingConfig,
        class_count: int,
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        nonlocal calls
        calls += 1
        if calls == 5:
            raise role_probes.ProbeConvergenceError("selected refit failed")
        return original_fit(x, y, regularization, training, class_count)

    monkeypatch.setattr(role_probes, "_fit_parameters", fail_refit)
    with pytest.raises(role_probes.ProbeConvergenceError, match="selected refit failed"):
        train_role_probe(_dataset(layers=(3, 7)), replace(_training_config(), layer_indices=(3, 7)))
    assert calls == 5


def test_unexpected_candidate_runtime_error_aborts_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fail(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("unexpected CUDA failure")

    monkeypatch.setattr(role_probes, "_fit_parameters", fail)
    with pytest.raises(RuntimeError, match="unexpected CUDA failure"):
        train_role_probe(_dataset(layers=(3, 7)), replace(_training_config(), layer_indices=(3, 7)))
    assert calls == 1


def test_training_enforces_preregistered_document_and_token_counts() -> None:
    dataset = _dataset()
    with pytest.raises(ValueError, match="document count"):
        train_role_probe(dataset, replace(_training_config(), expected_document_count=21))
    boundary_masked = train_role_probe(
        dataset,
        replace(_training_config(), maximum_content_tokens_per_document=5),
    )
    assert boundary_masked.neutral_valid
    with pytest.raises(ValueError, match="content-token count"):
        train_role_probe(
            dataset,
            replace(_training_config(), maximum_content_tokens_per_document=6),
        )


@pytest.mark.parametrize("tolerance", [nan, inf, -inf])
def test_training_rejects_nonfinite_optimizer_tolerance(tolerance: float) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        replace(_training_config().optimizer, tolerance=tolerance)


def test_training_config_requires_canonical_candidates_and_valid_split() -> None:
    with pytest.raises(ValueError, match="increasing distinct"):
        replace(_training_config(), layer_indices=(7, 3))
    with pytest.raises(ValueError, match="lambda_grid must be increasing"):
        replace(_training_config(), lambda_grid=(0.1, 0.01))
    with pytest.raises(ValueError, match="split fractions"):
        replace(_training_config(), train_fraction=0.9, validation_fraction=0.1)
    with pytest.raises(ValueError, match="non-negative layers"):
        replace(_training_config(), layer_indices=(True,))
    with pytest.raises(ValueError, match="seed must be an integer"):
        replace(_training_config(), seed=True)
    with pytest.raises(ValueError, match="integer of at least three"):
        replace(_training_config(), expected_document_count=True)


def test_multinomial_training_preserves_explicit_role_order() -> None:
    roles = ("reasoning", "assistant", "tool")
    bundle = train_role_probe(
        _dataset(roles=roles, layers=(5,), document_count=18),
        _training_config(selected_layer=5),
    )
    assert bundle.provenance.roles == roles
    assert bundle.coefficients.shape == (3, 5)
    assert bundle.neutral_test_metrics.per_role_accuracy == (
        ("reasoning", 1.0),
        ("assistant", 1.0),
        ("tool", 1.0),
    )
    full_projection = project_role(
        bundle,
        np.asarray([[3.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        provenance=bundle.provenance,
        layer_index=bundle.selected_layer_index,
        require_qa_eligible=False,
    )
    assert full_projection.mean_reasoning_probability > 0.99
    assert full_projection.mean_probabilities[2] < 0.01

    def measured(prefix: str) -> ActivationDataset:
        conversations = _dataset(
            kind="untouched-native-conversations",
            roles=roles,
            layers=(5,),
            document_count=12,
            document_prefix=prefix,
            content_token_offset=5_000 if prefix == "calibration" else 10_000,
        )
        keep = np.isin(conversations.roles, ("reasoning", "assistant"))
        return replace(
            conversations,
            activations=conversations.activations[keep],
            document_ids=conversations.document_ids[keep],
            roles=conversations.roles[keep],
            content_token_index=conversations.content_token_index[keep],
            content_token_id=conversations.content_token_id[keep],
            sequence_token_index=conversations.sequence_token_index[keep],
            filler_document_ids=conversations.filler_document_ids[keep],
        )

    qualified = _qualify(
        bundle,
        calibration=measured("calibration"),
        test=measured("test"),
    )
    assert qualified.qualification is not None
    assert qualified.qualification.test_metrics.per_role_accuracy == (
        ("reasoning", 1.0),
        ("assistant", 1.0),
    )


@pytest.mark.parametrize("class_count", [2, 3])
def test_portable_softmax_exactly_matches_sklearn_probabilities(class_count: int) -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(90, 4))
    y = np.arange(90) % class_count
    x[:, 0] += y * 2
    classifier = LogisticRegression(C=0.7, solver="lbfgs", max_iter=500).fit(x, y)
    coefficients, intercepts = _parameters(classifier, class_count)
    expected = classifier.predict_proba(x)
    actual = _predict_parameters(coefficients, intercepts, x)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-15)


def test_sklearn_regularization_mapping_preserves_duplicated_mean_loss_objective() -> None:
    rng = np.random.default_rng(44)
    x = rng.normal(size=(90, 4))
    y = np.arange(90) % 3
    config = _training_config()
    first = _fit_parameters(x, y, 0.2, config, 3)
    duplicated = _fit_parameters(np.tile(x, (2, 1)), np.tile(y, 2), 0.4, config, 3)
    first_probabilities = _predict_parameters(first[0], first[1], x)
    duplicated_probabilities = _predict_parameters(duplicated[0], duplicated[1], x)
    np.testing.assert_allclose(first_probabilities, duplicated_probabilities, rtol=2e-6, atol=2e-7)
    first_objective = _multinomial_objective(first[0], first[1], x, y, 0.2)
    duplicated_objective = _multinomial_objective(
        duplicated[0], duplicated[1], np.tile(x, (2, 1)), np.tile(y, 2), 0.4
    )
    assert first_objective == pytest.approx(duplicated_objective, rel=2e-8)


def test_sklearn_probe_does_not_regularize_class_imbalance_intercept() -> None:
    x = np.zeros((100, 2))
    y = np.asarray([0] * 80 + [1] * 20)
    coefficients, intercepts, _ = _fit_parameters(x, y, 1.0, _training_config(), 2)
    probabilities = _predict_parameters(coefficients, intercepts, x[:1])[0]
    np.testing.assert_allclose(probabilities, (0.8, 0.2), rtol=2e-4, atol=5e-5)


def test_sklearn_convergence_warning_is_an_explicit_candidate_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Classifier:
        def fit(self, *args: object, **kwargs: object) -> None:
            raise role_probes.ConvergenceWarning("solver did not converge\nwithin tolerance")

    monkeypatch.setattr(role_probes, "LogisticRegression", lambda **kwargs: Classifier())
    with pytest.raises(
        role_probes.ProbeConvergenceError,
        match="solver did not converge within tolerance",
    ):
        _fit_parameters(np.ones((4, 2)), np.arange(4) % 2, 0.1, _training_config(), 2)


@pytest.mark.parametrize(
    (
        "solver_message",
        "expected_iterations",
        "free_bytes",
        "nonfinite",
        "invalid_device_dtype",
        "error",
    ),
    (
        ("", 0, 8 * 1024**3, False, False, None),
        ("", 5_000, 8 * 1024**3, False, False, None),
        (
            "L-BFGS: max iterations reached",
            5_000,
            8 * 1024**3,
            False,
            False,
            "reported a fitting failure.*L-BFGS: max iterations reached",
        ),
        ("", 5_001, 8 * 1024**3, False, False, "invalid iteration count"),
        ("", 10, 8 * 1024**3, True, False, "non-finite parameters"),
        ("", 10, 1, False, False, "lacks device headroom"),
        ("", 10, 8 * 1024**3, False, True, "lost its FP32 Fortran-order contract"),
    ),
)
def test_cuml_fit_enforces_layout_memory_solver_and_result_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    solver_message: str,
    expected_iterations: int,
    free_bytes: int,
    nonfinite: bool,
    invalid_device_dtype: bool,
    error: str | None,
) -> None:
    fit_x = np.arange(12, dtype=np.float64).reshape(6, 2) + 0.25

    class Pool:
        calls = 0

        def free_all_blocks(self) -> None:
            self.calls += 1

    class Classifier:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["penalty_normalized"] is True
            assert kwargs["linesearch_max_iter"] == 100
            assert kwargs["tol"] == 1e-3

        def fit(self, x: Any, y: np.ndarray, **kwargs: object) -> None:
            assert x.dtype == np.float32 and x.flags.f_contiguous
            assert y.dtype == np.int32
            np.testing.assert_array_equal(x.values, fit_x.astype(np.float32))
            assert len(x.data.calls) == 1
            _, copied_bytes = x.data.calls[0]
            assert copied_bytes == fit_x.astype(np.float32).nbytes
            assert kwargs == {"sample_weight": None, "convert_dtype": False}
            if solver_message:
                print(solver_message)
            self.classes_ = np.arange(3)
            self.coef_ = np.full((3, x.shape[1]), np.nan if nonfinite else 0.0, dtype=np.float32)
            self.intercept_ = np.zeros(3, dtype=np.float32)
            self.n_iter_ = np.asarray([expected_iterations])

    class DevicePointer:
        def __init__(self, values: np.ndarray) -> None:
            self.values = values
            self.calls: list[tuple[int, int]] = []

        def copy_from_host(self, pointer: int, size: int) -> None:
            self.calls.append((pointer, size))
            ctypes.memmove(self.values.ctypes.data, pointer, size)

    def empty(shape: tuple[int, ...], *, dtype: Any, order: str) -> SimpleNamespace:
        assert shape == fit_x.shape
        assert dtype is np.float32
        assert order == "F"
        values = np.empty(
            shape,
            dtype=np.float64 if invalid_device_dtype else dtype,
            order="F",
        )
        return SimpleNamespace(
            values=values,
            data=DevicePointer(values),
            dtype=values.dtype,
            flags=values.flags,
            nbytes=values.nbytes,
            shape=values.shape,
        )

    def asarray(value: object) -> np.ndarray:
        result = np.asarray(value)
        assert result.ndim == 1
        return result

    pool = Pool()
    cupy = SimpleNamespace(
        asarray=asarray,
        empty=empty,
        cuda=SimpleNamespace(runtime=SimpleNamespace(memGetInfo=lambda: (free_bytes, 0))),
        get_default_memory_pool=lambda: pool,
    )
    original_import = role_probes.importlib.import_module

    def import_module(name: str) -> object:
        if name == "cupy":
            return cupy
        if name == "cuml.linear_model":
            return SimpleNamespace(LogisticRegression=Classifier)
        return original_import(name)

    monkeypatch.setattr(role_probes.importlib, "import_module", import_module)
    config = replace(_training_config(), optimizer=_cuml_optimizer_config(tolerance=1e-3))
    if error is not None:
        expected_error = (
            role_probes.ProbeConvergenceError if solver_message else (RuntimeError, MemoryError)
        )
        with pytest.raises(expected_error, match=error):
            _fit_parameters(fit_x, np.arange(6) % 3, 1.0, config, 3)
    else:
        coefficients, intercepts, fit_iterations = _fit_parameters(
            fit_x, np.arange(6) % 3, 1.0, config, 3
        )
        assert coefficients.shape == (3, 2)
        assert intercepts.shape == (3,)
        assert fit_iterations == [expected_iterations]
    assert pool.calls == 2


def test_optimizer_runtime_digest_and_numerical_contract_are_immutable() -> None:
    optimizer = _cuml_optimizer_config()
    with pytest.raises(ValueError, match="digest"):
        replace(optimizer, runtime_sha256="0" * 64)
    runtime = json.loads(optimizer.runtime_json)
    runtime["matrix"]["array_order"] = "C"
    with pytest.raises(ValueError, match="numerical settings"):
        optimizer_config_from_runtime(runtime)


def test_cuml_runtime_identity_is_reconstructed_and_checked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    logistic_path = tmp_path / "logistic_regression.py"
    qn_path = tmp_path / "qn.abi3.so"
    libcuml_path = tmp_path / "libcuml.so"
    for path, content in (
        (logistic_path, b"logistic"),
        (qn_path, b"qn"),
        (libcuml_path, b"libcuml"),
    ):
        path.write_bytes(content)
    cupy = SimpleNamespace(
        cuda=SimpleNamespace(
            runtime=SimpleNamespace(
                getDevice=lambda: 0,
                getDeviceProperties=lambda _: {
                    "name": b"test-gpu",
                    "major": 8,
                    "minor": 9,
                },
                driverGetVersion=lambda: 13_000,
                runtimeGetVersion=lambda: 13_020,
            )
        )
    )
    modules = {
        "cupy": cupy,
        "cuml.linear_model.logistic_regression": SimpleNamespace(__file__=str(logistic_path)),
        "cuml.solvers.qn": SimpleNamespace(__file__=str(qn_path)),
    }
    original_import = role_probes.importlib.import_module
    monkeypatch.setattr(
        role_probes.importlib,
        "import_module",
        lambda name: modules.get(name) or original_import(name),
    )
    monkeypatch.setattr(role_probes.importlib.metadata, "version", lambda _: "test-version")
    monkeypatch.setattr(
        role_probes.importlib.metadata,
        "distribution",
        lambda _: SimpleNamespace(locate_file=lambda _: libcuml_path),
    )

    optimizer = role_probes.cuml_optimizer_config(tolerance=1e-3)
    runtime = json.loads(optimizer.runtime_json)
    assert optimizer_config_from_runtime(runtime) == optimizer
    assert optimizer.tolerance == runtime["solver"]["tol"] == 1e-3
    assert runtime["cuda"] == {
        "compute_capability": "8.9",
        "device_name": "test-gpu",
        "driver_version": 13_000,
        "runtime_version": 13_020,
    }
    assert runtime["implementation_sha256"] == {
        "cuml/linear_model/logistic_regression.py": role_probes._path_sha256(logistic_path),
        "cuml/solvers/qn.abi3.so": role_probes._path_sha256(qn_path),
        "libcuml/lib64/libcuml.so": role_probes._path_sha256(libcuml_path),
    }
    role_probes._validate_optimizer_runtime(optimizer)
    modules["cuml.linear_model.logistic_regression"] = SimpleNamespace(__file__=None)
    with pytest.raises(ValueError, match="implementation is not hashable"):
        role_probes.cuml_optimizer_config()
    modules["cuml.linear_model.logistic_regression"] = SimpleNamespace(__file__=str(logistic_path))
    monkeypatch.setattr(
        role_probes.importlib.metadata,
        "distribution",
        lambda _: SimpleNamespace(locate_file=lambda _: tmp_path / "missing.so"),
    )
    with pytest.raises(ValueError, match="runtime file is missing"):
        role_probes.cuml_optimizer_config()
    monkeypatch.setattr(
        role_probes,
        "cuml_optimizer_config",
        lambda **_: role_probes.sklearn_optimizer_config(),
    )
    with pytest.raises(ValueError, match="live optimizer runtime"):
        role_probes._validate_optimizer_runtime(optimizer)


def test_cuml_runtime_identity_fails_without_optional_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = role_probes.importlib.import_module

    def import_module(name: str) -> object:
        if name == "cupy":
            raise ImportError("missing")
        return original_import(name)

    monkeypatch.setattr(role_probes.importlib, "import_module", import_module)
    with pytest.raises(RuntimeError, match="pinned isolated GPU environment"):
        role_probes.cuml_optimizer_config()


def test_cuml_qn_matches_sklearn_reference_when_explicitly_enabled() -> None:
    if os.environ.get("REASONESE_TEST_CUML") != "1":
        pytest.skip("set REASONESE_TEST_CUML=1 in the pinned GPU fitter environment")
    rng = np.random.default_rng(2606)
    x = rng.normal(size=(600, 12)).astype(np.float32)
    logits = x[:, :5] + rng.normal(scale=1.5, size=(600, 5))
    y = np.argmax(logits, axis=1).astype(np.int64)
    regularization = 0.1
    sklearn_parameters = _fit_parameters(x, y, regularization, _training_config(), 5)
    cuml_config = replace(_training_config(), optimizer=role_probes.cuml_optimizer_config())
    cuml_parameters = _fit_parameters(x, y, regularization, cuml_config, 5)
    sklearn_probabilities = _predict_parameters(sklearn_parameters[0], sklearn_parameters[1], x)
    cuml_probabilities = _predict_parameters(cuml_parameters[0], cuml_parameters[1], x)
    np.testing.assert_allclose(cuml_probabilities, sklearn_probabilities, rtol=5e-4, atol=5e-4)
    np.testing.assert_array_equal(
        np.argmax(cuml_probabilities, axis=1), np.argmax(sklearn_probabilities, axis=1)
    )
    sklearn_objective = _multinomial_objective(
        sklearn_parameters[0], sklearn_parameters[1], x, y, regularization
    )
    cuml_objective = _multinomial_objective(
        cuml_parameters[0], cuml_parameters[1], x, y, regularization
    )
    assert cuml_objective == pytest.approx(sklearn_objective, rel=1e-5)
    duplicated = _fit_parameters(
        np.tile(x, (2, 1)), np.tile(y, 2), regularization * 2, cuml_config, 5
    )
    duplicated_probabilities = _predict_parameters(duplicated[0], duplicated[1], x)
    np.testing.assert_allclose(duplicated_probabilities, cuml_probabilities, rtol=5e-4, atol=5e-4)
    duplicated_objective = _multinomial_objective(
        duplicated[0],
        duplicated[1],
        np.tile(x, (2, 1)),
        np.tile(y, 2),
        regularization * 2,
    )
    assert duplicated_objective == pytest.approx(cuml_objective, rel=1e-5)
    zero_x = np.zeros((100, 2), dtype=np.float32)
    imbalanced_y = np.asarray([0] * 80 + [1] * 20)
    imbalance = _fit_parameters(zero_x, imbalanced_y, 1.0, cuml_config, 2)
    imbalance_probabilities = _predict_parameters(imbalance[0], imbalance[1], zero_x[:1])[0]
    np.testing.assert_allclose(imbalance_probabilities, (0.8, 0.2), rtol=5e-4, atol=5e-4)


def test_qualification_recomputes_untouched_conversation_metrics() -> None:
    bundle = train_role_probe(_dataset(), _training_config())
    qualified = _qualify(bundle)
    assert qualified.qualification is not None
    assert qualified.qualification.passed
    assert qualified.qualification.calibration_conversation_count == 12
    assert qualified.qualification.test_metrics.document_accuracy == 1.0
    assert qualified.qualification.test.bootstrap_auc.auc == 1.0
    assert qualified.qualification.calibration.usable
    assert qualified.qualification.threshold_reasoning_sensitivity == 1.0
    assert qualified.qualification.threshold_final_specificity == 1.0
    assert qualified.qa_eligible

    test_scores = qualified.qualification.test.scores
    changed_threshold = replace(
        qualified.qualification.calibration,
        threshold=np.nextafter(max(test_scores.reasoning_scores), np.inf),
    )
    changed_qualification = replace(qualified.qualification, calibration=changed_threshold)
    assert changed_qualification.test.passed
    assert changed_qualification.threshold_reasoning_sensitivity == 0.0
    assert changed_qualification.threshold_final_specificity == 1.0
    assert not changed_qualification.passed

    projection = project_role(
        qualified,
        np.asarray([[4.0, 0.0, 0.0, 0.0, 0.1]], dtype=np.float32),
        provenance=qualified.provenance,
        layer_index=qualified.selected_layer_index,
    )
    assert projection.predicted_role == "reasoning"
    assert np.allclose(projection.token_probabilities.sum(axis=1), 1.0, atol=1e-12)
    assert projection.mean_probabilities[0] > 0.99


def test_qualification_rejects_surrogate_or_nonconversation_data() -> None:
    bundle = train_role_probe(_dataset(), _training_config())
    with pytest.raises(ValueError, match="untouched native conversation"):
        _qualify(bundle, calibration=_dataset())
    surrogate = _dataset(
        kind="untouched-native-conversations",
        document_count=12,
        model_revision="surrogate-commit",
        document_prefix="calibration",
    )
    with pytest.raises(ValueError, match="exact model pipeline"):
        _qualify(bundle, calibration=surrogate)
    protocol_mismatch = replace(
        _dataset(
            kind="untouched-native-conversations",
            document_count=12,
            document_prefix="calibration",
            content_token_offset=5_000,
        ),
        provenance=replace(
            _dataset(kind="untouched-native-conversations", document_count=12).provenance,
            extraction_protocol="different-extraction-protocol",
        ),
    )
    with pytest.raises(ValueError, match="exact model pipeline"):
        _qualify(bundle, calibration=protocol_mismatch)

    conversations = _dataset(
        kind="untouched-native-conversations",
        document_count=12,
        document_prefix="calibration",
        content_token_offset=5_000,
    )
    overlapping_ids = conversations.document_ids.copy()
    overlapping_ids[overlapping_ids == "calibration-000"] = bundle.split.test[0]
    with pytest.raises(ValueError, match="disjoint from neutral"):
        _qualify(bundle, calibration=replace(conversations, document_ids=overlapping_ids))

    with pytest.raises(ValueError, match="document-disjoint"):
        _qualify(bundle, test=conversations)
    duplicated_test_content = _dataset(
        kind="untouched-native-conversations",
        document_count=12,
        document_prefix="test",
        content_token_offset=5_000,
    )
    with pytest.raises(ValueError, match="content-disjoint"):
        _qualify(bundle, test=duplicated_test_content)

    reused_reasoning = _dataset(
        kind="untouched-native-conversations",
        document_count=12,
        document_prefix="test",
        content_token_offset=10_000,
    )
    token_ids = reused_reasoning.content_token_id.copy()
    reasoning_rows = reused_reasoning.roles == "reasoning"
    token_ids[reasoning_rows] -= 5_000
    with pytest.raises(ValueError, match="segments must be content-disjoint"):
        _qualify(bundle, test=replace(reused_reasoning, content_token_id=token_ids))

    neutral_overlap = _dataset(
        kind="untouched-native-conversations",
        document_count=12,
        document_prefix="calibration",
    )
    with pytest.raises(ValueError, match="native and neutral"):
        _qualify(bundle, calibration=neutral_overlap)


def test_every_native_conversation_requires_both_measured_roles() -> None:
    conversations = _dataset(kind="untouched-native-conversations", document_count=6)
    keep = ~(
        (conversations.document_ids == "conversation-000") & (conversations.roles == "assistant")
    )
    with pytest.raises(ValueError, match="every untouched conversation"):
        replace(
            conversations,
            activations=conversations.activations[keep],
            document_ids=conversations.document_ids[keep],
            roles=conversations.roles[keep],
            content_token_index=conversations.content_token_index[keep],
            content_token_id=conversations.content_token_id[keep],
            sequence_token_index=conversations.sequence_token_index[keep],
            filler_document_ids=conversations.filler_document_ids[keep],
        )


def test_failed_empirical_threshold_keeps_probe_out_of_qa() -> None:
    bundle = train_role_probe(_dataset(), _training_config())
    conversations = _dataset(
        kind="untouched-native-conversations",
        document_count=12,
        document_prefix="test",
        content_token_offset=10_000,
    )
    # Exchange the two role directions while keeping labels unchanged.
    flipped = conversations.activations.copy()
    flipped[:, :, [0, 1]] = flipped[:, :, [1, 0]]
    failed = _qualify(bundle, test=replace(conversations, activations=flipped))
    assert failed.qualification is not None
    assert not failed.qualification.passed
    assert not failed.qa_eligible
    with pytest.raises(ValueError, match="lacks passing"):
        project_role(
            failed,
            flipped[:2, 1, :],
            provenance=failed.provenance,
            layer_index=failed.selected_layer_index,
        )


def test_qualification_rejects_metrics_inconsistent_with_native_gates() -> None:
    qualified = _qualify(train_role_probe(_dataset(), _training_config()))
    assert qualified.qualification is not None
    metrics = replace(
        qualified.qualification.test_metrics,
        per_role_accuracy=(("reasoning", 0.0), ("assistant", 1.0)),
    )
    with pytest.raises(ValueError, match="role gate does not match"):
        replace(qualified.qualification, test_metrics=metrics)


def test_qualification_metrics_must_keep_exact_role_order_and_confusion_shape() -> None:
    qualified = _qualify(train_role_probe(_dataset(), _training_config()))
    assert qualified.qualification is not None
    metrics = qualified.qualification.test_metrics
    with pytest.raises(ValueError, match="per-role metrics must be non-empty and aligned"):
        replace(
            metrics,
            per_role_document_accuracy=tuple(reversed(metrics.per_role_document_accuracy)),
        )
    with pytest.raises(ValueError, match="conversation confusion matrix"):
        replace(
            qualified,
            qualification=replace(
                qualified.qualification,
                test_metrics=replace(
                    metrics,
                    confusion_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                ),
            ),
        )


def test_artifact_round_trip_preserves_exact_numpy_scoring(tmp_path: Path) -> None:
    bundle = qualify_role_probe(
        train_role_probe(_dataset(), _training_config()),
        _dataset(
            kind="untouched-native-conversations",
            document_count=12,
            document_prefix="calibration",
            content_token_offset=5_000,
        ),
        _dataset(
            kind="untouched-native-conversations",
            document_count=12,
            document_prefix="test",
            content_token_offset=10_000,
        ),
        prompt_partition_name="native-prompt-partitions.json",
        prompt_partition_sha256="e" * 64,
    )
    path = tmp_path / "nemotron-role-probe.npz"
    save_role_probe(bundle, path)
    loaded = load_role_probe(path)
    activations = np.asarray(
        [[3.0, 0.0, 0.0, 0.0, 0.1], [0.0, 3.0, 0.0, 0.0, 0.1]],
        dtype=np.float32,
    )
    expected = project_role(
        bundle,
        activations,
        provenance=bundle.provenance,
        layer_index=bundle.selected_layer_index,
    )
    actual = project_role(
        loaded,
        activations,
        provenance=loaded.provenance,
        layer_index=loaded.selected_layer_index,
    )
    assert actual.roles == expected.roles
    assert actual.predicted_role == expected.predicted_role
    assert np.array_equal(actual.token_probabilities, expected.token_probabilities)
    assert actual.mean_probabilities == expected.mean_probabilities
    assert loaded.provenance.weights_hash_kind == "hf_lfs_manifest_sha256"
    assert loaded.split.fingerprint == bundle.split.fingerprint
    assert loaded.selected_layer_index == bundle.selected_layer_index
    assert loaded.development_candidates == bundle.development_candidates
    assert loaded.failed_candidates == ()


def test_existing_all_success_artifact_loads_with_no_failed_candidates(tmp_path: Path) -> None:
    path = tmp_path / "pre-failure-evidence-probe.npz"
    save_role_probe(train_role_probe(_dataset(), _training_config()), path)
    with np.load(path, allow_pickle=False) as archive:
        envelope = json.loads(np.asarray(archive["metadata"]).tobytes().decode())
        coefficients = archive["coefficients"].copy()
        intercepts = archive["intercepts"].copy()
    metadata = envelope["probe"]
    assert metadata.pop("failed_candidates") == []
    envelope["fingerprint"] = role_probes._artifact_fingerprint(metadata, coefficients, intercepts)
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata=np.frombuffer(role_probes._json(envelope), dtype=np.uint8),
            coefficients=coefficients,
            intercepts=intercepts,
        )
    assert load_role_probe(path).failed_candidates == ()


def test_artifact_rejects_mismatched_native_test_document_count(tmp_path: Path) -> None:
    bundle = _qualify(train_role_probe(_dataset(), _training_config()))
    path = tmp_path / "mismatched-native-count.npz"
    save_role_probe(bundle, path)
    with np.load(path, allow_pickle=False) as archive:
        envelope = json.loads(np.asarray(archive["metadata"]).tobytes().decode())
        coefficients = archive["coefficients"].copy()
        intercepts = archive["intercepts"].copy()
    metadata = envelope["probe"]
    metadata["qualification"]["test_metrics"]["document_count"] = 1
    envelope["fingerprint"] = role_probes._artifact_fingerprint(metadata, coefficients, intercepts)
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata=np.frombuffer(role_probes._json(envelope), dtype=np.uint8),
            coefficients=coefficients,
            intercepts=intercepts,
        )
    with pytest.raises(ValueError, match="native test metric count"):
        load_role_probe(path)


def test_artifact_fingerprint_uses_the_exact_stored_parameter_dtype(tmp_path: Path) -> None:
    bundle = train_role_probe(_dataset(), _training_config())
    float32_bundle = replace(
        bundle,
        coefficients=bundle.coefficients.astype(np.float32),
        intercepts=bundle.intercepts.astype(np.float32),
    )
    path = tmp_path / "float32-source.npz"
    save_role_probe(float32_bundle, path)
    loaded = load_role_probe(path)
    np.testing.assert_array_equal(loaded.coefficients, float32_bundle.coefficients)
    np.testing.assert_array_equal(loaded.intercepts, float32_bundle.intercepts)


def test_artifact_fingerprint_detects_parameter_tampering(tmp_path: Path) -> None:
    bundle = train_role_probe(_dataset(), _training_config())
    path = tmp_path / "probe.npz"
    save_role_probe(bundle, path)
    with np.load(path, allow_pickle=False) as archive:
        metadata = archive["metadata"].copy()
        coefficients = archive["coefficients"].copy()
        intercepts = archive["intercepts"].copy()
    coefficients[0, 0] += 0.25
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata=metadata,
            coefficients=coefficients,
            intercepts=intercepts,
        )
    with pytest.raises(ValueError, match="fingerprint"):
        load_role_probe(path)


def test_dataset_fingerprint_covers_activations_labels_and_provenance() -> None:
    dataset = _dataset()
    original = activation_dataset_fingerprint(dataset)
    changed = dataset.activations.copy()
    changed[0, 0, 0] += 0.01
    assert activation_dataset_fingerprint(replace(dataset, activations=changed)) != original
    conversation = replace(
        dataset,
        provenance=replace(
            dataset.provenance,
            dataset_kind="untouched-native-conversations",
            filler_pool_kind="none",
            filler_source_sha256=None,
            filler_documents=0,
        ),
        filler_document_ids=np.full(len(dataset.roles), "", dtype=np.str_),
    )
    assert activation_dataset_fingerprint(conversation) != original


def test_projection_shape_finiteness_and_qualification_are_enforced() -> None:
    bundle = train_role_probe(_dataset(), _training_config())
    with pytest.raises(ValueError, match="lacks passing"):
        project_role(
            bundle,
            np.zeros((2, 5), dtype=np.float32),
            provenance=bundle.provenance,
            layer_index=bundle.selected_layer_index,
        )
    with pytest.raises(ValueError, match="shape"):
        project_role(
            bundle,
            np.zeros((2, 4), dtype=np.float32),
            provenance=bundle.provenance,
            layer_index=bundle.selected_layer_index,
            require_qa_eligible=False,
        )
    nonfinite = np.zeros((2, 5), dtype=np.float32)
    nonfinite[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        project_role(
            bundle,
            nonfinite,
            provenance=bundle.provenance,
            layer_index=bundle.selected_layer_index,
            require_qa_eligible=False,
        )
    mismatched = replace(bundle.provenance, model_dtype="float16")
    with pytest.raises(ValueError, match="exact model pipeline"):
        project_role(
            bundle,
            np.zeros((2, 5), dtype=np.float32),
            provenance=mismatched,
            layer_index=bundle.selected_layer_index,
            require_qa_eligible=False,
        )
    with pytest.raises(ValueError, match="trained probe layer"):
        project_role(
            bundle,
            np.zeros((2, 5), dtype=np.float32),
            provenance=bundle.provenance,
            layer_index=3,
            require_qa_eligible=False,
        )
    with pytest.raises(ValueError, match="dtype"):
        project_role(
            bundle,
            np.zeros((2, 5), dtype=np.float16),
            provenance=bundle.provenance,
            layer_index=bundle.selected_layer_index,
            require_qa_eligible=False,
        )


def test_probe_rejects_metrics_that_do_not_match_its_role_space() -> None:
    bundle = train_role_probe(_dataset(), _training_config())
    swapped = replace(
        bundle.validation_metrics,
        per_role_accuracy=tuple(reversed(bundle.validation_metrics.per_role_accuracy)),
        per_role_document_accuracy=tuple(
            reversed(bundle.validation_metrics.per_role_document_accuracy)
        ),
    )
    with pytest.raises(ValueError, match="metrics do not match probe roles"):
        replace(bundle, validation_metrics=swapped)

    with pytest.raises(ValueError, match="successful candidate evidence must follow grid order"):
        replace(bundle, development_candidates=tuple(reversed(bundle.development_candidates)))

    malformed_candidate = replace(
        bundle.development_candidates[0],
        metrics=replace(bundle.development_candidates[0].metrics, document_count=999),
    )
    with pytest.raises(ValueError, match="development split"):
        replace(
            bundle,
            development_candidates=(malformed_candidate,) + bundle.development_candidates[1:],
        )

    malformed_roles = replace(
        bundle.development_candidates[0],
        metrics=replace(
            bundle.development_candidates[0].metrics,
            per_role_accuracy=tuple(
                reversed(bundle.development_candidates[0].metrics.per_role_accuracy)
            ),
            per_role_document_accuracy=tuple(
                reversed(bundle.development_candidates[0].metrics.per_role_document_accuracy)
            ),
        ),
    )
    with pytest.raises(ValueError, match="candidate metrics do not match probe roles"):
        replace(
            bundle,
            development_candidates=(malformed_roles,) + bundle.development_candidates[1:],
        )
