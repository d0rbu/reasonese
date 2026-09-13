"""Activation-role probe training, qualification, and artifact contracts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from reasonese.role_probes import (
    ActivationDataset,
    ActivationProvenance,
    ProbeTrainingConfig,
    RoleProbe,
    _parameters,
    _predict_parameters,
    activation_dataset_fingerprint,
    load_role_probe,
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
        layer_index=selected_layer,
        minimum_neutral_accuracy=0.95,
        minimum_neutral_per_role_accuracy=0.95,
        minimum_neutral_document_accuracy=0.95,
        minimum_neutral_per_role_document_accuracy=0.95,
        lambda_grid=(0.01, 0.1),
        train_fraction=0.7,
        validation_fraction=0.15,
        seed=19,
        max_iterations=500,
    )


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
    assert bundle.training.layer_index == 7
    assert bundle.training.tolerance == 1e-4
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
        layer_index=bundle.training.layer_index,
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
        layer_index=qualified.training.layer_index,
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
            layer_index=failed.training.layer_index,
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
        layer_index=bundle.training.layer_index,
    )
    actual = project_role(
        loaded,
        activations,
        provenance=loaded.provenance,
        layer_index=loaded.training.layer_index,
    )
    assert actual.roles == expected.roles
    assert actual.predicted_role == expected.predicted_role
    assert np.array_equal(actual.token_probabilities, expected.token_probabilities)
    assert actual.mean_probabilities == expected.mean_probabilities
    assert loaded.provenance.weights_hash_kind == "hf_lfs_manifest_sha256"
    assert loaded.split.fingerprint == bundle.split.fingerprint


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
            layer_index=bundle.training.layer_index,
        )
    with pytest.raises(ValueError, match="shape"):
        project_role(
            bundle,
            np.zeros((2, 4), dtype=np.float32),
            provenance=bundle.provenance,
            layer_index=bundle.training.layer_index,
            require_qa_eligible=False,
        )
    nonfinite = np.zeros((2, 5), dtype=np.float32)
    nonfinite[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        project_role(
            bundle,
            nonfinite,
            provenance=bundle.provenance,
            layer_index=bundle.training.layer_index,
            require_qa_eligible=False,
        )
    mismatched = replace(bundle.provenance, model_dtype="float16")
    with pytest.raises(ValueError, match="exact model pipeline"):
        project_role(
            bundle,
            np.zeros((2, 5), dtype=np.float32),
            provenance=mismatched,
            layer_index=bundle.training.layer_index,
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
            layer_index=bundle.training.layer_index,
            require_qa_eligible=False,
        )


def test_probe_rejects_metrics_that_do_not_match_its_role_space() -> None:
    bundle = train_role_probe(_dataset(), _training_config())
    swapped = replace(
        bundle.validation_metrics,
        per_role_accuracy=tuple(reversed(bundle.validation_metrics.per_role_accuracy)),
    )
    with pytest.raises(ValueError, match="metrics do not match probe roles"):
        replace(bundle, validation_metrics=swapped)
