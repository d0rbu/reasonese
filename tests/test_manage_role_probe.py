"""Frozen role-probe management CLI contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import reasonese.manage_role_probe as manage
from reasonese.role_probe_extraction import NEMOTRON_ADAPTER, ProbeRole
from reasonese.role_probes import sklearn_optimizer_config


def _protocol() -> dict[str, Any]:
    return {
        "models": {
            "nemotron": {
                "revision": NEMOTRON_ADAPTER.model_revision,
                "layer_index": 26,
            },
            "gemma": {
                "revision": manage.GEMMA_ADAPTER.model_revision,
                "layer_index": 30,
            },
        },
        "neutral_split": {"train": 0.6, "development": 0.2, "test": 0.2, "seed": 0},
        "neutral_gate": {
            "token_accuracy": 0.9,
            "per_role_token_accuracy": 0.85,
            "document_macro_token_accuracy": 0.9,
            "per_role_document_macro_token_accuracy": 0.85,
        },
        "neutral_validation_documents": 60,
        "stored_activation_dtype": manage._FROZEN_ACTIVATION_DTYPE,
        "roles": [str(role) for role in ProbeRole],
        "native_dialogues_per_model": 24,
        "native_split": {"calibration": 12, "test": 12},
        "native_gate": {
            "required_roles": ["reasoning", "assistant"],
            "minimum_role_accuracy": manage.MIN_ROLE_ACCURACY,
            "minimum_document_macro_accuracy": manage.MIN_DOCUMENT_MACRO_ACCURACY,
            "reasoning_vs_final_segment_auc": manage.MIN_TEST_AUC,
            "auc_bootstrap_lower_95_bound_must_exceed": manage.MIN_AUC_BOOTSTRAP_LOWER,
        },
    }


def _write_protocol(tmp_path: Path) -> Path:
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(_protocol()), encoding="utf-8")
    return path


def _adoption_fixture(tmp_path: Path) -> tuple[argparse.Namespace, SimpleNamespace, dict[str, Any]]:
    protocol = _segment_protocol()
    protocol_path = tmp_path / "segment-protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    activations = tmp_path / "activations"
    activations.mkdir()
    (activations / "manifest.json").write_text("{}", encoding="utf-8")
    reference_path = tmp_path / "reference.npz"
    reference_path.write_bytes(b"reference probe")
    parameters = tmp_path / "layer-13-lambda-100.npz"
    roles = tuple(str(role) for role in ProbeRole)
    hidden_size = 3
    coefficients = np.arange(len(roles) * hidden_size, dtype=np.float64).reshape(
        len(roles), hidden_size
    )
    intercepts = np.arange(len(roles), dtype=np.float64)
    mean = np.asarray([0.5, 1.5, 2.5], dtype=np.float64)
    scale = np.asarray([1.5, 2.5, 3.5], dtype=np.float64)
    standardized_coefficients = coefficients * scale[np.newaxis, :]
    standardized_bias = intercepts + coefficients @ mean
    with parameters.open("wb") as handle:
        np.savez(
            handle,
            raw_coefficients=coefficients,
            raw_bias=intercepts,
            standardized_coefficients=standardized_coefficients,
            standardized_bias=standardized_bias,
            mean=mean,
            scale=scale,
        )
    metrics = {
        "accuracy": 0.8,
        "document_accuracy": 0.8,
        "negative_log_likelihood": 0.5,
        "per_role_accuracy": [[role, 0.8] for role in roles],
        "per_role_document_accuracy": [[role, 0.8] for role in roles],
        "confusion_matrix": [[1 if row == col else 0 for col in range(len(roles))] for row in range(len(roles))],
        "token_count": 100,
        "document_count": 50,
    }
    candidates = []
    for regularization in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0):
        candidate = {
            "layer": 13,
            "lambda": regularization,
            "development_metrics": dict(metrics),
        }
        if regularization == 100.0:
            candidate["development_metrics"] = {**metrics, "accuracy": 0.99}
            candidate["diagnostic_parameters"] = parameters.name
            candidate["diagnostic_parameters_sha256"] = manage._sha256(parameters)
            candidate["raw_coefficients_sha256"] = manage._array_sha256(coefficients)
            candidate["raw_intercepts_sha256"] = manage._array_sha256(intercepts)
        candidates.append(candidate)
    report = {
        "status": "diagnostic-only-standardized-train-development-grid",
        "baseline_probe_sha256": manage._sha256(reference_path),
        "neutral_activation_manifest_sha256": manage._sha256(activations / "manifest.json"),
        "optimizer_runtime_sha256": manage.optimizer_config_from_runtime(
            protocol["optimizer"]
        ).runtime_sha256,
        "held_out_test_rows_scored": False,
        "native_rows_scored": False,
        "refit_performed": False,
        "probe_artifact_written": False,
        "failed_candidates": 0,
        "failures": [],
        "scaler": {
            "fit_documents": 3,
            "mean_sha256": manage._array_sha256(mean),
            "scale_sha256": manage._array_sha256(scale),
        },
        "candidates": candidates,
        "successful_candidates": len(candidates),
    }
    diagnostic_path = tmp_path / "diagnostic.json"
    diagnostic_path.write_text(json.dumps(report), encoding="utf-8")
    reference = SimpleNamespace(
        provenance=SimpleNamespace(
            native_template_adapter=NEMOTRON_ADAPTER.name,
            roles=roles,
            hidden_size=hidden_size,
        ),
        split=SimpleNamespace(train=("a", "b", "c")),
    )
    args = argparse.Namespace(
        reference_probe=reference_path,
        activations=activations,
        diagnostic=diagnostic_path,
        parameters=parameters,
        protocol=protocol_path,
        output=tmp_path / "adopted.npz",
    )
    return args, reference, report


def _expanded_protocol() -> dict[str, Any]:
    protocol = _protocol()
    protocol["neutral_validation_documents"] = 250
    protocol["neutral_max_content_tokens"] = 1_024
    protocol["neutral_max_filler_tokens"] = 513
    protocol["neutral_max_sequence_tokens"] = 2_048
    protocol["neutral_filler_length_distribution"] = (
        manage.FROZEN_NEUTRAL_FILLER_LENGTH_DISTRIBUTION
    )
    protocol["native_prompt_partitions_sha256"] = "a" * 64
    protocol["neutral_target_source_sha256"] = "b" * 64
    protocol["neutral_filler_source_sha256"] = "c" * 64
    protocol["candidate_failure_policy"] = manage._CANDIDATE_FAILURE_POLICY
    protocol["optimizer"] = json.loads(sklearn_optimizer_config().runtime_json)
    protocol["models"] = {
        "nemotron": {
            "revision": NEMOTRON_ADAPTER.model_revision,
            "candidate_layers": [13, 20, 26],
        },
        "gemma": {
            "revision": manage.GEMMA_ADAPTER.model_revision,
            "candidate_layers": [15, 23, 30],
        },
    }
    return protocol


def _segment_protocol() -> dict[str, Any]:
    protocol = _expanded_protocol()
    protocol["qualification_policy"] = manage.SEGMENT_MEAN_REASONING_PROBABILITY
    protocol["native_token_metrics"] = "diagnostic only"
    protocol["models"]["nemotron"]["candidate_layers"] = [13]
    protocol["native_gate"] = {
        "required_segments": ["reasoning", "assistant"],
        "reasoning_vs_final_segment_auc": manage.MIN_TEST_AUC,
        "auc_bootstrap_lower_95_bound_must_exceed": manage.MIN_AUC_BOOTSTRAP_LOWER,
        "heldout_reasoning_sensitivity_at_calibrated_threshold": (
            manage.MIN_THRESHOLD_REASONING_SENSITIVITY
        ),
        "heldout_final_specificity_at_calibrated_threshold": (
            manage.MIN_THRESHOLD_FINAL_SPECIFICITY
        ),
    }
    return protocol


def test_frozen_protocol_accepts_both_pinned_adapters_and_rejects_drift() -> None:
    for adapter in (NEMOTRON_ADAPTER, manage.GEMMA_ADAPTER):
        config = manage._validate_frozen_protocol(_protocol(), adapter.name)
        assert config.layer_indices == (manage._ADAPTER_LEGACY_LAYER[adapter.name],)
        assert config.minimum_neutral_accuracy == 0.9
        assert config.optimizer.max_iterations == 2_000
        assert config.optimizer.tolerance == 1e-4

        expanded = manage._validate_frozen_protocol(_expanded_protocol(), adapter.name)
        expected = (13, 20, 26) if adapter is NEMOTRON_ADAPTER else (15, 23, 30)
        assert expanded.layer_indices == expected
        assert expanded.expected_document_count == 250
        assert expanded.maximum_content_tokens_per_document == 1_024

    mutations = (
        ({}, "required training fields"),
        ({**_protocol(), "neutral_validation_documents": 59}, "expanded search or 60-document"),
        ({**_protocol(), "stored_activation_dtype": "float16"}, "float32"),
        ({**_protocol(), "neutral_split": {"train": 1.0}}, "lacks neutral split"),
        ({**_protocol(), "neutral_gate": {}}, "lacks neutral split or gate"),
        ({**_protocol(), "roles": ["reasoning", "assistant"]}, "five frozen native roles"),
        ({**_protocol(), "native_dialogues_per_model": 23}, "12/12 native split"),
        ({**_protocol(), "native_split": {"calibration": 24}}, "12/12 native split"),
        ({**_protocol(), "native_gate": {}}, "frozen native gates"),
    )
    for protocol, message in mutations:
        with pytest.raises(ValueError, match=message):
            manage._validate_frozen_protocol(protocol, NEMOTRON_ADAPTER.name)
    changed_model = _protocol()
    changed_model["models"]["nemotron"]["revision"] = "changed"
    with pytest.raises(ValueError, match="pinned native adapter"):
        manage._validate_frozen_protocol(changed_model, NEMOTRON_ADAPTER.name)
    changed_layers = _expanded_protocol()
    changed_layers["models"]["nemotron"]["candidate_layers"] = [13, 26]
    changed_config = manage._validate_frozen_protocol(changed_layers, NEMOTRON_ADAPTER.name)
    assert changed_config.layer_indices == (13, 26)
    assert changed_config.protocol_sha256 != expanded.protocol_sha256
    for invalid_layers in ([20, 13, 26], [13, True, 26]):
        invalid = _expanded_protocol()
        invalid["models"]["nemotron"]["candidate_layers"] = invalid_layers
        with pytest.raises(ValueError, match="invalid document or layer search"):
            manage._validate_frozen_protocol(invalid, NEMOTRON_ADAPTER.name)

    segment = manage._validate_frozen_protocol(_segment_protocol(), NEMOTRON_ADAPTER.name)
    assert segment.layer_indices == (13,)
    assert segment.qualification_policy == manage.SEGMENT_MEAN_REASONING_PROBABILITY
    for field, value in (
        ("native_token_metrics", "gating"),
        ("qualification_policy", "unknown"),
    ):
        invalid = _segment_protocol()
        invalid[field] = value
        with pytest.raises(ValueError):
            manage._validate_frozen_protocol(invalid, NEMOTRON_ADAPTER.name)
    for field in ("neutral_validation_documents", "neutral_max_content_tokens"):
        invalid = _expanded_protocol()
        invalid[field] = True
        with pytest.raises(ValueError, match="invalid document or layer search"):
            manage._validate_frozen_protocol(invalid, NEMOTRON_ADAPTER.name)
    invalid_seed = _expanded_protocol()
    invalid_seed["neutral_split"] = {
        "train": 0.6,
        "development": 0.2,
        "test": 0.2,
        "seed": True,
    }
    with pytest.raises(ValueError, match="split and gate values must be numeric"):
        manage._validate_frozen_protocol(invalid_seed, NEMOTRON_ADAPTER.name)
    missing_source = _expanded_protocol()
    del missing_source["neutral_target_source_sha256"]
    with pytest.raises(ValueError, match="bind target and filler sources"):
        manage._validate_frozen_protocol(missing_source, NEMOTRON_ADAPTER.name)
    for field in ("neutral_max_filler_tokens", "neutral_max_sequence_tokens"):
        invalid = _expanded_protocol()
        invalid[field] = 512 if field == "neutral_max_filler_tokens" else 0
        with pytest.raises(ValueError, match="invalid neutral construction"):
            manage._validate_frozen_protocol(invalid, NEMOTRON_ADAPTER.name)
    invalid_distribution = _expanded_protocol()
    invalid_distribution["neutral_filler_length_distribution"] = "uniform"
    with pytest.raises(ValueError, match="invalid neutral construction"):
        manage._validate_frozen_protocol(invalid_distribution, NEMOTRON_ADAPTER.name)


def test_protocol_rejects_invalid_training_bindings() -> None:
    for field in ("models", "neutral_split", "neutral_gate"):
        invalid = _protocol()
        if field == "models":
            invalid[field] = {"nemotron": []}
        else:
            invalid[field] = []
        with pytest.raises(ValueError, match="invalid training fields"):
            manage._validate_frozen_protocol(invalid, NEMOTRON_ADAPTER.name)

    missing_partitions = _expanded_protocol()
    del missing_partitions["native_prompt_partitions_sha256"]
    with pytest.raises(ValueError, match="bind native prompt partitions"):
        manage._validate_frozen_protocol(missing_partitions, NEMOTRON_ADAPTER.name)
    missing_optimizer = _expanded_protocol()
    del missing_optimizer["optimizer"]
    with pytest.raises(ValueError, match="bind the optimizer runtime"):
        manage._validate_frozen_protocol(missing_optimizer, NEMOTRON_ADAPTER.name)
    missing_failure_policy = _expanded_protocol()
    del missing_failure_policy["candidate_failure_policy"]
    with pytest.raises(ValueError, match="candidate failure policy"):
        manage._validate_frozen_protocol(missing_failure_policy, NEMOTRON_ADAPTER.name)
    wrong_failure_policy = _expanded_protocol()
    wrong_failure_policy["candidate_failure_policy"] = "exclude all runtime failures"
    with pytest.raises(ValueError, match="candidate failure policy"):
        manage._validate_frozen_protocol(wrong_failure_policy, NEMOTRON_ADAPTER.name)
    mismatched_test_fraction = _protocol()
    mismatched_test_fraction["neutral_split"] = {
        "train": 0.6,
        "development": 0.2,
        "test": 0.3,
        "seed": 0,
    }
    with pytest.raises(ValueError, match="test fraction"):
        manage._validate_frozen_protocol(mismatched_test_fraction, NEMOTRON_ADAPTER.name)


def test_json_partition_and_parser_boundaries(tmp_path: Path) -> None:
    value = tmp_path / "value.json"
    value.write_text('{"ok":true}', encoding="utf-8")
    assert manage._json_object(value) == {"ok": True}
    assert len(manage._sha256(value)) == 64
    value.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON object"):
        manage._json_object(value)
    value.write_text("not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON object"):
        manage._json_object(value)

    partitions = tmp_path / "partitions.json"
    partitions.write_text(
        json.dumps(
            [
                {"source_id": "cal-a", "split": "calibration"},
                {"source_id": "test-a", "split": "test"},
            ]
        ),
        encoding="utf-8",
    )
    assert manage._partition_document_ids(partitions, "calibration") == {"cal-a"}
    partitions.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a list"):
        manage._partition_document_ids(partitions, "calibration")
    partitions.write_text('[{"source_id":"x"}]', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid native prompt partition"):
        manage._partition_document_ids(partitions, "calibration")

    parser = manage.build_parser()
    args = parser.parse_args(
        [
            "train",
            "--adapter",
            NEMOTRON_ADAPTER.name,
            "--activations",
            "a",
            "--protocol",
            "p",
            "--output",
            "o",
        ]
    )
    assert args.run is manage._train
    assert args.activations == Path("a")


def test_train_wires_frozen_dataset_and_reports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = SimpleNamespace(
        document_ids=np.asarray([f"doc-{index}" for index in range(60)]),
        provenance=SimpleNamespace(
            activation_dtype="float32",
            layer_indices=(26,),
            native_template_adapter=NEMOTRON_ADAPTER.name,
            model_id=NEMOTRON_ADAPTER.model_id,
            model_revision=NEMOTRON_ADAPTER.model_revision,
        ),
    )
    split = SimpleNamespace(
        train=tuple(range(36)), validation=tuple(range(12)), test=tuple(range(12))
    )
    probe = SimpleNamespace(
        neutral_valid=True,
        qa_eligible=False,
        selected_layer_index=26,
        regularization_lambda=0.25,
        development_candidates=(object(), object()),
        failed_candidates=(object(),),
        split=split,
    )
    saved: dict[str, Any] = {}
    monkeypatch.setattr(manage, "load_activation_dataset", lambda _path: dataset)
    monkeypatch.setattr(manage, "train_role_probe", lambda actual, config: probe)
    monkeypatch.setattr(
        manage, "save_role_probe", lambda actual, path: saved.update(probe=actual, path=path)
    )
    args = argparse.Namespace(
        protocol=_write_protocol(tmp_path),
        activations=tmp_path / "activations",
        adapter=NEMOTRON_ADAPTER.name,
        output=tmp_path / "probe.npz",
    )
    manage._train(args)
    report = json.loads(capsys.readouterr().out)
    assert report["neutral_valid"] is True
    assert report["successful_candidates"] == 2
    assert report["failed_candidates"] == 1
    assert report["split"] == {"train": 36, "validation": 12, "test": 12}
    assert saved == {"probe": probe, "path": args.output}

    dataset.document_ids = np.asarray(["one"])
    with pytest.raises(ValueError, match="document count"):
        manage._train(args)
    dataset.document_ids = np.asarray([f"doc-{index}" for index in range(60)])
    dataset.provenance.activation_dtype = "float16"
    with pytest.raises(ValueError, match="float32"):
        manage._train(args)
    dataset.provenance.activation_dtype = "float32"
    dataset.provenance.model_revision = "wrong"
    with pytest.raises(ValueError, match="selected native adapter"):
        manage._train(args)


def test_train_rejects_expanded_dataset_identity_before_fitting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    protocol = _expanded_protocol()
    protocol_path = tmp_path / "expanded-protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    activations = tmp_path / "activations"
    activations.mkdir()
    (activations / "manifest.json").write_text(
        json.dumps(
            {
                "max_content_tokens": 1_024,
                "max_filler_tokens": 513,
                "max_sequence_tokens": 2_048,
                "seed": 0,
                "extraction_protocol": "role-confusion-appendix-g-reasoning-assistant-v1",
            }
        ),
        encoding="utf-8",
    )
    dataset = SimpleNamespace(
        document_ids=np.asarray([f"doc-{index}" for index in range(250)]),
        provenance=SimpleNamespace(
            activation_dtype="float32",
            layer_indices=(13, 20, 26),
            source_sha256="b" * 64,
            filler_source_sha256="c" * 64,
            native_template_adapter=NEMOTRON_ADAPTER.name,
            model_id=NEMOTRON_ADAPTER.model_id,
            model_revision=NEMOTRON_ADAPTER.model_revision,
        ),
    )
    fit_calls: list[object] = []
    monkeypatch.setattr(manage, "load_activation_dataset", lambda _path: dataset)
    monkeypatch.setattr(
        manage,
        "train_role_probe",
        lambda *_args, **_kwargs: fit_calls.append(object()),
    )
    args = argparse.Namespace(
        protocol=protocol_path,
        activations=activations,
        adapter=NEMOTRON_ADAPTER.name,
        output=tmp_path / "probe.npz",
    )

    dataset.provenance.layer_indices = (13, 20)
    with pytest.raises(ValueError, match="layers"):
        manage._train(args)
    dataset.provenance.layer_indices = (13, 20, 26)
    dataset.provenance.source_sha256 = "d" * 64
    with pytest.raises(ValueError, match="sources"):
        manage._train(args)
    assert fit_calls == []


def test_adopt_standardized_validates_saved_provenance_and_reports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args, reference, report = _adoption_fixture(tmp_path)
    adopted = SimpleNamespace(
        training=SimpleNamespace(qualification_policy=manage.SEGMENT_MEAN_REASONING_PROBABILITY),
        selected_layer_index=13,
        regularization_lambda=100.0,
        neutral_valid=False,
        qa_eligible=True,
    )
    captured: dict[str, Any] = {}
    monkeypatch.setattr(manage, "load_role_probe", lambda _path: reference)
    monkeypatch.setattr(manage, "load_activation_dataset", lambda _path: object())
    monkeypatch.setattr(
        manage,
        "adopt_saved_probe_parameters",
        lambda *values: captured.update(values=values) or adopted,
    )
    monkeypatch.setattr(
        manage, "save_role_probe", lambda value, path: captured.update(saved=(value, path))
    )

    manage._adopt_standardized(args)

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "fit_performed": False,
        "neutral_token_diagnostics_meet_historical_gates": False,
        "output": str(args.output),
        "qa_eligible": True,
        "qualification_policy": manage.SEGMENT_MEAN_REASONING_PROBABILITY,
        "selected_lambda": 100.0,
        "selected_layer": 13,
    }
    assert captured["saved"] == (adopted, args.output)
    adoption = captured["values"][-1]
    assert adoption.method == manage.SAVED_STANDARDIZED_PARAMETERS
    assert adoption.fit_document_count == 3
    assert report["successful_candidates"] == 8


def test_adopt_standardized_rejects_malformed_parameter_archive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args, reference, _ = _adoption_fixture(tmp_path)
    with args.parameters.open("wb") as handle:
        np.savez(handle, raw_coefficients=np.zeros((5, 3)))
    report = manage._json_object(args.diagnostic)
    report["candidates"][6]["diagnostic_parameters_sha256"] = manage._sha256(args.parameters)
    args.diagnostic.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(manage, "load_role_probe", lambda _path: reference)
    with pytest.raises(ValueError, match="invalid saved standardized parameter archive"):
        manage._adopt_standardized(args)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("report", "saved standardized diagnostic provenance"),
        ("selection", "DEV-selected candidate"),
        ("parameter-hash", "saved standardized parameter hashes"),
        ("scale", "saved standardized parameters have invalid arrays"),
        ("standardization", "raw-space parameters do not match"),
        ("protocol", "frozen native gates"),
    ],
)
def test_adopt_standardized_rejects_integrity_tampering_before_activation_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    args, reference, _ = _adoption_fixture(tmp_path)
    report = manage._json_object(args.diagnostic)
    if mutation == "report":
        report["refit_performed"] = True
        args.diagnostic.write_text(json.dumps(report), encoding="utf-8")
    elif mutation == "selection":
        report["candidates"][6]["diagnostic_parameters"] = "other.npz"
        args.diagnostic.write_text(json.dumps(report), encoding="utf-8")
    elif mutation == "parameter-hash":
        report["candidates"][6]["raw_coefficients_sha256"] = "0" * 64
        args.diagnostic.write_text(json.dumps(report), encoding="utf-8")
    elif mutation in {"scale", "standardization"}:
        with np.load(args.parameters, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        if mutation == "scale":
            arrays["scale"] = arrays["scale"].copy()
            arrays["scale"][0] = -1.0
        else:
            arrays["standardized_bias"] = arrays["standardized_bias"].copy()
            arrays["standardized_bias"][0] += 1.0
        with args.parameters.open("wb") as handle:
            np.savez(handle, **arrays)
        report["candidates"][6]["diagnostic_parameters_sha256"] = manage._sha256(
            args.parameters
        )
        args.diagnostic.write_text(json.dumps(report), encoding="utf-8")
    else:
        protocol = manage._json_object(args.protocol)
        protocol["native_gate"]["required_segments"] = ["reasoning"]
        args.protocol.write_text(json.dumps(protocol), encoding="utf-8")

    monkeypatch.setattr(manage, "load_role_probe", lambda _path: reference)
    monkeypatch.setattr(
        manage,
        "load_activation_dataset",
        lambda _path: pytest.fail("activation dataset loaded after integrity rejection"),
    )
    with pytest.raises(ValueError, match=message):
        manage._adopt_standardized(args)


def test_qualify_binds_partitions_and_reports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    partitions = tmp_path / "partitions.json"
    partitions.write_text(
        json.dumps(
            [
                {"source_id": "cal-a", "split": "calibration"},
                {"source_id": "test-a", "split": "test"},
            ]
        ),
        encoding="utf-8",
    )
    protocol = _expanded_protocol()
    protocol["native_prompt_partitions_sha256"] = manage._sha256(partitions)
    protocol_path = tmp_path / "expanded-protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    config = manage._validate_frozen_protocol(
        protocol,
        NEMOTRON_ADAPTER.name,
        protocol_sha256=manage._sha256(protocol_path),
    )
    probe = SimpleNamespace(
        provenance=SimpleNamespace(native_template_adapter=NEMOTRON_ADAPTER.name),
        training=config,
        qualification=None,
    )
    calibration = SimpleNamespace(document_ids=np.asarray(["cal-a"]))
    test = SimpleNamespace(document_ids=np.asarray(["test-a"]))
    qualification = SimpleNamespace(
        threshold_reasoning_sensitivity=0.9,
        threshold_final_specificity=1.0,
        passed=True,
        segment_passed=True,
        calibration=SimpleNamespace(usable=True, threshold=0.7),
        test=SimpleNamespace(
            passed=True,
            bootstrap_auc=SimpleNamespace(auc=0.9, lower_95=0.8),
        ),
    )
    qualified = SimpleNamespace(qa_eligible=True, qualification=qualification)
    monkeypatch.setattr(manage, "load_role_probe", lambda _path: probe)
    monkeypatch.setattr(
        manage,
        "load_native_activation_dataset",
        lambda path: calibration if path.name == "calibration" else test,
    )
    monkeypatch.setattr(manage, "qualify_role_probe", lambda *_args, **_kwargs: qualified)
    saved: dict[str, Any] = {}
    monkeypatch.setattr(
        manage, "save_role_probe", lambda value, path: saved.update(value=value, path=path)
    )
    args = argparse.Namespace(
        protocol=protocol_path,
        probe=tmp_path / "probe",
        calibration=tmp_path / "calibration",
        test=tmp_path / "test",
        prompt_partitions=partitions,
        output=tmp_path / "qualified",
    )
    manage._qualify(args)
    report = json.loads(capsys.readouterr().out)
    assert report["native_test_passed"] is True
    assert report["native_test_auc_lower_95"] == 0.8
    assert saved == {"value": qualified, "path": args.output}

    probe.qualification = qualification
    with pytest.raises(ValueError, match="already qualified"):
        manage._qualify(args)
    probe.qualification = None
    probe.training = SimpleNamespace()
    with pytest.raises(ValueError, match="training configuration"):
        manage._qualify(args)
    probe.training = config
    calibration.document_ids = np.asarray(["wrong"])
    with pytest.raises(ValueError, match="prompt partitions"):
        manage._qualify(args)

    diagnostic_path = _write_protocol(tmp_path)
    probe.training = manage._validate_frozen_protocol(
        _protocol(),
        NEMOTRON_ADAPTER.name,
        protocol_sha256=manage._sha256(diagnostic_path),
    )
    args.protocol = diagnostic_path
    with pytest.raises(ValueError, match="diagnostic probes cannot be qualified"):
        manage._qualify(args)


def test_extract_native_wires_exact_checkpoint_and_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    transformers = pytest.importorskip("transformers")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    manifest = {
        "adapter": NEMOTRON_ADAPTER.name,
        "model_id": NEMOTRON_ADAPTER.model_id,
        "revision": NEMOTRON_ADAPTER.model_revision,
        "max_layer": 26,
        "weights_sha256": "a" * 64,
        "weights_hash_kind": "test",
    }
    (checkpoint / "prefix-checkpoint-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    protocol = tmp_path / "expanded-protocol.json"
    protocol.write_text(json.dumps(_expanded_protocol()), encoding="utf-8")
    partitions = tmp_path / "partitions.json"
    partitions.write_text("[]", encoding="utf-8")
    dialogue_dir = tmp_path / "dialogues"
    dialogue_dir.mkdir()
    (dialogue_dir / "one.json").write_text("{}", encoding="utf-8")
    tokenizer = object()
    model = object()
    dialogues = (object(),)
    dataset = object()
    calls: dict[str, Any] = {}
    events: list[str] = []
    runtime = {"runtime": "test"}
    runtime_sha256 = manage.hashlib.sha256(
        json.dumps(runtime, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: events.append("tokenizer") or tokenizer,
    )
    monkeypatch.setattr(manage, "validate_prefix_checkpoint_identity", lambda *args: None)
    monkeypatch.setattr(
        manage, "load_prefix_model", lambda *args, **kwargs: events.append("model") or model
    )
    monkeypatch.setattr(
        manage, "model_runtime_identity", lambda *args, **kwargs: (runtime_sha256, runtime)
    )
    monkeypatch.setattr(
        manage,
        "load_native_dialogues",
        lambda *args, **kwargs: events.append("dialogues") or dialogues,
    )
    monkeypatch.setattr(
        manage,
        "extract_native_activations",
        lambda *args, **kwargs: calls.update(args=args, kwargs=kwargs) or dataset,
    )
    monkeypatch.setattr(
        manage,
        "save_native_activation_dataset",
        lambda value, path: calls.update(saved=(value, path)),
    )
    args = argparse.Namespace(
        adapter=NEMOTRON_ADAPTER.name,
        protocol=protocol,
        checkpoint=checkpoint,
        execution_device="cuda:0",
        dialogue_dir=dialogue_dir,
        dialogue_glob="*.json",
        split="calibration",
        prompt_partitions=partitions,
        output=tmp_path / "native",
    )
    manage._extract_native(args)
    assert capsys.readouterr().out.strip() == str(args.output)
    assert events[:3] == ["dialogues", "tokenizer", "model"]
    assert calls["args"][:4] == (model, tokenizer, NEMOTRON_ADAPTER, dialogues)
    assert calls["kwargs"]["layers"] == (13, 20, 26)
    assert calls["kwargs"]["identity"].runtime_sha256 == runtime_sha256
    assert calls["saved"] == (dataset, args.output)

    manifest["model_id"] = "wrong"
    (checkpoint / "prefix-checkpoint-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="model_id"):
        manage._extract_native(args)


def test_extract_native_validates_dialogues_before_loading_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "prefix-checkpoint-manifest.json").write_text(
        json.dumps(
            {
                "adapter": NEMOTRON_ADAPTER.name,
                "model_id": NEMOTRON_ADAPTER.model_id,
                "revision": NEMOTRON_ADAPTER.model_revision,
                "max_layer": 26,
                "weights_sha256": "a" * 64,
                "weights_hash_kind": "test",
            }
        ),
        encoding="utf-8",
    )
    protocol = tmp_path / "expanded-protocol.json"
    protocol.write_text(json.dumps(_expanded_protocol()), encoding="utf-8")
    partitions = tmp_path / "partitions.json"
    partitions.write_text("[]", encoding="utf-8")
    dialogues = tmp_path / "dialogues"
    dialogues.mkdir()
    args = argparse.Namespace(
        adapter=NEMOTRON_ADAPTER.name,
        protocol=protocol,
        checkpoint=checkpoint,
        execution_device="cuda:0",
        dialogue_dir=dialogues,
        dialogue_glob="*.json",
        split="calibration",
        prompt_partitions=partitions,
        output=tmp_path / "native",
    )
    events: list[str] = []
    monkeypatch.setattr(manage, "validate_prefix_checkpoint_identity", lambda *args: None)

    def reject_dialogues(*args: object, **kwargs: object) -> tuple[object, ...]:
        del args, kwargs
        events.append("dialogues")
        raise ValueError("invalid native dialogue")

    monkeypatch.setattr(manage, "load_native_dialogues", reject_dialogues)
    monkeypatch.setattr(
        manage, "load_prefix_model", lambda *args, **kwargs: pytest.fail("model was loaded")
    )
    with pytest.raises(ValueError, match="invalid native dialogue"):
        manage._extract_native(args)
    assert events == ["dialogues"]


def test_main_dispatches_selected_command(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[argparse.Namespace] = []
    namespace = argparse.Namespace(run=called.append)
    parser = SimpleNamespace(parse_args=lambda _argv: namespace)
    monkeypatch.setattr(manage, "build_parser", lambda: parser)
    manage.main(["train"])
    assert called == [namespace]
