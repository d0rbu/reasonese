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


def _expanded_protocol() -> dict[str, Any]:
    protocol = _protocol()
    protocol["neutral_validation_documents"] = 250
    protocol["neutral_max_content_tokens"] = 1_024
    protocol["native_prompt_partitions_sha256"] = "a" * 64
    protocol["neutral_target_source_sha256"] = "b" * 64
    protocol["neutral_filler_source_sha256"] = "c" * 64
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


def test_frozen_protocol_accepts_both_pinned_adapters_and_rejects_drift() -> None:
    for adapter in (NEMOTRON_ADAPTER, manage.GEMMA_ADAPTER):
        config = manage._validate_frozen_protocol(_protocol(), adapter.name)
        assert config.layer_indices == (manage._ADAPTER_LEGACY_LAYER[adapter.name],)
        assert config.minimum_neutral_accuracy == 0.9
        assert config.max_iterations == 2_000
        assert config.tolerance == 1e-4

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
    runtime = {"runtime": "test"}
    runtime_sha256 = manage.hashlib.sha256(
        json.dumps(runtime, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer
    )
    monkeypatch.setattr(manage, "validate_prefix_checkpoint_identity", lambda *args: None)
    monkeypatch.setattr(manage, "load_prefix_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(
        manage, "model_runtime_identity", lambda *args, **kwargs: (runtime_sha256, runtime)
    )
    monkeypatch.setattr(manage, "load_native_dialogues", lambda *args, **kwargs: dialogues)
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


def test_main_dispatches_selected_command(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[argparse.Namespace] = []
    namespace = argparse.Namespace(run=called.append)
    parser = SimpleNamespace(parse_args=lambda _argv: namespace)
    monkeypatch.setattr(manage, "build_parser", lambda: parser)
    manage.main(["train"])
    assert called == [namespace]
