"""Exercise the frozen probe workflow through its public command interface."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import reasonese.manage_role_probe as cli
from reasonese.native_probe_activations import save_native_activation_dataset
from reasonese.role_probe_extraction import NEMOTRON_ADAPTER, ProbeRole
from reasonese.role_probes import load_role_probe
from tests.test_role_probes import _dataset


def _protocol() -> dict[str, Any]:
    return {
        "models": {"nemotron": {"revision": NEMOTRON_ADAPTER.model_revision, "layer_index": 26}},
        "neutral_split": {"train": 0.6, "development": 0.2, "test": 0.2, "seed": 0},
        "neutral_gate": {
            "token_accuracy": 0.9,
            "per_role_token_accuracy": 0.85,
            "document_macro_token_accuracy": 0.9,
            "per_role_document_macro_token_accuracy": 0.85,
        },
        "neutral_validation_documents": 60,
        "stored_activation_dtype": "float32 (exact promotion of BF16 hook outputs; avoid FP16 range loss)",
        "roles": [str(role) for role in ProbeRole],
        "native_dialogues_per_model": 24,
        "native_split": {"calibration": 12, "test": 12},
        "native_gate": {
            "required_roles": ["reasoning", "assistant"],
            "minimum_role_accuracy": 0.75,
            "minimum_document_macro_accuracy": 0.75,
            "reasoning_vs_final_segment_auc": 0.85,
            "auc_bootstrap_lower_95_bound_must_exceed": 0.5,
        },
    }


def _neutral():
    data = _dataset(document_count=60, roles=tuple(str(role) for role in ProbeRole), layers=(26,))
    return replace(
        data,
        provenance=replace(
            data.provenance,
            native_template_adapter=NEMOTRON_ADAPTER.name,
            model_id=NEMOTRON_ADAPTER.model_id,
            model_revision=NEMOTRON_ADAPTER.model_revision,
        ),
    )


def test_cli_trains_then_qualifies_without_learning_from_native_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    neutral = _neutral()
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(_protocol()))
    monkeypatch.setattr(cli, "load_activation_dataset", lambda _: neutral)
    unqualified = tmp_path / "neutral.npz"
    cli.main(
        [
            "train",
            "--adapter",
            NEMOTRON_ADAPTER.name,
            "--activations",
            "synthetic",
            "--protocol",
            str(protocol),
            "--output",
            str(unqualified),
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert report["split"] == {"train": 36, "validation": 12, "test": 12}
    assert report["neutral_valid"] and not report["qa_eligible"]
    trained = load_role_probe(unqualified)
    partitions = []
    for split, offset in (("calibration", 5000), ("test", 10000)):
        native = _dataset(
            kind="untouched-native-conversations",
            document_count=12,
            layers=(26,),
            document_prefix=split,
            content_token_offset=offset,
        )
        native_vectors = np.zeros_like(native.activations)
        native_vectors[native.roles == "reasoning", :, 3] = 3.0
        native = replace(
            native,
            activations=native_vectors,
            provenance=replace(
                trained.provenance,
                dataset_kind=native.provenance.dataset_kind,
                source_name=split,
                source_sha256=hashlib.sha256(split.encode()).hexdigest(),
                filler_pool_kind="none",
                filler_source_sha256=None,
                filler_documents=0,
            ),
        )
        save_native_activation_dataset(native, tmp_path / split)
        for index, document_id in enumerate(sorted(set(native.document_ids.tolist()))):
            partitions.append(
                {
                    "index": len(partitions),
                    "source_id": document_id,
                    "split": split,
                    "normalized_prompt_sha256": hashlib.sha256(
                        f"{split}-{index}".encode()
                    ).hexdigest(),
                }
            )
    partition_path = tmp_path / "partitions.json"
    partition_path.write_text(json.dumps(partitions))
    qualified_path = tmp_path / "qualified.npz"
    args = [
        "qualify",
        "--probe",
        str(unqualified),
        "--calibration",
        str(tmp_path / "calibration"),
        "--test",
        str(tmp_path / "test"),
        "--prompt-partitions",
        str(partition_path),
        "--protocol",
        str(protocol),
        "--output",
        str(qualified_path),
    ]
    cli.main(args)
    result = load_role_probe(qualified_path)
    assert json.loads(capsys.readouterr().out)["qa_eligible"]
    assert result.qa_eligible
    np.testing.assert_array_equal(trained.coefficients, result.coefficients)
    np.testing.assert_array_equal(trained.intercepts, result.intercepts)
    assert result.split == trained.split
    assert result.qualification is not None
    assert result.qualification.test.bootstrap_auc.auc == 1.0
    assert load_role_probe(unqualified).qualification is None

    with pytest.raises(FileExistsError):
        cli.main(args)
    args[-1] = str(tmp_path / "fresh.npz")
    args[2] = str(qualified_path)
    with pytest.raises(ValueError, match="already qualified"):
        cli.main(args)
    args[2] = str(unqualified)
    partitions[0]["source_id"] = "substituted-conversation"
    partition_path.write_text(json.dumps(partitions))
    with pytest.raises(ValueError, match="do not match the prompt partitions"):
        cli.main(args)


def test_extract_native_checks_protocol_layer_and_committed_manifest_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(_protocol()))
    (tmp_path / "prefix-checkpoint-manifest.json").write_text(
        json.dumps(
            {
                "adapter": NEMOTRON_ADAPTER.name,
                "model_id": NEMOTRON_ADAPTER.model_id,
                "revision": NEMOTRON_ADAPTER.model_revision,
                "max_layer": 26,
            }
        )
    )
    args = [
        "extract-native",
        "--adapter",
        NEMOTRON_ADAPTER.name,
        "--checkpoint",
        str(tmp_path),
        "--dialogue-dir",
        str(tmp_path),
        "--dialogue-glob",
        "native-*.json",
        "--prompt-partitions",
        str(tmp_path / "partitions.json"),
        "--protocol",
        str(protocol),
        "--split",
        "calibration",
        "--layer",
        "26",
        "--output",
        str(tmp_path / "out"),
    ]

    class ReachedWeightValidation(Exception):
        pass

    def validated_manifest(*_: object) -> None:
        raise ReachedWeightValidation

    monkeypatch.setattr(cli, "validate_prefix_checkpoint_identity", validated_manifest)
    with pytest.raises(ReachedWeightValidation):
        cli.main(args)
    args[args.index("--layer") + 1] = "25"
    with pytest.raises(ValueError, match="frozen protocol"):
        cli.main(args)
