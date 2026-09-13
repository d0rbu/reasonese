"""Train, qualify, and extract native validation data for activation role probes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from reasonese.axes import Assistant
from reasonese.native_probe_activations import (
    extract_native_activations,
    load_native_activation_dataset,
    load_native_dialogues,
    save_native_activation_dataset,
)
from reasonese.probe_statistics import (
    MIN_AUC_BOOTSTRAP_LOWER,
    MIN_DOCUMENT_MACRO_ACCURACY,
    MIN_ROLE_ACCURACY,
    MIN_TEST_AUC,
)
from reasonese.role_probe_extraction import (
    GEMMA_ADAPTER,
    NATIVE_ADAPTERS,
    NEMOTRON_ADAPTER,
    ExtractionIdentity,
    ProbeRole,
    load_activation_dataset,
    load_prefix_model,
    model_runtime_identity,
    validate_prefix_checkpoint_identity,
)
from reasonese.role_probes import (
    ProbeTrainingConfig,
    load_role_probe,
    qualify_role_probe,
    save_role_probe,
    train_role_probe,
)

_ADAPTER_ASSISTANTS = {
    NEMOTRON_ADAPTER.name: Assistant.NEMOTRON_3_5_LIGHTNING,
    GEMMA_ADAPTER.name: Assistant.GEMMA_4_31B_IT,
}
_ADAPTER_PROTOCOL_KEYS = {
    NEMOTRON_ADAPTER.name: "nemotron",
    GEMMA_ADAPTER.name: "gemma",
}
_ADAPTER_PROTOCOL_LAYERS = {
    NEMOTRON_ADAPTER.name: 26,
    GEMMA_ADAPTER.name: 30,
}
_FROZEN_ACTIVATION_DTYPE = "float32 (exact promotion of BF16 hook outputs; avoid FP16 range loss)"


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"invalid JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _partition_document_ids(path: Path, split: str) -> set[str]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("native prompt partitions must be a list")
    try:
        return {
            record["source_id"]
            for record in records
            if isinstance(record, dict) and record["split"] == split
        }
    except (KeyError, TypeError) as error:
        raise ValueError("invalid native prompt partition record") from error


def _protocol_training_config(protocol: dict[str, Any], adapter_name: str) -> ProbeTrainingConfig:
    try:
        model = protocol["models"][_ADAPTER_PROTOCOL_KEYS[adapter_name]]
        split = protocol["neutral_split"]
        gates = protocol["neutral_gate"]
        documents = protocol["neutral_validation_documents"]
        precision = protocol["stored_activation_dtype"]
    except (KeyError, TypeError) as error:
        raise ValueError("probe protocol lacks required training fields") from error
    if (
        not isinstance(model, dict)
        or not isinstance(split, dict)
        or not isinstance(gates, dict)
        or documents != 60
        or precision != _FROZEN_ACTIVATION_DTYPE
    ):
        raise ValueError("probe protocol must retain the frozen 60-document float32 design")
    adapter = NATIVE_ADAPTERS[adapter_name]
    if (
        model.get("revision") != adapter.model_revision
        or model.get("layer_index") != _ADAPTER_PROTOCOL_LAYERS[adapter_name]
    ):
        raise ValueError("probe protocol does not match the pinned native adapter")
    if split != {"train": 0.6, "development": 0.2, "test": 0.2, "seed": 0}:
        raise ValueError("probe protocol does not match the frozen neutral split")
    if gates != {
        "token_accuracy": 0.9,
        "per_role_token_accuracy": 0.85,
        "document_macro_token_accuracy": 0.9,
        "per_role_document_macro_token_accuracy": 0.85,
    }:
        raise ValueError("probe protocol does not match the frozen neutral gates")
    return ProbeTrainingConfig(
        layer_index=model["layer_index"],
        minimum_neutral_accuracy=gates["token_accuracy"],
        minimum_neutral_per_role_accuracy=gates["per_role_token_accuracy"],
        minimum_neutral_document_accuracy=gates["document_macro_token_accuracy"],
        minimum_neutral_per_role_document_accuracy=gates["per_role_document_macro_token_accuracy"],
        train_fraction=split["train"],
        validation_fraction=split["development"],
        seed=split["seed"],
    )


def _validate_frozen_protocol(protocol: dict[str, Any], adapter_name: str) -> ProbeTrainingConfig:
    config = _protocol_training_config(protocol, adapter_name)
    if set(protocol.get("roles", ())) != {str(role) for role in ProbeRole}:
        raise ValueError("probe protocol does not contain the five frozen native roles")
    if protocol.get("native_dialogues_per_model") != 24 or protocol.get("native_split") != {
        "calibration": 12,
        "test": 12,
    }:
        raise ValueError("probe protocol must retain the frozen 12/12 native split")
    if protocol.get("native_gate") != {
        "required_roles": ["reasoning", "assistant"],
        "minimum_role_accuracy": MIN_ROLE_ACCURACY,
        "minimum_document_macro_accuracy": MIN_DOCUMENT_MACRO_ACCURACY,
        "reasoning_vs_final_segment_auc": MIN_TEST_AUC,
        "auc_bootstrap_lower_95_bound_must_exceed": MIN_AUC_BOOTSTRAP_LOWER,
    }:
        raise ValueError("probe protocol does not match the frozen native gates")
    return config


def _train(args: argparse.Namespace) -> None:
    protocol = _json_object(args.protocol)
    dataset = load_activation_dataset(args.activations)
    config = _validate_frozen_protocol(protocol, args.adapter)
    if len(set(dataset.document_ids.tolist())) != 60:
        raise ValueError("probe training requires the frozen 60 neutral documents")
    if dataset.provenance.activation_dtype != "float32":
        raise ValueError("probe training requires float32 activation storage")
    adapter = NATIVE_ADAPTERS[args.adapter]
    if (
        dataset.provenance.native_template_adapter != adapter.name
        or dataset.provenance.model_id != adapter.model_id
        or dataset.provenance.model_revision != adapter.model_revision
    ):
        raise ValueError("neutral activations do not match the selected native adapter")
    probe = train_role_probe(dataset, config)
    save_role_probe(probe, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "neutral_valid": probe.neutral_valid,
                "qa_eligible": probe.qa_eligible,
                "selected_lambda": probe.regularization_lambda,
                "split": {
                    "train": len(probe.split.train),
                    "validation": len(probe.split.validation),
                    "test": len(probe.split.test),
                },
            },
            sort_keys=True,
        )
    )


def _qualify(args: argparse.Namespace) -> None:
    protocol = _json_object(args.protocol)
    probe = load_role_probe(args.probe)
    frozen_config = _validate_frozen_protocol(protocol, probe.provenance.native_template_adapter)
    if probe.training != frozen_config:
        raise ValueError("probe training configuration does not match the frozen protocol")
    if probe.qualification is not None:
        raise ValueError("input probe is already qualified")
    calibration = load_native_activation_dataset(args.calibration)
    test = load_native_activation_dataset(args.test)
    if set(calibration.document_ids.tolist()) != _partition_document_ids(
        args.prompt_partitions, "calibration"
    ) or set(test.document_ids.tolist()) != _partition_document_ids(args.prompt_partitions, "test"):
        raise ValueError("native activation documents do not match the prompt partitions")
    qualified = qualify_role_probe(
        probe,
        calibration,
        test,
        prompt_partition_name=args.prompt_partitions.name,
        prompt_partition_sha256=_sha256(args.prompt_partitions),
    )
    save_role_probe(qualified, args.output)
    assert qualified.qualification is not None
    print(
        json.dumps(
            {
                "output": str(args.output),
                "qa_eligible": qualified.qa_eligible,
                "calibration_usable": qualified.qualification.calibration.usable,
                "calibration_threshold": qualified.qualification.calibration.threshold,
                "native_test_passed": qualified.qualification.test.passed,
                "native_test_auc": qualified.qualification.test.bootstrap_auc.auc,
                "native_test_auc_lower_95": qualified.qualification.test.bootstrap_auc.lower_95,
            },
            sort_keys=True,
        )
    )


def _extract_native(args: argparse.Namespace) -> None:
    adapter = NATIVE_ADAPTERS[args.adapter]
    frozen_config = _validate_frozen_protocol(_json_object(args.protocol), adapter.name)
    if args.layer != frozen_config.layer_index:
        raise ValueError("native extraction layer does not match the frozen protocol")
    assistant = _ADAPTER_ASSISTANTS[adapter.name]
    checkpoint_manifest = _json_object(args.checkpoint / "prefix-checkpoint-manifest.json")
    for key, expected in {
        "adapter": adapter.name,
        "model_id": adapter.model_id,
        "model_revision": adapter.model_revision,
        "max_layer": args.layer,
    }.items():
        if checkpoint_manifest.get(key) != expected:
            raise ValueError(f"prefix checkpoint {key} does not match the requested extraction")
    validate_prefix_checkpoint_identity(args.checkpoint, checkpoint_manifest)
    try:
        import torch  # ty: ignore[unresolved-import, unused-ignore-comment]
        import transformers  # ty: ignore[unresolved-import, unused-ignore-comment]
        from transformers import AutoTokenizer  # ty: ignore[unresolved-import, unused-ignore-comment]
    except ImportError as error:  # pragma: no cover - minimal installations lack probe extras
        raise RuntimeError("native activation extraction requires the 'probes' extra") from error
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, use_fast=True)
    model = load_prefix_model(
        args.checkpoint,
        adapter,
        max_layer=args.layer,
        execution_device=args.execution_device,
    )
    runtime_sha256, runtime = model_runtime_identity(model, adapter, checkpoint=args.checkpoint)
    dialogues = load_native_dialogues(
        tuple(sorted(args.dialogue_dir.glob(args.dialogue_glob))),
        assistant=assistant,
        split=args.split,
        prompt_partitions=args.prompt_partitions,
    )
    dataset = extract_native_activations(
        model,
        tokenizer,
        adapter,
        dialogues,
        layers=(args.layer,),
        identity=ExtractionIdentity(
            weights_sha256=checkpoint_manifest["weights_sha256"],
            weights_hash_kind=checkpoint_manifest["weights_hash_kind"],
            tokenizer_id=adapter.model_id,
            tokenizer_revision=adapter.model_revision,
            source_name=args.dialogue_glob,
            model_dtype="bfloat16",
            transformers_version=transformers.__version__,
            torch_version=torch.__version__,
            runtime_sha256=runtime_sha256,
            runtime=runtime,
        ),
    )
    save_native_activation_dataset(dataset, args.output)
    print(args.output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reasonese-role-probe")
    subcommands = parser.add_subparsers(dest="command", required=True)

    train = subcommands.add_parser("train", help="fit on a frozen neutral activation artifact")
    train.add_argument("--adapter", choices=sorted(NATIVE_ADAPTERS), required=True)
    train.add_argument("--activations", type=Path, required=True)
    train.add_argument("--protocol", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.set_defaults(run=_train)

    qualify = subcommands.add_parser("qualify", help="attach frozen native validation evidence")
    qualify.add_argument("--probe", type=Path, required=True)
    qualify.add_argument("--calibration", type=Path, required=True)
    qualify.add_argument("--test", type=Path, required=True)
    qualify.add_argument("--prompt-partitions", type=Path, required=True)
    qualify.add_argument("--protocol", type=Path, required=True)
    qualify.add_argument("--output", type=Path, required=True)
    qualify.set_defaults(run=_qualify)

    extract = subcommands.add_parser(
        "extract-native", help="replay one frozen native dialogue partition"
    )
    extract.add_argument("--adapter", choices=sorted(NATIVE_ADAPTERS), required=True)
    extract.add_argument("--checkpoint", type=Path, required=True)
    extract.add_argument("--dialogue-dir", type=Path, required=True)
    extract.add_argument("--dialogue-glob", required=True)
    extract.add_argument("--prompt-partitions", type=Path, required=True)
    extract.add_argument("--protocol", type=Path, required=True)
    extract.add_argument("--split", choices=("calibration", "test"), required=True)
    extract.add_argument("--layer", type=int, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--execution-device", default="cuda:0")
    extract.set_defaults(run=_extract_native)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.run(args)


if __name__ == "__main__":  # pragma: no cover
    main()
