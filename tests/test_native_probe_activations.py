"""Untouched native-dialogue extraction and artifact tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import reasonese.native_probe_activations as native_module
from reasonese.axes import Assistant
from reasonese.native_probe_activations import (
    NativeDialogue,
    extract_native_activations,
    load_native_activation_dataset,
    load_native_dialogues,
    save_native_activation_dataset,
)
from reasonese.role_probe_extraction import ExtractionIdentity
from reasonese.role_probes import activation_dataset_fingerprint
from tests.test_probe_rendering import _tokenizer_and_adapter


def _dialogues(split: str = "calibration") -> tuple[NativeDialogue, ...]:
    return tuple(
        NativeDialogue(
            f"dialogue-{split}-{index:02d}",
            split,
            f"Question number {index}?",
            f"I will reason about question {index}.",
            f"The final answer is {index}.",
            f"native-{index:02d}.json",
            f"{index + 1:064x}",
            "nvidia/nemotron-3.5-lightning:free",
            "Nvidia",
            "nvidia/nemotron-3.5-lightning:free",
            "OpenAssistant/oasst1",
            "revision",
            f"{index + 100:064x}",
        )
        for index in range(12)
    )


def _identity() -> ExtractionIdentity:
    runtime = {"runtime": "test"}
    runtime_sha = hashlib.sha256(
        json.dumps(runtime, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    return ExtractionIdentity(
        weights_sha256="a" * 64,
        weights_hash_kind="prefix-checkpoint-files-sha256-v1",
        tokenizer_id="test/tokenizer",
        tokenizer_revision="tokenizer-revision",
        source_name="native-json",
        model_dtype="bfloat16",
        transformers_version="test",
        torch_version="test",
        runtime_sha256=runtime_sha,
        runtime=runtime,
    )


def _write_dialogue_sources(tmp_path: Path) -> tuple[tuple[Path, ...], Path]:
    partitions = []
    paths = []
    for index in range(24):
        split = "calibration" if index % 2 == 0 else "test"
        source_id = f"oasst-{index:02d}"
        text = f"Human prompt {index}"
        partitions.append(
            {
                "index": index,
                "source_id": source_id,
                "normalized_prompt_sha256": f"{index + 1:064x}",
                "split": split,
            }
        )
        path = tmp_path / f"native-{index:02d}.json"
        path.write_text(
            json.dumps(
                {
                    "assistant": str(Assistant.NEMOTRON_3_5_LIGHTNING),
                    "route": "nvidia/nemotron-3.5-lightning:free",
                    "prompt": {
                        "id": source_id,
                        "split": split,
                        "text": text,
                        "source": "OpenAssistant/oasst1",
                        "revision": "revision",
                    },
                    "request": {
                        "messages": [{"role": "user", "content": text}],
                        "temperature": 0.7,
                        "reasoning": {"enabled": True, "exclude": False},
                    },
                    "response": {
                        "model": "nvidia/nemotron-3.5-lightning:free",
                        "provider": "Nvidia",
                        "choices": [
                            {
                                "message": {
                                    "reasoning": f"Reasoning {index}\n",
                                    "content": f"Final {index}",
                                }
                            }
                        ],
                    },
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)
    partition_path = tmp_path / "partitions.json"
    partition_path.write_text(json.dumps(partitions), encoding="utf-8")
    return tuple(paths), partition_path


def test_native_dialogue_loader_selects_one_frozen_partition(tmp_path: Path) -> None:
    paths, partitions = _write_dialogue_sources(tmp_path)
    dialogues = load_native_dialogues(
        paths,
        assistant=Assistant.NEMOTRON_3_5_LIGHTNING,
        split="calibration",
        prompt_partitions=partitions,
    )
    assert len(dialogues) == 12
    assert {row.split for row in dialogues} == {"calibration"}
    assert {row.document_id for row in dialogues} == {
        f"oasst-{index:02d}" for index in range(0, 24, 2)
    }
    assert all(row.reasoning.endswith("\n") for row in dialogues)


def test_native_dialogue_loader_rejects_changed_request(tmp_path: Path) -> None:
    paths, partitions = _write_dialogue_sources(tmp_path)
    changed = json.loads(paths[0].read_text(encoding="utf-8"))
    changed["request"]["temperature"] = 0
    paths[0].write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="request does not match frozen protocol"):
        load_native_dialogues(
            paths,
            assistant=Assistant.NEMOTRON_3_5_LIGHTNING,
            split="calibration",
            prompt_partitions=partitions,
        )


def test_native_extraction_replays_both_segments_and_round_trips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer, adapter = _tokenizer_and_adapter()

    def capture(
        model: object,
        adapter: object,
        *,
        input_ids: tuple[int, ...],
        token_positions: tuple[int, ...],
        layers: tuple[int, ...],
    ) -> np.ndarray:
        rows = len(token_positions)
        values = np.zeros((rows, len(layers), 5), dtype=np.float32)
        values[:, :, 0] = np.asarray(token_positions)[:, None]
        return values

    monkeypatch.setattr(native_module, "capture_token_activations", capture)
    dataset = extract_native_activations(
        object(),
        tokenizer,
        adapter,
        _dialogues(),
        layers=(3, 7),
        identity=_identity(),
    )
    assert set(dataset.roles) == {"reasoning", "assistant"}
    assert len(set(dataset.document_ids)) == 12
    assert dataset.provenance.activation_dtype == "float32"
    assert dataset.provenance.masked_control_tokens > 0

    output = tmp_path / "native-activations"
    save_native_activation_dataset(dataset, output)
    loaded = load_native_activation_dataset(output)
    assert activation_dataset_fingerprint(loaded) == activation_dataset_fingerprint(dataset)


def test_native_artifact_checksum_rejects_mutated_array(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer, adapter = _tokenizer_and_adapter()
    monkeypatch.setattr(
        native_module,
        "capture_token_activations",
        lambda model, adapter, *, input_ids, token_positions, layers: np.zeros(
            (len(token_positions), len(layers), 5), dtype=np.float32
        ),
    )
    dataset = extract_native_activations(
        object(), tokenizer, adapter, _dialogues(), layers=(7,), identity=_identity()
    )
    output = tmp_path / "native-activations"
    save_native_activation_dataset(dataset, output)
    with (output / "role.npy").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_native_activation_dataset(output)
