"""Concrete local activation-probe scorer and cache tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import reasonese.local_probe_qa as local_module
from reasonese.axes import Assistant
from reasonese.local_probe_qa import LocalProbeQaScorer
from reasonese.probe_rendering import RenderedProbeContext
from reasonese.role_probe_extraction import EXTRACTION_PROTOCOL, NEMOTRON_ADAPTER, ProbeRole
from reasonese.role_probes import load_role_probe, save_role_probe, train_role_probe
from tests.test_probe_qa import _requests
from tests.test_role_probes import _dataset, _qualify, _training_config


def _scorer_files(tmp_path: Path) -> tuple[Path, Path, str]:
    roles = tuple(str(role) for role in ProbeRole)
    probe = _qualify(
        train_role_probe(_dataset(roles=roles), _training_config()),
        calibration=_dataset(
            kind="untouched-native-conversations",
            roles=roles,
            document_count=12,
            document_prefix="calibration",
            content_token_offset=5_000,
        ),
        test=_dataset(
            kind="untouched-native-conversations",
            roles=roles,
            document_count=12,
            document_prefix="test",
            content_token_offset=10_000,
        ),
    )
    runtime_sha256 = "f" * 64
    provenance = replace(
        probe.provenance,
        model_id=NEMOTRON_ADAPTER.model_id,
        model_revision=NEMOTRON_ADAPTER.model_revision,
        tokenizer_id=NEMOTRON_ADAPTER.model_id,
        tokenizer_revision=NEMOTRON_ADAPTER.model_revision,
        chat_template_sha256=NEMOTRON_ADAPTER.chat_template_sha256,
        native_template_adapter=NEMOTRON_ADAPTER.name,
        activation_site=NEMOTRON_ADAPTER.activation_site,
        runtime_sha256=runtime_sha256,
        layer_indices=(26,),
        extraction_protocol=EXTRACTION_PROTOCOL,
    )
    probe = replace(
        probe,
        provenance=provenance,
        training=replace(probe.training, layer_index=26),
    )
    probe_path = tmp_path / "probe.npz"
    save_role_probe(probe, probe_path)
    checkpoint = tmp_path / "prefix"
    checkpoint.mkdir()
    (checkpoint / "prefix-checkpoint-manifest.json").write_text(
        json.dumps(
            {
                "adapter": NEMOTRON_ADAPTER.name,
                "model_id": NEMOTRON_ADAPTER.model_id,
                "revision": NEMOTRON_ADAPTER.model_revision,
                "max_layer": 26,
                "weights_sha256": provenance.weights_sha256,
                "weights_hash_kind": provenance.weights_hash_kind,
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "bundles.json"
    config.write_text(
        json.dumps(
            {
                "bundles": [
                    {
                        "assistant": str(Assistant.NEMOTRON_3_5_LIGHTNING),
                        "adapter": NEMOTRON_ADAPTER.name,
                        "checkpoint": "prefix",
                        "probe": "probe.npz",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return config, tmp_path / "probe-cache.json", runtime_sha256


def _patch_runtime(
    monkeypatch: pytest.MonkeyPatch, runtime_sha256: str
) -> tuple[list[object], list[tuple[int, ...]]]:
    loaded: list[object] = []
    captured: list[tuple[int, ...]] = []
    transformer_module = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: object())
    )
    torch_module = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    )
    real_import = local_module.importlib.import_module
    monkeypatch.setattr(
        local_module.importlib,
        "import_module",
        lambda name: (
            transformer_module
            if name == "transformers"
            else torch_module
            if name == "torch"
            else real_import(name)
        ),
    )
    monkeypatch.setattr(local_module, "validate_prefix_checkpoint_identity", lambda *args: None)

    def load(*args: Any, **kwargs: Any) -> object:
        model = object()
        loaded.append(model)
        return model

    monkeypatch.setattr(local_module, "load_prefix_model", load)
    monkeypatch.setattr(
        local_module,
        "model_runtime_identity",
        lambda *args, **kwargs: (runtime_sha256, {"runtime": "test"}),
    )

    def render(tokenizer: object, adapter: object, setup: object, position: int):
        input_ids = tuple(range(12))
        positions = (1, 2) if position == 1 else (7, 8)
        return RenderedProbeContext(
            input_ids,
            (positions,),
            (tuple(input_ids[index] for index in positions),),
            "a" * 64,
        )

    monkeypatch.setattr(local_module, "render_collector_probe_context", render)

    def capture(
        model: object,
        adapter: object,
        *,
        input_ids: tuple[int, ...],
        token_positions: tuple[int, ...],
        layers: tuple[int, ...],
    ) -> np.ndarray:
        captured.append(token_positions)
        values = np.zeros((len(token_positions), 1, 5), dtype=np.float32)
        values[:, 0, 0] = 0.1
        return values

    monkeypatch.setattr(local_module, "capture_token_activations", capture)
    return loaded, captured


def test_local_scorer_preflights_groups_contexts_and_reuses_exact_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, cache, runtime_sha256 = _scorer_files(tmp_path)
    loaded, captured = _patch_runtime(monkeypatch, runtime_sha256)
    _, requests = _requests()
    scorer = LocalProbeQaScorer(config, cache, execution_device="cpu")
    verdicts = scorer.check(requests)

    assert len(verdicts) == 4
    assert len(loaded) == 2
    assert captured == [(1, 2, 7, 8), (1, 2, 7, 8)]
    assert len({row.context_fingerprint for row in verdicts}) == 4
    assert all(len(row.role_probabilities) == 5 for row in verdicts)
    assert "OpenRouter" in scorer.limitations[0]
    assert not any(
        str(request.setup.content_for_input(0)) in cache.read_text() for request in requests
    )

    assert scorer.check(requests) == verdicts
    assert len(loaded) == 2
    assert len(captured) == 2


def test_local_scorer_rejects_runtime_mismatch_before_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, cache, runtime_sha256 = _scorer_files(tmp_path)
    loaded, captured = _patch_runtime(monkeypatch, runtime_sha256)
    monkeypatch.setattr(
        local_module,
        "model_runtime_identity",
        lambda *args, **kwargs: ("0" * 64, {"runtime": "changed"}),
    )
    _, requests = _requests()
    with pytest.raises(ValueError, match="runtime does not match"):
        LocalProbeQaScorer(config, cache, execution_device="cpu").check(requests)
    assert len(loaded) == 1
    assert captured == []


@pytest.mark.parametrize("corruption", ["roles", "decision"])
def test_local_scorer_rejects_internally_inconsistent_cache_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    config, cache, runtime_sha256 = _scorer_files(tmp_path)
    _patch_runtime(monkeypatch, runtime_sha256)
    _, requests = _requests()
    LocalProbeQaScorer(config, cache, execution_device="cpu").check(requests)
    raw = json.loads(cache.read_text(encoding="utf-8"))
    record = next(iter(raw["records"].values()))
    if corruption == "roles":
        record["role_probabilities"][:2] = reversed(record["role_probabilities"][:2])
        expected = "do not match the qualified probe"
    else:
        record["complies"] = not record["complies"]
        record["issue"] = "tampered cached decision" if record["complies"] is False else None
        expected = "does not match its probability and threshold"
    cache.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match=expected):
        LocalProbeQaScorer(config, cache, execution_device="cpu").check(requests)


def test_probe_bundle_rejects_assistant_adapter_mismatch(tmp_path: Path) -> None:
    config = tmp_path / "bundles.json"
    config.write_text(
        json.dumps(
            {
                "bundles": [
                    {
                        "assistant": str(Assistant.GEMMA_4_31B_IT),
                        "adapter": NEMOTRON_ADAPTER.name,
                        "checkpoint": "prefix",
                        "probe": "probe.npz",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="assistant does not match"):
        LocalProbeQaScorer(config, tmp_path / "cache.json")


def test_preflight_rejects_probe_without_native_qualification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, cache, _ = _scorer_files(tmp_path)
    probe = load_role_probe(tmp_path / "probe.npz")
    unqualified = tmp_path / "unqualified.npz"
    save_role_probe(replace(probe, qualification=None), unqualified)
    raw = json.loads(config.read_text(encoding="utf-8"))
    raw["bundles"][0]["probe"] = unqualified.name
    config.write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setattr(local_module, "validate_prefix_checkpoint_identity", lambda *args: None)
    monkeypatch.setattr(
        local_module.importlib,
        "import_module",
        lambda name: SimpleNamespace(),
    )

    with pytest.raises(ValueError, match="not QA-eligible"):
        LocalProbeQaScorer(config, cache).preflight((Assistant.NEMOTRON_3_5_LIGHTNING,))
