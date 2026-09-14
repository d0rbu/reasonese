"""Concrete local activation-probe scorer and cache tests."""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

import reasonese.local_probe_qa as local_module
from reasonese.axes import Assistant, Framing
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
        training=replace(probe.training, layer_indices=(26,)),
        selected_layer_index=26,
        development_candidates=tuple(
            replace(candidate, layer_index=26) for candidate in probe.development_candidates
        ),
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
) -> tuple[list[object], list[tuple[tuple[int, ...], tuple[int, ...]]]]:
    loaded: list[object] = []
    captured: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
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
    monkeypatch.setattr(local_module, "_pinned_nemotron_cumsum", lambda *args: nullcontext())

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
        captured.append((input_ids, token_positions))
        values = np.zeros((len(token_positions), 1, 5), dtype=np.float32)
        values[:, 0, 0] = 0.1
        return values

    monkeypatch.setattr(local_module, "capture_token_activations", capture)
    return loaded, captured


def test_local_scorer_captures_target_prefixes_and_reuses_exact_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, cache, runtime_sha256 = _scorer_files(tmp_path)
    loaded, captured = _patch_runtime(monkeypatch, runtime_sha256)
    _, requests = _requests()
    scorer = LocalProbeQaScorer(config, cache, execution_device="cpu")
    verdicts = scorer.check(requests)

    assert len(verdicts) == 4
    assert len(loaded) == 2
    assert captured == [
        (tuple(range(3)), (1, 2)),
        (tuple(range(9)), (7, 8)),
        (tuple(range(3)), (1, 2)),
        (tuple(range(9)), (7, 8)),
    ]
    assert len({row.context_fingerprint for row in verdicts}) == 4
    assert all(len(row.role_probabilities) == 5 for row in verdicts)
    assert "OpenRouter" in scorer.limitations[0]
    assert not any(
        str(request.setup.content_for_input(0)) in cache.read_text() for request in requests
    )

    assert scorer.check(requests) == verdicts
    assert len(loaded) == 2
    assert len(captured) == 4


def test_local_scorer_invalidates_old_full_context_cache_and_hits_new_prefix_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, cache, runtime_sha256 = _scorer_files(tmp_path)
    _, captured = _patch_runtime(monkeypatch, runtime_sha256)
    _, requests = _requests()
    request = requests[0]

    initial_scorer = LocalProbeQaScorer(config, cache, execution_device="cpu")
    initial_scorer.check((request,))
    raw = json.loads(cache.read_text(encoding="utf-8"))
    current_key, record = next(iter(raw["records"].items()))
    prepared = initial_scorer._prepared[request.setup.matchup.assistant]
    context = local_module.render_collector_probe_context(
        object(), prepared.bundle.adapter, request.setup, request.position
    )
    prefix = initial_scorer._target_prefix(context)
    monkeypatch.setattr(local_module, "CAPTURE_POLICY", "segment_prefix_capture_v1")
    previous_prefix_key = initial_scorer._context_key(
        prepared,
        request,
        prefix.input_ids,
        prefix.token_positions[0],
        prefix.content_token_ids[0],
        prefix.render_config_sha256,
    )
    monkeypatch.setattr(
        local_module, "CAPTURE_POLICY", "segment_prefix_capture_v2_nemotron_cumsum_h1"
    )
    assert previous_prefix_key != current_key
    legacy_key = local_module._canonical_sha256(
        {
            "probe_sha256": prepared.probe_sha256,
            "runtime_sha256": prepared.probe.provenance.runtime_sha256,
            "weights_sha256": prepared.probe.provenance.weights_sha256,
            "adapter": prepared.bundle.adapter.name,
            "assistant": str(request.setup.matchup.assistant),
            "study_id": request.study_id,
            "permutation": request.permutation,
            "position": request.position,
            "render_config_sha256": context.render_config_sha256,
            "input_ids": context.input_ids,
            "token_positions": context.token_positions[0],
            "content_token_ids": context.content_token_ids[0],
        }
    )
    assert legacy_key != current_key

    # A pre-fix full-context record must miss under the explicit prefix policy.
    cache.write_text(
        json.dumps({"format_version": 1, "records": {legacy_key: record}}),
        encoding="utf-8",
    )
    captured.clear()
    scorer = LocalProbeQaScorer(config, cache, execution_device="cpu")
    scorer.check((request,))
    assert len(captured) == 1

    # The newly written prefix record is reusable on the next check.
    captured.clear()
    scorer.check((request,))
    assert captured == []


class _FakeConfig:
    def __init__(self, head_tile: int) -> None:
        self.head_tile = head_tile
        self.pre_hook = None

    def all_kwargs(self) -> dict[str, int]:
        return {
            "BLOCK_SIZE_H": self.head_tile,
            "num_ctas": 1,
            "num_stages": 3,
            "num_warps": 4,
        }


class _FakeAutotuner:
    def __init__(self) -> None:
        self.configs = [_FakeConfig(value) for value in (1, 2, 4, 8, 16, 32, 64)]
        self.cache = {("observed",): self.configs[3]}
        self.best_config = self.configs[3]
        self.nargs = {"prior": True}


def _patch_cumsum_runtime(
    monkeypatch: pytest.MonkeyPatch, tuner: _FakeAutotuner
) -> object:
    root = "_mamba_ssm_cuda_test"
    combined_module_name = f"{root}.ops.triton.ssd_combined"
    state_module_name = f"{root}.ops.triton.ssd_chunk_state"

    def combined() -> None:
        pass

    combined.__module__ = combined_module_name
    model_module = ModuleType("test_nemotron_model")
    model_module.__dict__["mamba_chunk_scan_combined"] = combined
    model_type = type("FakeModel", (), {"__module__": model_module.__name__})
    state_module = SimpleNamespace(_chunk_cumsum_fwd_kernel=SimpleNamespace(fn=tuner))
    monkeypatch.setitem(sys.modules, model_module.__name__, model_module)
    monkeypatch.setitem(
        sys.modules,
        "_mamba_ssm_cuda_unused.ops.triton.ssd_chunk_state",
        SimpleNamespace(_chunk_cumsum_fwd_kernel=_FakeAutotuner()),
    )
    real_import = local_module.importlib.import_module
    monkeypatch.setattr(
        local_module.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(Autotuner=_FakeAutotuner)
            if name == "triton.runtime.autotuner"
            else state_module
            if name == state_module_name
            else real_import(name)
        ),
    )
    return model_type()


def test_nemotron_cumsum_pin_uses_h1_despite_stale_cache_and_restores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tuner = _FakeAutotuner()
    original_configs = tuner.configs
    original_cache = dict(tuner.cache)
    original_best = tuner.best_config
    original_nargs = tuner.nargs
    model = _patch_cumsum_runtime(monkeypatch, tuner)

    with local_module._pinned_nemotron_cumsum(model, NEMOTRON_ADAPTER, "cuda:0"):
        assert len(tuner.configs) == 1
        assert tuner.configs[0].all_kwargs()["BLOCK_SIZE_H"] == 1
        assert tuner.cache == original_cache

    assert tuner.configs is original_configs
    assert tuner.cache == original_cache
    assert tuner.best_config is original_best
    assert tuner.nargs is original_nargs


def test_nemotron_cumsum_pin_restores_after_capture_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tuner = _FakeAutotuner()
    original_configs = tuner.configs
    model = _patch_cumsum_runtime(monkeypatch, tuner)

    with (
        pytest.raises(RuntimeError, match="capture failed"),
        local_module._pinned_nemotron_cumsum(model, NEMOTRON_ADAPTER, "cuda:0"),
    ):
        tuner.best_config = tuner.configs[0]
        tuner.nargs = {"active": True}
        raise RuntimeError("capture failed")

    assert tuner.configs is original_configs
    assert tuner.best_config is original_configs[3]
    assert tuner.nargs == {"prior": True}


def test_nemotron_cumsum_pin_fails_closed_on_incompatible_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tuner = _FakeAutotuner()
    tuner.configs[0] = _FakeConfig(3)
    model = _patch_cumsum_runtime(monkeypatch, tuner)

    with (
        pytest.raises(RuntimeError, match="configs differ"),
        local_module._pinned_nemotron_cumsum(model, NEMOTRON_ADAPTER, "cuda:0"),
    ):
        pytest.fail("incompatible config registry entered capture scope")


@pytest.mark.parametrize(
    ("adapter", "device"),
    [(NEMOTRON_ADAPTER, "cpu"), (SimpleNamespace(name="gemma-native"), "cuda:0")],
)
def test_cumsum_pin_does_not_touch_cpu_or_other_adapters(
    monkeypatch: pytest.MonkeyPatch, adapter: Any, device: str
) -> None:
    monkeypatch.setattr(
        local_module,
        "_nemotron_cumsum_autotuner",
        lambda model: pytest.fail("unaffected scorer resolved a Nemotron Triton kernel"),
    )
    with local_module._pinned_nemotron_cumsum(object(), adapter, device):
        pass


def test_target_prefix_drops_only_external_future_and_preserves_span_positions() -> None:
    before = RenderedProbeContext(
        (10, 11, 12, 13, 40, 41), ((2, 3),), ((12, 13),), "a" * 64
    )
    after = RenderedProbeContext(
        (10, 11, 12, 13, 90, 91, 92), ((2, 3),), ((12, 13),), "a" * 64
    )

    first = LocalProbeQaScorer._target_prefix(before)
    second = LocalProbeQaScorer._target_prefix(after)

    assert first.input_ids == second.input_ids == (10, 11, 12, 13)
    assert first.token_positions == second.token_positions == ((2, 3),)
    assert first.content_token_ids == second.content_token_ids == ((12, 13),)


def test_target_prefix_rejects_multiple_measured_spans() -> None:
    context = RenderedProbeContext(
        (10, 11, 12, 13, 14), ((1,), (3,)), ((11,), (13,)), "a" * 64
    )

    with pytest.raises(ValueError, match="exactly one measured span"):
        LocalProbeQaScorer._target_prefix(context)


def test_local_scorer_keeps_compressed_scores_descriptive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, cache, runtime_sha256 = _scorer_files(tmp_path)
    _patch_runtime(monkeypatch, runtime_sha256)
    _, requests = _requests(Framing.COMPRESSED_NORMAL)
    verdicts = LocalProbeQaScorer(config, cache, execution_device="cpu").check(requests)
    compressed = tuple(
        verdict for verdict in verdicts if verdict.request.spec.framing is Framing.COMPRESSED_NORMAL
    )
    assert len(compressed) == 2
    assert all(verdict.complies is None and verdict.issue is None for verdict in compressed)


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


@pytest.mark.parametrize(
    "captured",
    [
        np.zeros((4, 1, 5), dtype=np.float64),
        np.zeros((4, 2, 5), dtype=np.float32),
        np.zeros((3, 1, 5), dtype=np.float32),
    ],
)
def test_local_scorer_rejects_capture_shape_layer_or_dtype_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    captured: np.ndarray,
) -> None:
    config, cache, runtime_sha256 = _scorer_files(tmp_path)
    _patch_runtime(monkeypatch, runtime_sha256)
    monkeypatch.setattr(local_module, "capture_token_activations", lambda *args, **kwargs: captured)
    _, requests = _requests()

    with pytest.raises(ValueError, match="invalid shape"):
        LocalProbeQaScorer(config, cache, execution_device="cpu").check(requests)


def test_local_scorer_uses_identical_prefix_for_different_future_suffixes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, cache, runtime_sha256 = _scorer_files(tmp_path)
    _, captured = _patch_runtime(monkeypatch, runtime_sha256)

    def mismatched_render(tokenizer: object, adapter: object, setup: object, position: int):
        input_ids = tuple(range(12 + position))
        positions = (1, 2)
        return RenderedProbeContext(
            input_ids,
            (positions,),
            (tuple(input_ids[index] for index in positions),),
            "a" * 64,
        )

    monkeypatch.setattr(local_module, "render_collector_probe_context", mismatched_render)
    _, requests = _requests()
    verdicts = LocalProbeQaScorer(config, cache, execution_device="cpu").check(requests)
    assert len(verdicts) == 4
    assert {input_ids for input_ids, _ in captured} == {(0, 1, 2),}
    assert {positions for _, positions in captured} == {(1, 2),}


def test_local_scorer_requires_a_bundle_for_every_assistant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, cache, runtime_sha256 = _scorer_files(tmp_path)
    _patch_runtime(monkeypatch, runtime_sha256)
    scorer = LocalProbeQaScorer(config, cache, execution_device="cpu")
    assert scorer.check(()) == ()
    with pytest.raises(ValueError, match="no local role-probe bundle"):
        scorer.preflight((Assistant.GEMMA_4_31B_IT,))


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


@pytest.mark.parametrize(
    "corruption", ["fields", "probability-type", "probability-row", "decision"]
)
def test_local_scorer_rejects_malformed_cached_verdicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    config, cache, runtime_sha256 = _scorer_files(tmp_path)
    _patch_runtime(monkeypatch, runtime_sha256)
    _, requests = _requests()
    LocalProbeQaScorer(config, cache, execution_device="cpu").check(requests)
    raw = json.loads(cache.read_text(encoding="utf-8"))
    record = next(iter(raw["records"].values()))
    if corruption == "fields":
        record.pop("issue")
        expected = "cache verdict"
    elif corruption == "probability-type":
        record["role_probabilities"] = {}
        expected = "cached role probabilities"
    elif corruption == "probability-row":
        record["role_probabilities"][0] = []
        expected = "cached role probabilities"
    else:
        record["complies"] = "yes"
        expected = "cached probe decision"
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


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "invalid JSON object"),
        ({"bundles": {}}, "must contain only a bundles list"),
        ({"bundles": []}, "requires distinct assistant bundles"),
        (
            {
                "bundles": [
                    {
                        "assistant": "unsupported",
                        "adapter": NEMOTRON_ADAPTER.name,
                        "checkpoint": "prefix",
                        "probe": "probe.npz",
                    }
                ]
            },
            "unsupported assistant or adapter",
        ),
        (
            {
                "bundles": [
                    {
                        "assistant": str(Assistant.NEMOTRON_3_5_LIGHTNING),
                        "adapter": NEMOTRON_ADAPTER.name,
                        "checkpoint": 3,
                        "probe": "probe.npz",
                    }
                ]
            },
            "paths must be strings",
        ),
    ],
)
def test_probe_bundle_config_rejects_malformed_records(
    tmp_path: Path, payload: object, message: str
) -> None:
    config = tmp_path / "bundles.json"
    config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        LocalProbeQaScorer(config, tmp_path / "cache.json")


def test_probe_bundle_config_rejects_invalid_json_and_incomplete_record(tmp_path: Path) -> None:
    config = tmp_path / "bundles.json"
    config.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON object"):
        LocalProbeQaScorer(config, tmp_path / "cache.json")
    config.write_text(json.dumps({"bundles": [{"assistant": "missing fields"}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid role-probe bundle record"):
        LocalProbeQaScorer(config, tmp_path / "cache.json")


@pytest.mark.parametrize(
    "payload",
    [
        {"format_version": 2, "records": {}},
        {"format_version": 1, "records": []},
    ],
)
def test_local_probe_cache_rejects_invalid_envelopes(tmp_path: Path, payload: object) -> None:
    config, cache, _ = _scorer_files(tmp_path)
    cache.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="probe-QA cache"):
        LocalProbeQaScorer(config, cache)


def test_preflight_rejects_checkpoint_manifest_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, cache, runtime_sha256 = _scorer_files(tmp_path)
    _patch_runtime(monkeypatch, runtime_sha256)
    manifest = tmp_path / "prefix" / "prefix-checkpoint-manifest.json"
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["max_layer"] += 1
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint does not match"):
        LocalProbeQaScorer(config, cache).preflight((Assistant.NEMOTRON_3_5_LIGHTNING,))


def test_preflight_rejects_incompatible_cumsum_policy_before_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, cache, runtime_sha256 = _scorer_files(tmp_path)
    _patch_runtime(monkeypatch, runtime_sha256)
    entered = 0

    @contextmanager
    def incompatible(*args: object):
        nonlocal entered
        entered += 1
        raise RuntimeError("cumsum config incompatible")
        yield

    monkeypatch.setattr(local_module, "_pinned_nemotron_cumsum", incompatible)
    scorer = LocalProbeQaScorer(config, cache)
    with pytest.raises(RuntimeError, match="cumsum config incompatible"):
        scorer.preflight((Assistant.NEMOTRON_3_5_LIGHTNING,))

    assert entered == 1
    assert scorer._prepared == {}
    assert not cache.exists()


def test_preflight_uses_extraction_provenance_for_checkpoint_depth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, cache, runtime_sha256 = _scorer_files(tmp_path)
    loaded_layers: list[int] = []
    _patch_runtime(monkeypatch, runtime_sha256)
    original_load = local_module.load_prefix_model

    def record_load(*args: Any, max_layer: int, **kwargs: Any) -> object:
        loaded_layers.append(max_layer)
        return original_load(*args, max_layer=max_layer, **kwargs)

    monkeypatch.setattr(local_module, "load_prefix_model", record_load)
    probe_path = tmp_path / "probe.npz"
    probe = load_role_probe(probe_path)
    restricted = replace(
        probe,
        provenance=replace(probe.provenance, layer_indices=(13, 26)),
        training=replace(probe.training, layer_indices=(13,)),
        selected_layer_index=13,
        development_candidates=tuple(
            replace(candidate, layer_index=13) for candidate in probe.development_candidates
        ),
    )
    save_role_probe(restricted, probe_path)

    LocalProbeQaScorer(config, cache).preflight((Assistant.NEMOTRON_3_5_LIGHTNING,))

    assert loaded_layers == [13]


def test_preflight_rejects_probe_pipeline_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, cache, runtime_sha256 = _scorer_files(tmp_path)
    _patch_runtime(monkeypatch, runtime_sha256)
    probe_path = tmp_path / "probe.npz"
    probe = load_role_probe(probe_path)
    save_role_probe(
        replace(probe, provenance=replace(probe.provenance, activation_dtype="float16")),
        tmp_path / "mismatched.npz",
    )
    raw = json.loads(config.read_text(encoding="utf-8"))
    raw["bundles"][0]["probe"] = "mismatched.npz"
    config.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match its native adapter"):
        LocalProbeQaScorer(config, cache).preflight((Assistant.NEMOTRON_3_5_LIGHTNING,))


def test_preflight_reports_missing_optional_runtime_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, cache, _ = _scorer_files(tmp_path)
    monkeypatch.setattr(
        local_module.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError(name)),
    )
    with pytest.raises(RuntimeError, match="requires the 'probes' extra"):
        LocalProbeQaScorer(config, cache).preflight((Assistant.NEMOTRON_3_5_LIGHTNING,))


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
