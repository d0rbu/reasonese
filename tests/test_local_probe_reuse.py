"""Prefix reuse must preserve request identity, score inputs, and recovery."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import reasonese.local_probe_qa as local_module
from reasonese.axes import Channel, Framing
from reasonese.local_probe_qa import LocalProbeQaScorer
from reasonese.probe_rendering import RenderedProbeContext
from tests.test_local_probe_qa import (
    _add_thresholds,
    _patch_runtime,
    _scorer_files,
    _threshold_payload,
)
from tests.test_probe_qa import _requests


@pytest.mark.parametrize("difference", ("prefix", "positions"))
def test_reuse_separates_different_capture_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, difference: str
) -> None:
    config, cache, runtime_sha256 = _scorer_files(tmp_path)
    _patch_runtime(monkeypatch, runtime_sha256)
    _, requests = _requests()
    calls = []

    def render(tokenizer, adapter, setup, position):
        ids = (10, 11, 12, 13)
        positions = (1, 3)
        if position == 2:
            if difference == "prefix":
                # The measured content is unchanged; preceding context differs.
                ids = (20, 11, 12, 13)
            else:
                # The complete prefix is unchanged; the measured span differs.
                positions = (2, 3)
        return RenderedProbeContext(
            ids, (positions,), (tuple(ids[i] for i in positions),), "a" * 64
        )

    def capture(model, adapter, *, input_ids, token_positions, layers):
        calls.append((input_ids, token_positions))
        values = np.zeros((len(token_positions), 1, 5), dtype=np.float32)
        values[:, 0, 0] = input_ids[0] + sum(token_positions)
        return values

    monkeypatch.setattr(local_module, "render_collector_probe_context", render)
    monkeypatch.setattr(local_module, "capture_token_activations", capture)
    verdicts = LocalProbeQaScorer(config, cache, execution_device="cpu").check(requests)

    assert len(calls) == 2
    assert calls[0] != calls[1]
    assert verdicts[0].role_probabilities != verdicts[1].role_probabilities
    assert verdicts[0].role_probabilities == verdicts[2].role_probabilities
    assert verdicts[1].role_probabilities == verdicts[3].role_probabilities
    assert len({row.context_fingerprint for row in verdicts}) == 4


@pytest.mark.parametrize("framing", (Framing.REASONESE_NORMAL, Framing.COMPRESSED_NORMAL))
def test_reuse_keeps_request_thresholds_masks_and_cache_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, framing: Framing
) -> None:
    config, cache, runtime_sha256 = _scorer_files(tmp_path)
    _, captured = _patch_runtime(monkeypatch, runtime_sha256)
    thresholds = _threshold_payload(reasoning_minimum=0.0, nonreasoning_maximum=1.0)
    thresholds[str(Channel.SYSTEM)]["nonreasoning_maximum"] = 0.0
    _add_thresholds(config, thresholds)
    _, requests = _requests(framing)

    def render(tokenizer, adapter, setup, position):
        return RenderedProbeContext(
            (10, 11, 12, 100 + position), ((1, 2),), ((11, 12),),
            str(position) * 64, masked_boundary_tokens=position - 1,
        )

    monkeypatch.setattr(local_module, "render_collector_probe_context", render)
    scorer = LocalProbeQaScorer(config, cache, execution_device="cpu")
    verdicts = scorer.check(requests)

    assert captured == [((10, 11, 12), (1, 2))]
    assert len({row.role_probabilities for row in verdicts}) == 1
    assert [row.masked_boundary_tokens for row in verdicts] == [0, 1, 0, 1]
    assert tuple(row.request for row in verdicts) == requests
    for row in verdicts:
        expected = (
            False if row.request.spec.channel is Channel.SYSTEM
            else None if framing is Framing.COMPRESSED_NORMAL
            else True
        )
        assert row.complies is expected
    records = json.loads(cache.read_text())["records"]
    assert set(records) == {row.context_fingerprint for row in verdicts}
    assert len(records) == 4
    # A fresh scorer still reads each request's own durable verdict.
    assert LocalProbeQaScorer(config, cache, execution_device="cpu").check(requests) == verdicts
    assert len(captured) == 1


def test_reuse_is_scoped_to_one_model_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, cache, runtime_sha256 = _scorer_files(tmp_path)
    loaded, captured = _patch_runtime(monkeypatch, runtime_sha256)
    _, requests = _requests()
    scorer = LocalProbeQaScorer(config, cache, execution_device="cpu")
    first = scorer.check(requests)
    later_requests = tuple(replace(row, study_id=f"later-{row.study_id}") for row in requests)
    later = scorer.check(later_requests)

    assert len(loaded) == 3  # preflight, then a fresh model for each scoring call
    assert len(captured) == 4  # two distinct prefixes per model invocation
    assert {row.context_fingerprint for row in first}.isdisjoint(
        row.context_fingerprint for row in later
    )
    assert [row.role_probabilities for row in first] == [row.role_probabilities for row in later]
    assert scorer.check(requests) == first
    assert len(captured) == 4


def test_reuse_preserves_completed_records_when_a_later_capture_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, cache, runtime_sha256 = _scorer_files(tmp_path)
    _patch_runtime(monkeypatch, runtime_sha256)
    _, requests = _requests()
    capture = local_module.capture_token_activations
    failed = False

    def interrupted_capture(model, adapter, *, input_ids, token_positions, layers):
        nonlocal failed
        if len(input_ids) == 9 and not failed:
            failed = True
            raise RuntimeError("interrupted capture")
        return capture(
            model, adapter, input_ids=input_ids, token_positions=token_positions, layers=layers
        )

    monkeypatch.setattr(local_module, "capture_token_activations", interrupted_capture)
    scorer = LocalProbeQaScorer(config, cache, execution_device="cpu")
    with pytest.raises(RuntimeError, match="interrupted capture"):
        scorer.check(requests)
    persisted = json.loads(cache.read_text())["records"]
    assert len(persisted) == 1

    resumed = LocalProbeQaScorer(config, cache, execution_device="cpu")
    verdicts = resumed.check(requests)
    records = json.loads(cache.read_text())["records"]
    assert len(records) == len(verdicts) == 4
    assert tuple(row.request for row in verdicts) == requests
    assert all(records[key] == value for key, value in persisted.items())
    assert resumed.check(requests) == verdicts
