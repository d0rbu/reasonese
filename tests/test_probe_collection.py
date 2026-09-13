"""Activation-probe QA runs before assistant collection and excludes whole edges."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from reasonese.axes import Assistant
from reasonese.cache import YamlMessageCache
from reasonese.collect_data import CollectionTask, collect_studies
from reasonese.conversation import GeneratedMessage, GeneratedText
from reasonese.message_qa import parse_message_qa
from reasonese.message_qa_cache import YamlMessageQaCache
from reasonese.openrouter import OpenRouterClient
from reasonese.probe_qa import (
    ProbeExpectation,
    ProbeQaRequest,
    ProbeQaVerdict,
    probe_expectation,
)
from reasonese.routing import CollectionRouting
from reasonese.study import build_trials
from reasonese.study_cache import SqliteStudyCache
from tests.test_authoring_exclusions import _seed, _studies
from tests.test_study_orchestration import (
    FakeTransport,
    _assistant_responses,
    _chat,
    _judge_batch,
    _manual_library,
)


def _verdict(request: ProbeQaRequest, *, complies: bool) -> ProbeQaVerdict:
    expectation = probe_expectation(request.spec)
    reasoning = 0.9 if expectation is ProbeExpectation.REASONING else 0.1
    return ProbeQaVerdict(
        request=request,
        context_fingerprint=f"context-{request.permutation}-{request.position}",
        role_probabilities=(
            ("system", 0.01),
            ("user", 0.02),
            ("tool", 0.01),
            ("reasoning", reasoning),
            ("assistant", 0.96 - reasoning),
        ),
        reasoning_probability=reasoning,
        expectation=expectation,
        complies=complies,
        issue=None if complies else "reasoning probability missed the frozen threshold",
    )


class RecordingScorer:
    def __init__(self, failed_index: int | None) -> None:
        self.failed_index = failed_index
        self.requests: tuple[ProbeQaRequest, ...] = ()
        self.preflighted: tuple[Assistant, ...] = ()

    def preflight(self, assistants: tuple[Assistant, ...]) -> None:
        self.preflighted = assistants

    def check(self, requests: tuple[ProbeQaRequest, ...]) -> tuple[ProbeQaVerdict, ...]:
        self.requests = requests
        return tuple(
            _verdict(request, complies=index != self.failed_index)
            for index, request in enumerate(requests)
        )


def test_failed_second_order_probe_excludes_every_rollout_before_provider_call(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    study = _studies()[0]
    messages, qa, _ = _seed(tmp_path, (study,), set())
    transport = FakeTransport([])
    scorer = RecordingScorer(failed_index=3)
    task = CollectionTask(study, tmp_path / "study")

    result = collect_studies(
        (task,),
        OpenRouterClient(transport),
        _manual_library(tmp_path, study),
        messages,
        qa,
        prefer_batch=False,
        probe_scorer=scorer,
    )[0]

    assert len(scorer.requests) == 4
    assert scorer.preflighted == (study.assistant,)
    assert [(row.permutation, row.position) for row in scorer.requests] == [
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
    ]
    assert scorer.requests[0].setup.matchup.inputs == tuple(
        reversed(scorer.requests[2].setup.matchup.inputs)
    )
    assert len(result.probe_qa_verdicts) == 4
    assert result.trials == result.observations == ()
    assert result.excluded_inputs == ()
    assert transport.post_calls == []
    assert (task.output_dir / "observations.jsonl").read_text() == ""
    assert "permutation=2 position=2" in caplog.text

    report = json.loads((tmp_path / "probe_qa_report.json").read_text())
    assert report["counts"] == {
        "scores": 4,
        "enforced_scores": 4,
        "descriptive_scores": 0,
        "failed_scores": 1,
        "planned_comparisons": 1,
        "excluded_comparisons": 1,
        "planned_trials": len(build_trials(study)),
        "excluded_trials": len(build_trials(study)),
    }
    order = {row["order"]: row for row in report["scores_by_axis"]["order"]}
    assert order["1"]["failed_scores"] == 0
    assert order["2"]["failed_scores"] == 1


def test_probe_scorer_error_propagates_before_provider_call(tmp_path: Path) -> None:
    study = _studies()[0]
    messages, qa, _ = _seed(tmp_path, (study,), set())
    transport = FakeTransport([])

    class BrokenScorer:
        def preflight(self, assistants: tuple[Assistant, ...]) -> None:
            pass

        def check(self, requests: tuple[ProbeQaRequest, ...]) -> tuple[ProbeQaVerdict, ...]:
            raise ValueError("probe artifact is not qualified")

    with pytest.raises(ValueError, match="not qualified"):
        collect_studies(
            (CollectionTask(study, tmp_path / "study"),),
            OpenRouterClient(transport),
            _manual_library(tmp_path, study),
            messages,
            qa,
            prefer_batch=False,
            probe_scorer=BrokenScorer(),
        )
    assert transport.post_calls == []
    assert not (tmp_path / "probe_qa_report.json").exists()


def test_probe_preflight_fails_before_uncached_authoring_or_qa(tmp_path: Path) -> None:
    study = _studies()[0]
    transport = FakeTransport([])

    class BrokenPreflight:
        def preflight(self, assistants: tuple[Assistant, ...]) -> None:
            raise ValueError("missing qualified role-probe artifact")

        def check(self, requests: tuple[ProbeQaRequest, ...]) -> tuple[ProbeQaVerdict, ...]:
            raise AssertionError("check must not run after failed preflight")

    with pytest.raises(ValueError, match="missing qualified"):
        collect_studies(
            (CollectionTask(study, tmp_path / "study"),),
            OpenRouterClient(transport),
            _manual_library(tmp_path, study),
            YamlMessageCache(tmp_path / "messages.yaml"),
            YamlMessageQaCache(tmp_path / "qa.yaml"),
            prefer_batch=False,
            probe_scorer=BrokenPreflight(),
        )
    assert transport.post_calls == []
    assert not (tmp_path / "authoring_report.json").exists()


def test_probe_mode_recollects_every_cached_trace_with_stale_authored_text(
    tmp_path: Path,
) -> None:
    study = _studies()[0]
    messages, qa, _ = _seed(tmp_path, (study,), set())
    task = CollectionTask(study, tmp_path / "study")
    manual = _manual_library(tmp_path, study)
    initial = FakeTransport(
        [*_assistant_responses(4), _judge_batch((True, False, False, True) * 2)]
    )
    collect_studies(
        (task,),
        OpenRouterClient(initial),
        manual,
        messages,
        qa,
        prefer_batch=False,
        probe_scorer=RecordingScorer(failed_index=None),
        routing=CollectionRouting(allow_paid=True),
    )

    changed_spec = study.inputs[0]
    changed = GeneratedMessage(
        changed_spec,
        GeneratedText.parse("A newly audited rendering of the same requested task."),
        _chat("author", "changed-author-response"),
    )
    messages.put_many((changed,))
    qa.put_many((parse_message_qa(changed, _chat('{"complies":true,"issues":[]}', "qa")),))
    trials = build_trials(study)
    with sqlite3.connect(task.output_dir / "collection.sqlite3") as connection:
        connection.execute("DELETE FROM traces WHERE trial_id = ?", (str(trials[0].trial_id),))

    resumed_transport = FakeTransport(
        [*_assistant_responses(4), _judge_batch((True, False, False, True) * 2)]
    )
    scorer = RecordingScorer(failed_index=None)
    result = collect_studies(
        (task,),
        OpenRouterClient(resumed_transport),
        manual,
        messages,
        qa,
        prefer_batch=False,
        probe_scorer=scorer,
        routing=CollectionRouting(allow_paid=True),
    )[0]

    assert result.trace_cache_hits == 0
    assert len(result.trials) == 4
    assert any(request.setup.content_for_input(0) == changed.content for request in scorer.requests)
    expected_by_matchup = {request.setup.matchup: request.setup for request in scorer.requests}
    cached = SqliteStudyCache(task.output_dir / "collection.sqlite3").load_traces(result.trials)
    assert all(
        cached[trial.trial_id].setup == expected_by_matchup[trial.matchup]
        for trial in result.trials
    )
