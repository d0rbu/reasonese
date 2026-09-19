"""Optional probe diagnostics leave collection eligibility and outcomes unchanged."""

from __future__ import annotations

import builtins
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from reasonese.axes import Assistant
from reasonese.collect_data import CollectionTask, collect_studies
from reasonese.openrouter import OpenRouterClient
from reasonese.probe_qa import (
    ProbeExpectation,
    ProbeQaMode,
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
    _manual_library,
)


def test_base_collection_imports_and_help_do_not_require_probe_dependencies() -> None:
    script = textwrap.dedent(
        """
        import contextlib
        import io
        import sys

        class BlockProbeDependencies:
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'sklearn', 'torch', 'transformers', 'kernels'}:
                    raise ModuleNotFoundError(fullname)
                return None

        sys.meta_path.insert(0, BlockProbeDependencies())
        from reasonese import collect_data, collect_studies
        for module in (collect_data, collect_studies):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    module.main(['--help'])
            except SystemExit as error:
                assert error.code == 0
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_explicit_off_cli_never_imports_the_local_scorer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from phantom.interval import Natural

    import reasonese.collect_data as collect_data_module

    study = _studies()[0]
    monkeypatch.setattr(collect_data_module, "load_study", lambda path: study)
    monkeypatch.setattr(
        collect_data_module,
        "collect_study",
        lambda *args, **kwargs: collect_data_module.CollectionResult(
            (), (), Natural.parse(0), Natural.parse(0)
        ),
    )
    original_import = builtins.__import__

    def forbid_local_probe(name, *args, **kwargs):
        if name == "reasonese.local_probe_qa":
            raise AssertionError("off mode imported the local scorer")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_local_probe)
    assert collect_data_module.main(
        [
            "--study",
            str(tmp_path / "unused-study.yaml"),
            "--output",
            str(tmp_path / "out"),
            "--probe-mode",
            "off",
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out)["probe_mode"] == "off"


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


def _sync_judgments(values: tuple[bool, ...]) -> list[dict[str, object]]:
    return [
        _chat(json.dumps({"completed": value}), f"judge-{index}")
        for index, value in enumerate(values)
    ]


def test_low_probe_score_keeps_trials_observations_and_order_balance(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    study = _studies()[0]
    messages, qa, _ = _seed(tmp_path, (study,), set())
    transport = FakeTransport(
        [*_assistant_responses(4), *_sync_judgments((True, False, False, True) * 2)]
    )
    scorer = RecordingScorer(failed_index=3)
    task = CollectionTask(study, tmp_path / "study")

    result = collect_studies(
        (task,),
        OpenRouterClient(transport),
        _manual_library(tmp_path, study),
        messages,
        qa,
        prefer_batch=False,
        probe_mode=ProbeQaMode.INLINE,
        probe_scorer=scorer,
        routing=CollectionRouting(allow_paid=True),
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
    assert len(result.trials) == len(build_trials(study))
    assert len(result.observations) == 2 * len(build_trials(study))
    assert result.excluded_inputs == ()
    assert result.probe_qa_issues == ()
    assert len(transport.post_calls) == 12
    assert "reference mismatch (diagnostic only)" in caplog.text

    report = json.loads((tmp_path / "probe_qa_report.json").read_text())
    assert report["counts"] == {
        "planned_requests": 4,
        "scores": 4,
        "directional_scores": 4,
        "descriptive_scores": 0,
        "reference_mismatches": 1,
        "missing_requests": 0,
        "error_requests": 0,
        "coverage": 1.0,
        "masked_boundary_tokens": 0,
    }
    order = {row["order"]: row for row in report["scores_by_axis"]["order"]}
    assert order["1"]["reference_mismatches"] == 0
    assert order["2"]["reference_mismatches"] == 1
    assert not report["missing"] and not report["errors"]


def test_probe_scorer_error_is_reported_but_does_not_stop_collection(tmp_path: Path) -> None:
    study = _studies()[0]
    messages, qa, _ = _seed(tmp_path, (study,), set())
    transport = FakeTransport(
        [*_assistant_responses(4), *_sync_judgments((True, False, False, True) * 2)]
    )

    class BrokenScorer:
        def preflight(self, assistants: tuple[Assistant, ...]) -> None:
            pass

        def check(self, requests: tuple[ProbeQaRequest, ...]) -> tuple[ProbeQaVerdict, ...]:
            raise ValueError("probe artifact is not qualified")

    result = collect_studies(
        (CollectionTask(study, tmp_path / "study"),),
        OpenRouterClient(transport),
        _manual_library(tmp_path, study),
        messages,
        qa,
        prefer_batch=False,
        probe_mode=ProbeQaMode.INLINE,
        probe_scorer=BrokenScorer(),
        routing=CollectionRouting(allow_paid=True),
    )[0]
    assert len(result.trials) == len(build_trials(study))
    assert len(result.observations) == 2 * len(build_trials(study))
    assert len(transport.post_calls) == 12
    report = json.loads((tmp_path / "probe_qa_report.json").read_text())
    assert report["counts"]["scores"] == 0
    assert report["counts"]["error_requests"] == 4
    assert report["counts"]["missing_requests"] == 0
    assert all(row["reason"].startswith("probe scoring failed") for row in report["errors"])


def test_probe_preflight_error_is_diagnostic_and_collection_continues(tmp_path: Path) -> None:
    study = _studies()[0]
    messages, qa, _ = _seed(tmp_path, (study,), set())
    transport = FakeTransport(
        [*_assistant_responses(4), *_sync_judgments((True, False, False, True) * 2)]
    )

    class BrokenPreflight:
        def preflight(self, assistants: tuple[Assistant, ...]) -> None:
            raise ValueError("missing qualified role-probe artifact")

        def check(self, requests: tuple[ProbeQaRequest, ...]) -> tuple[ProbeQaVerdict, ...]:
            raise AssertionError("check must not run after failed preflight")

    result = collect_studies(
        (CollectionTask(study, tmp_path / "study"),),
        OpenRouterClient(transport),
        _manual_library(tmp_path, study),
        messages,
        qa,
        prefer_batch=False,
        probe_mode=ProbeQaMode.INLINE,
        probe_scorer=BrokenPreflight(),
        routing=CollectionRouting(allow_paid=True),
    )[0]
    assert len(result.trials) == len(build_trials(study))
    assert len(result.observations) == 2 * len(build_trials(study))
    assert len(transport.post_calls) == 12
    report = json.loads((tmp_path / "probe_qa_report.json").read_text())
    assert report["counts"]["error_requests"] == 4
    assert all(row["reason"].startswith("probe preflight failed") for row in report["errors"])


def test_warm_probe_mode_preserves_cached_trace_context_and_outcomes(
    tmp_path: Path,
) -> None:
    study = _studies()[0]
    messages, qa, _ = _seed(tmp_path, (study,), set())
    task = CollectionTask(study, tmp_path / "study")
    manual = _manual_library(tmp_path, study)
    initial = FakeTransport(
        [*_assistant_responses(4), *_sync_judgments((True, False, False, True) * 2)]
    )
    baseline = collect_studies(
        (task,),
        OpenRouterClient(initial),
        manual,
        messages,
        qa,
        prefer_batch=False,
        routing=CollectionRouting(allow_paid=True),
    )

    trials = build_trials(study)
    before = SqliteStudyCache(task.output_dir / "collection.sqlite3").load_traces(trials)
    transport = FakeTransport([])
    scorer = RecordingScorer(failed_index=None)
    result = collect_studies(
        (task,),
        OpenRouterClient(transport),
        manual,
        messages,
        qa,
        prefer_batch=False,
        probe_mode=ProbeQaMode.INLINE,
        probe_scorer=scorer,
        routing=CollectionRouting(allow_paid=True),
    )[0]

    assert result.trace_cache_hits == len(trials)
    assert len(result.trials) == len(trials)
    assert result.observations == baseline[0].observations
    assert transport.post_calls == []
    assert len(scorer.requests) == 4
    expected_by_matchup = {request.setup.matchup: request.setup for request in scorer.requests}
    cached = SqliteStudyCache(task.output_dir / "collection.sqlite3").load_traces(result.trials)
    assert all(
        cached[trial.trial_id].setup == expected_by_matchup[trial.matchup]
        for trial in result.trials
    )
    after = SqliteStudyCache(task.output_dir / "collection.sqlite3").load_traces(trials)
    assert before == after
