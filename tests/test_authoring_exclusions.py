"""Whole-comparison QA exclusions, reporting, replay, and unchanged retained outcomes."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from reasonese.authoring_report import authoring_report
from reasonese.axes import Assistant, Author, Channel, Framing
from reasonese.cache import YamlMessageCache
from reasonese.collect_data import CollectionTask, collect_studies
from reasonese.collect_studies import main as collect_cli
from reasonese.conversation import GeneratedMessage, GeneratedText
from reasonese.io import write_study_suite
from reasonese.message_qa import parse_message_qa
from reasonese.message_qa_cache import YamlMessageQaCache
from reasonese.openrouter import OpenRouterClient
from reasonese.planning import PromptSpec
from reasonese.routing import CollectionRouting
from reasonese.study import Study, build_trials, make_study
from reasonese.study_cache import SqliteStudyCache
from tests.test_study_orchestration import (
    FakeTransport,
    _assistant_responses,
    _chat,
    _judge_batch,
    _manual_library,
    _spec,
)


def _studies() -> tuple[Study, ...]:
    bad = replace(
        _spec("Find a prime.", Channel.README, Author.NEMOTRON_3_5_LIGHTNING),
        framing=Framing.REASONESE_PERSUASIVE,
    )
    good = _spec("Return a table.", Channel.USER, Author.GEMMA_4_31B_IT)
    other = _spec("Return an integer.", Channel.USER, Author.NEMOTRON_3_5_LIGHTNING)
    return (
        make_study((bad, good), Assistant.NEMOTRON_3_5_LIGHTNING, 2),
        make_study((other, good), Assistant.NEMOTRON_3_5_LIGHTNING, 1),
        make_study((bad, other), Assistant.GEMMA_4_31B_IT, 1),
    )


def _seed(root: Path, studies: tuple[Study, ...], failed: set[PromptSpec]):
    messages = tuple(
        GeneratedMessage(spec, GeneratedText.parse(str(spec.instruction)), _chat("author", "a"))
        for spec in dict.fromkeys(spec for study in studies for spec in study.inputs)
    )
    verdicts = tuple(
        parse_message_qa(
            message,
            _chat(
                json.dumps(
                    {
                        "complies": message.spec not in failed,
                        "issues": ["Added algorithm."] if message.spec in failed else [],
                    }
                ),
                "qa",
            ),
        )
        for message in messages
    )
    message_cache = YamlMessageCache(root / "generated_messages.yaml")
    qa_cache = YamlMessageQaCache(root / "message_qa.yaml")
    message_cache.put_many(messages)
    qa_cache.put_many(verdicts)
    return message_cache, qa_cache, verdicts


@pytest.mark.parametrize("shared", [False, True])
def test_mixed_exclusions_preserve_both_orders_and_exact_retained_results(
    tmp_path: Path,
    shared: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    studies = _studies()
    root = tmp_path / "mixed"
    messages, qa, _ = _seed(root, studies, {studies[0].inputs[0]})
    tasks = tuple(CollectionTask(study, root / str(i)) for i, study in enumerate(studies))
    manual = _manual_library(tmp_path, *studies)
    cache = SqliteStudyCache(root / "collection.sqlite3") if shared else None
    transport = FakeTransport([*_assistant_responses(2), _judge_batch((True, False, False, True))])
    results = collect_studies(
        tasks,
        OpenRouterClient(transport),
        manual,
        messages,
        qa,
        prefer_batch=False,
        routing=CollectionRouting(allow_paid=True),
        shared_cache=cache,
    )
    assert [len(r.trials) for r in results] == [0, 2, 0]
    assert [len(r.observations) for r in results] == [0, 4, 0]
    assert [len(r.excluded_inputs) for r in results] == [1, 0, 1]
    assert len(transport.post_calls) == 3  # Only retained assistant trials and their judges.
    assert "author=Nemotron 3.5 Lightning" in caplog.text
    assert "framing=reasonese-persuasive" in caplog.text
    assert "channel=README.md" in caplog.text
    report = json.loads((root / "authoring_report.json").read_text())
    assert report["counts"] == {
        "unique_inputs": 3,
        "failed_inputs": 1,
        "planned_comparisons": 3,
        "excluded_comparisons": 2,
        "planned_trials": 8,
        "excluded_trials": 6,
    }
    assert [r["failed_input_indices"] for r in report["comparisons"]] == [[0], [], [0]]
    assert report["comparisons"][0]["trial_ids"] == [
        str(t.trial_id) for t in build_trials(studies[0])
    ]
    assert len(set(report["comparisons"][0]["cell_ids"])) == 2
    author_rows = {r["author"]: r for r in report["inputs_by_axis"]["author"]}
    assert author_rows["Nemotron 3.5 Lightning"]["inputs"] == 2
    assert author_rows["Nemotron 3.5 Lightning"]["failed_inputs"] == 1
    affected = {r["author"]: r for r in report["comparisons_by_axis"]["author"]}
    assert (
        affected["Nemotron 3.5 Lightning"]["planned_comparisons"] == 3
    )  # Two endpoints count once.
    assert affected["Gemma 4 31B"]["excluded_comparisons"] == 1  # Passing partner loses its edge.
    assert report["comparisons_by_axis"]["assistant"] == [
        {
            "assistant": "Gemma 4 31B",
            "planned_comparisons": 1,
            "excluded_comparisons": 1,
            "planned_trials": 2,
            "excluded_trials": 2,
        },
        {
            "assistant": "Nemotron 3.5 Lightning",
            "planned_comparisons": 2,
            "excluded_comparisons": 1,
            "planned_trials": 6,
            "excluded_trials": 4,
        },
    ]
    # No new calls, permission, or key are needed to replay QA-rejected, uncollected studies.
    warm = collect_studies(
        tasks, None, manual, messages, qa, prefer_batch=False, shared_cache=cache
    )
    assert [r.observations for r in warm] == [r.observations for r in results]
    assert [r.trace_cache_hits for r in warm] == [0, 2, 0]
    assert json.loads((root / "authoring_report.json").read_text()) == report

    # Collect the surviving comparison alone: exact outcomes, requests, IDs and rows match.
    baseline = tmp_path / "baseline"
    bm, bq, _ = _seed(baseline, (studies[1],), set())
    bt = FakeTransport([*_assistant_responses(2), _judge_batch((True, False, False, True))])
    only = collect_studies(
        (CollectionTask(studies[1], baseline),),
        OpenRouterClient(bt),
        manual,
        bm,
        bq,
        prefer_batch=False,
        routing=CollectionRouting(allow_paid=True),
    )[0]
    assert only.observations == results[1].observations
    assert bt.post_calls == transport.post_calls


def test_two_failed_endpoints_count_one_comparison_and_no_synthetic_outcomes(
    tmp_path: Path,
) -> None:
    study = _studies()[0]
    messages, qa, _ = _seed(tmp_path, (study,), set(study.inputs))
    result = collect_studies(
        (CollectionTask(study, tmp_path / "study"),),
        None,
        _manual_library(tmp_path, study),
        messages,
        qa,
        prefer_batch=False,
    )[0]
    assert len(result.excluded_inputs) == 2
    assert result.trials == result.observations == ()
    report = json.loads((tmp_path / "authoring_report.json").read_text())
    assert report["counts"]["failed_inputs"] == 2
    assert report["counts"]["excluded_comparisons"] == 1
    assert report["counts"]["excluded_trials"] == 4
    assert report["comparisons"][0]["failed_input_indices"] == [0, 1]
    assert (tmp_path / "study/observations.jsonl").read_text() == ""


def test_rejected_warm_comparison_removes_stale_rows_but_preserves_raw_caches(
    tmp_path: Path,
) -> None:
    study = _studies()[1]
    messages, qa, _ = _seed(tmp_path, (study,), set())
    task = CollectionTask(study, tmp_path / "study")
    manual = _manual_library(tmp_path, study)
    transport = FakeTransport([*_assistant_responses(2), _judge_batch((True, False, False, True))])
    collect_studies(
        (task,),
        OpenRouterClient(transport),
        manual,
        messages,
        qa,
        prefer_batch=False,
        routing=CollectionRouting(allow_paid=True),
    )
    cache = SqliteStudyCache(task.output_dir / "collection.sqlite3")
    traces = cache.load_traces(build_trials(study))
    judgments = cache.load_judgments(build_trials(study))
    _seed(tmp_path, (study,), {study.inputs[0]})  # An explicit re-audit rejected this exact text.
    (tmp_path / "observations.jsonl").write_text("old aggregate")
    result = collect_studies((task,), None, manual, messages, qa, prefer_batch=False)[0]
    assert result.observations == ()
    assert not (tmp_path / "observations.jsonl").exists()
    assert (task.output_dir / "observations.jsonl").read_text() == ""
    assert cache.load_traces(build_trials(study)) == traces
    assert cache.load_judgments(build_trials(study)) == judgments


def test_report_survives_later_assistant_error(tmp_path: Path) -> None:
    studies = _studies()
    messages, qa, _ = _seed(tmp_path, studies, {studies[0].inputs[0]})
    tasks = tuple(CollectionTask(s, tmp_path / str(i)) for i, s in enumerate(studies))

    class BrokenTransport:
        def post_json(self, path, body):
            raise ValueError("transport failed")

        def get_json(self, path):
            raise AssertionError("unexpected GET")

    with pytest.raises(ValueError, match="transport failed"):
        collect_studies(
            tasks,
            OpenRouterClient(BrokenTransport()),
            _manual_library(tmp_path, *studies),
            messages,
            qa,
            prefer_batch=False,
            routing=CollectionRouting(allow_paid=True),
        )
    assert (
        json.loads((tmp_path / "authoring_report.json").read_text())["counts"][
            "excluded_comparisons"
        ]
        == 2
    )
    assert (tasks[0].output_dir / "observations.jsonl").read_text() == ""


@pytest.mark.parametrize("suite", [False, True])
def test_cli_reports_all_excluded_successfully_without_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    suite: bool,
) -> None:
    import yaml

    from reasonese.study import study_to_dict

    study = _studies()[0]
    root = tmp_path / "output"
    _seed(root, (study,), {study.inputs[0]})
    manual = _manual_library(tmp_path, study)
    source = tmp_path / "study.yaml"
    if suite:
        write_study_suite(source, (study,))
    else:
        source.write_text(yaml.safe_dump(study_to_dict(study)))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert (
        collect_cli(
            [
                "--suite" if suite else "--study",
                str(source),
                "--output",
                str(root),
                "--user-messages",
                str(manual.root),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["trials"] == summary["observations"] == 0
    assert summary["excluded_comparisons"] == 1
    assert summary["excluded_trials"] == 4
    assert Path(summary["authoring_report"]).is_file()
    if suite:
        assert (root / "observations.jsonl").read_text() == ""


def test_report_rejects_missing_extra_or_conflicting_verdicts(tmp_path: Path) -> None:
    studies = _studies()
    _, _, verdicts = _seed(tmp_path, studies, set())
    with pytest.raises(ValueError, match="match the planned inputs"):
        authoring_report(studies, verdicts[:1])
    with pytest.raises(ValueError, match="match the planned inputs"):
        authoring_report((studies[0],), verdicts)
    conflicting = replace(verdicts[0], content=GeneratedText.parse("different"))
    with pytest.raises(ValueError, match="conflicting"):
        authoring_report(studies, (*verdicts, conflicting))


def test_fresh_negative_qa_filters_before_any_assistant_request(tmp_path: Path) -> None:
    from tests.test_message_qa import _qa_batch

    study = _studies()[0]
    messages, qa, _ = _seed(tmp_path, (study,), set())
    qa.path.unlink()
    transport = FakeTransport([_qa_batch(((False, ["Changed task."]), (True, [])))])
    result = collect_studies(
        (CollectionTask(study, tmp_path / "study"),),
        OpenRouterClient(transport),
        _manual_library(tmp_path, study),
        messages,
        qa,
        prefer_batch=False,
        routing=CollectionRouting(allow_paid=True),
    )[0]
    assert result.trials == ()
    assert len(transport.post_calls) == 1
    assert transport.post_calls[0][0] == "/api/beta/batches"
    assert result.excluded_inputs[0].issues == ("Changed task.",)


def test_changed_text_requires_new_qa_and_can_restore_excluded_comparison(tmp_path: Path) -> None:
    from tests.test_study_orchestration import _message_qa_batch

    study = _studies()[1]
    messages, qa, _ = _seed(tmp_path, (study,), {study.inputs[0]})
    tasks = (CollectionTask(study, tmp_path / "study"),)
    manual = _manual_library(tmp_path, study)
    assert collect_studies(tasks, None, manual, messages, qa, prefer_batch=False)[0].excluded_inputs
    old = messages.load()[0]
    messages.put_many((replace(old, content=GeneratedText.parse("Corrected request.")),))
    with pytest.raises(ValueError, match="allow-paid"):
        collect_studies(tasks, None, manual, messages, qa, prefer_batch=False)
    transport = FakeTransport(
        [_message_qa_batch(1), *_assistant_responses(2), _judge_batch((True, False, False, True))]
    )
    restored = collect_studies(
        tasks,
        OpenRouterClient(transport),
        manual,
        messages,
        qa,
        prefer_batch=False,
        routing=CollectionRouting(allow_paid=True),
    )[0]
    assert restored.excluded_inputs == ()
    assert len(restored.trials) == 2
    assert (
        json.loads((tmp_path / "authoring_report.json").read_text())["counts"][
            "excluded_comparisons"
        ]
        == 0
    )


def test_malformed_qa_is_an_error_not_an_exclusion(tmp_path: Path) -> None:
    from tests.test_study_orchestration import _batch_result

    study = _studies()[0]
    messages, qa, _ = _seed(tmp_path, (study,), set())
    qa.path.unlink()
    transport = FakeTransport(
        [
            {
                "id": "qa",
                "status": "completed",
                "results": [
                    _batch_result(f"request-{i}", _chat("not JSON", f"qa-{i}")) for i in range(2)
                ],
            }
        ]
    )
    with pytest.raises(ValueError, match="not valid JSON"):
        collect_studies(
            (CollectionTask(study, tmp_path / "study"),),
            OpenRouterClient(transport),
            _manual_library(tmp_path, study),
            messages,
            qa,
            prefer_batch=False,
            routing=CollectionRouting(allow_paid=True),
        )
    assert not (tmp_path / "authoring_report.json").exists()
    assert len(transport.post_calls) == 1


def test_cold_authoring_still_requires_permission_and_key(tmp_path: Path) -> None:
    study = _studies()[0]
    tasks = (CollectionTask(study, tmp_path / "study"),)
    manual = _manual_library(tmp_path, study)
    messages = YamlMessageCache(tmp_path / "messages.yaml")
    qa = YamlMessageQaCache(tmp_path / "qa.yaml")
    transport = FakeTransport([])
    with pytest.raises(ValueError, match="allow-paid"):
        collect_studies(
            tasks, OpenRouterClient(transport), manual, messages, qa, prefer_batch=False
        )
    assert not transport.post_calls
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        collect_studies(
            tasks,
            None,
            manual,
            messages,
            qa,
            prefer_batch=False,
            routing=CollectionRouting(allow_paid=True),
        )


def test_cached_passing_qa_still_requires_permission_and_key_for_assistant(tmp_path: Path) -> None:
    study = _studies()[0]
    tasks = (CollectionTask(study, tmp_path / "study"),)
    messages, qa, _ = _seed(tmp_path, (study,), set())
    manual = _manual_library(tmp_path, study)
    with pytest.raises(ValueError, match="allow-paid"):
        collect_studies(tasks, None, manual, messages, qa, prefer_batch=False)
    with pytest.raises(ValueError, match="uncached conversation trials"):
        collect_studies(
            tasks,
            None,
            manual,
            messages,
            qa,
            prefer_batch=False,
            routing=CollectionRouting(allow_paid=True),
        )
    assert (
        json.loads((tmp_path / "authoring_report.json").read_text())["counts"][
            "excluded_comparisons"
        ]
        == 0
    )


def test_single_study_cli_reports_manual_qa_exclusion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import yaml

    from reasonese.collect_data import main as collect_one_cli
    from reasonese.study import study_to_dict
    from tests.test_study_orchestration import _study

    study = _study()
    root = tmp_path / "output"
    _seed(root, (study,), {study.inputs[0]})
    manual = _manual_library(tmp_path, study)
    source = tmp_path / "study.yaml"
    source.write_text(yaml.safe_dump(study_to_dict(study)))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert (
        collect_one_cli(
            ["--study", str(source), "--output", str(root), "--user-messages", str(manual.root)]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["excluded_comparisons"] == 1
    assert summary["excluded_trials"] == 2
    assert summary["observations"] == summary["trials"] == 0
    report = json.loads(Path(summary["authoring_report"]).read_text())
    assert report["inputs_by_axis"]["author"] == [
        {"author": "user", "inputs": 2, "failed_inputs": 1}
    ]
