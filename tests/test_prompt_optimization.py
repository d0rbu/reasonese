from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from phantom.interval import Natural

import reasonese.prompt_optimization as optimization
from reasonese.axes import Assistant, Author, Channel, Framing, Instruction
from reasonese.check_messages import MessageQaRunResult
from reasonese.conversation import (
    AUTHORING_BRIEFS,
    BASELINE_AUTHORING_BRIEF,
    REASONESE_NATURAL_AUTHORING_BRIEF,
    SEMANTIC_PRESERVATION_AUTHORING_BRIEF,
    GeneratedMessage,
    GeneratedText,
    authoring_instructions,
    authoring_request,
)
from reasonese.manual_messages import ManualMessageLibrary
from reasonese.message_qa import (
    MessageQaVerdict,
    QaIssue,
    message_qa_request,
    message_qa_rubric_fingerprint,
)
from reasonese.planning import PromptSpec
from reasonese.probe_qa import ProbeExpectation, ProbeQaRequest, ProbeQaVerdict, probe_expectation
from reasonese.routing import CollectionRouting
from reasonese.study import Study, make_study


def _spec(number: int, framing: Framing, channel: Channel) -> PromptSpec:
    return PromptSpec(
        Instruction.parse(f"Perform bounded task {number} and return its result."),
        framing,
        channel,
        Author.NEMOTRON_3_5_LIGHTNING,
    )


def _studies() -> tuple[Study, ...]:
    studies = []
    for number, framing in enumerate(Framing):
        anchor_framing = Framing.CASUAL if framing is Framing.NORMAL else Framing.NORMAL
        studies.append(
            make_study(
                (_spec(number * 2, framing, Channel.USER),
                 _spec(number * 2 + 1, anchor_framing, Channel.SYSTEM)),
                Assistant.NEMOTRON_3_5_LIGHTNING,
                1,
            )
        )
    return tuple(studies)


def _pair_ids(studies: tuple[Study, ...]) -> dict[str, str]:
    return {str(spec.instruction): "pilot-pair" for study in studies for spec in study.inputs}


def _pair_identity() -> dict[str, object]:
    return {"path": "/tmp/pairs.yaml", "sha256": "pair-sha", "pairs": []}


def _message(spec: PromptSpec, suffix: str = "") -> GeneratedMessage:
    return GeneratedMessage(spec, GeneratedText.parse(f"I will perform task {suffix or spec.instruction}."), {})


class _Probe:
    def __init__(self, *, compressed_descriptive: bool = True) -> None:
        self.preflighted: tuple[Assistant, ...] = ()
        self.requests: tuple[ProbeQaRequest, ...] = ()
        self.compressed_descriptive = compressed_descriptive

    def preflight(self, assistants: tuple[Assistant, ...]) -> None:
        self.preflighted = assistants

    def check(self, requests: tuple[ProbeQaRequest, ...]) -> tuple[ProbeQaVerdict, ...]:
        self.requests = requests
        verdicts = []
        for index, request in enumerate(requests):
            expectation = probe_expectation(request.spec)
            if expectation is ProbeExpectation.REASONING:
                probability = 0.9
                complies: bool | None = True
            elif expectation is ProbeExpectation.NONREASONING:
                probability = 0.1
                complies = True
            elif self.compressed_descriptive:
                probability = 0.5
                complies = None
            else:
                probability = 0.5
                complies = None
            probabilities = (
                ("system", 0.025),
                ("user", 0.025),
                ("tool", 0.025),
                ("reasoning", probability),
                ("assistant", 0.925 - probability),
            )
            verdicts.append(
                ProbeQaVerdict(
                    request,
                    f"context-{index}",
                    probabilities,
                    probability,
                    expectation,
                    complies,
                    None,
                )
            )
        return tuple(verdicts)


def _fake_materialize(specs, client, cache, manual, *, prefer_batch, routing, authoring_brief):
    return tuple(_message(spec, authoring_brief.name) for spec in specs)


def _fake_audit(messages, qa_cache, client, *, routing, prefer_batch=True):
    verdicts = tuple(
        MessageQaVerdict(message.spec, message.content, True, (), {"id": "qa"})
        for message in messages
    )
    return MessageQaRunResult(verdicts, Natural.parse(0))


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    brief=BASELINE_AUTHORING_BRIEF,
    audit=_fake_audit,
    probe: _Probe | None = None,
):
    studies = _studies()
    monkeypatch.setattr(optimization, "materialize_specs", _fake_materialize)
    monkeypatch.setattr(optimization, "audit_messages", audit)
    scorer = probe or _Probe()
    return optimization.evaluate_prompt_brief(
        studies,
        brief=brief,
        output=tmp_path,
        client=None,
        manual_messages=ManualMessageLibrary(tmp_path / "manual"),
        probe_scorer=scorer,
        assistant=Assistant.NEMOTRON_3_5_LIGHTNING,
        authors=(Author.NEMOTRON_3_5_LIGHTNING,),
        pair_ids=_pair_ids(studies),
        pair_identity=_pair_identity(),
        probe_identity={"config": "probe"},
        routing=CollectionRouting(allow_paid=True),
    )


def test_briefs_are_immutable_and_candidate_does_not_change_controls() -> None:
    spec = _spec(1, Framing.REASONESE_NORMAL, Channel.USER)
    assert authoring_instructions(spec, brief=BASELINE_AUTHORING_BRIEF) == authoring_instructions(spec)
    assert authoring_instructions(spec, brief=REASONESE_NATURAL_AUTHORING_BRIEF) != authoring_instructions(spec)
    assert authoring_request(_spec(2, Framing.COMPRESSED_NORMAL, Channel.USER), brief=REASONESE_NATURAL_AUTHORING_BRIEF) == authoring_request(_spec(2, Framing.COMPRESSED_NORMAL, Channel.USER))
    assert REASONESE_NATURAL_AUTHORING_BRIEF.fingerprint == REASONESE_NATURAL_AUTHORING_BRIEF.fingerprint
    assert set(AUTHORING_BRIEFS) == {
        "baseline",
        "reasonese-natural-v1",
        "semantic-preservation-v2",
    }
    with pytest.raises(ValueError, match="name"):
        type(BASELINE_AUTHORING_BRIEF)("", "")
    message = _message(spec)
    baseline_qa = message_qa_request(message)
    candidate_author = authoring_request(spec, brief=REASONESE_NATURAL_AUTHORING_BRIEF)
    candidate_qa = message_qa_request(message)
    assert baseline_qa == candidate_qa
    assert message_qa_rubric_fingerprint()
    assert candidate_author != baseline_qa


def test_semantic_preservation_v2_covers_all_framings_and_keeps_qa_fixed() -> None:
    candidate = SEMANTIC_PRESERVATION_AUTHORING_BRIEF
    assert candidate.framings == tuple(Framing)
    for number, framing in enumerate(Framing):
        spec = _spec(100 + number, framing, Channel.USER)
        baseline_request = authoring_request(spec, brief=BASELINE_AUTHORING_BRIEF)
        candidate_request = authoring_request(spec, brief=candidate)
        assert baseline_request != candidate_request
        assert candidate.guidance in candidate_request["messages"][0]["content"]

        qa_evidence = json.loads(message_qa_request(_message(spec))["messages"][1]["content"])
        assert qa_evidence["exact_authoring_instructions"] == authoring_instructions(spec)
        assert candidate.guidance not in qa_evidence["exact_authoring_instructions"]


@pytest.mark.parametrize(
    "change",
    [
        lambda studies: studies[:-1],
        lambda studies: tuple(
            make_study(studies[0].inputs, studies[0].assistant, 2) for _ in (0,)
        ) + studies[1:],
    ],
)
def test_suite_validation_rejects_missing_framing_or_multiple_rollouts(change) -> None:
    studies = change(_studies())
    with pytest.raises(ValueError):
        optimization._validate_suite(
            studies,
            assistant=Assistant.NEMOTRON_3_5_LIGHTNING,
            authors=(Author.NEMOTRON_3_5_LIGHTNING,),
        )


def test_suite_validation_enforces_author_assistant_and_work_limits() -> None:
    studies = _studies()
    common = {"assistant": Assistant.NEMOTRON_3_5_LIGHTNING, "authors": (Author.NEMOTRON_3_5_LIGHTNING,)}
    with pytest.raises(ValueError, match="at least one"):
        optimization._validate_suite((), **common)
    with pytest.raises(ValueError, match="unique"):
        optimization._validate_suite(studies, authors=(Author.NEMOTRON_3_5_LIGHTNING,) * 2, assistant=common["assistant"])
    with pytest.raises(ValueError, match="model authors"):
        optimization._validate_suite(studies, authors=(Author.GEMMA_4_31B_IT,), assistant=common["assistant"])
    with pytest.raises(ValueError, match="selected"):
        optimization._validate_suite(studies, authors=common["authors"], assistant=Assistant.GEMMA_4_31B_IT)
    with pytest.raises(ValueError, match="authors"):
        optimization._validate_suite(studies, authors=(Author.USER,), assistant=common["assistant"])
    oversized = studies + tuple(
        make_study(
            (_spec(1000 + index * 2, Framing.NORMAL, Channel.USER),
             _spec(1001 + index * 2, Framing.CASUAL, Channel.SYSTEM)),
            Assistant.NEMOTRON_3_5_LIGHTNING,
            1,
        )
        for index in range(25)
    )
    with pytest.raises(ValueError, match="at most"):
        optimization._validate_suite(oversized, **common)


def test_identity_and_pair_binding_fail_closed(tmp_path: Path) -> None:
    with pytest.raises((OSError, ValueError)):
        optimization._probe_identity(tmp_path / "missing.json")
    pair_path = tmp_path / "pairs.yaml"
    pair_path.write_text(
        "pairs:\n"
        "  - id: one\n"
        "    skill: python\n"
        "    conflict: content\n"
        "    first: First exact task.\n"
        "    second: Second exact task.\n"
        "    rationale: They conflict.\n"
    )
    mapping, identity = optimization._pair_identity(pair_path)
    assert mapping == {"First exact task.": "one", "Second exact task.": "one"}
    assert identity["sha256"] == optimization._sha256(pair_path)
    with pytest.raises(ValueError, match="missing"):
        optimization._bind_pair_ids((_spec(1, Framing.NORMAL, Channel.USER),), {})
    with pytest.raises(ValueError, match="exact instruction pair"):
        optimization._bind_pair_ids((_spec(1, Framing.NORMAL, Channel.USER),), None)

    duplicate_pair_path = tmp_path / "duplicate-pairs.yaml"
    duplicate_pair_path.write_text(
        "pairs:\n"
        "  - id: one\n"
        "    skill: python\n"
        "    conflict: content\n"
        "    first: First exact task.\n"
        "    second: Second exact task.\n"
        "    rationale: They conflict.\n"
        "  - id: two\n"
        "    skill: python\n"
        "    conflict: content\n"
        "    first: First exact task.\n"
        "    second: Third exact task.\n"
        "    rationale: They conflict.\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="reused"):
        optimization._pair_identity(duplicate_pair_path)


def test_evaluation_records_candidate_requests_probe_orders_and_no_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = _Probe()
    summary = _run(tmp_path, monkeypatch, brief=REASONESE_NATURAL_AUTHORING_BRIEF, probe=probe)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    requests = json.loads((tmp_path / "authoring_requests.json").read_text())
    report = json.loads((tmp_path / "probe_qa_report.json").read_text())
    assert summary["message_qa"]["messages"] == 16
    assert manifest["candidate_fingerprint"] == REASONESE_NATURAL_AUTHORING_BRIEF.fingerprint
    assert manifest["work"]["probe_qa_requests"] == 32
    assert len(requests["requests"]) == 16
    assert any(
        "ordinary first-person note" in item["request"]["messages"][0]["content"]
        for item in requests["requests"]
    )
    assert len(probe.requests) == 32
    assert probe.preflighted == (Assistant.NEMOTRON_3_5_LIGHTNING,)
    assert report["counts"]["scores"] == 32
    assert report["counts"]["descriptive_scores"] == 4
    assert report["combined_eligible_studies"] == 8
    assert manifest["stages"]["assistant_execution"] is False
    assert manifest["stages"]["response_judging"] is False
    assert manifest["stages"]["tool_calls"] is False


def test_probe_still_scores_message_qa_failures_and_reports_combined_exclusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def one_failed_audit(messages, qa_cache, client, *, routing, prefer_batch=True):
        verdicts = tuple(
            MessageQaVerdict(
                message.spec,
                message.content,
                index != 0,
                () if index else (QaIssue.parse("failed authoring"),),
                {"id": f"qa-{index}"},
            )
            for index, message in enumerate(messages)
        )
        return MessageQaRunResult(verdicts, Natural.parse(0))

    probe = _Probe()
    _run(tmp_path, monkeypatch, audit=one_failed_audit, probe=probe)
    report = json.loads((tmp_path / "probe_qa_report.json").read_text())
    assert len(probe.requests) == 32
    assert report["message_qa_excluded_studies"]
    assert report["combined_eligible_studies"] < 8
    assert report["counts"]["scores"] == 32


def test_probe_report_keeps_reused_anchor_contexts_distinct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    anchor = _spec(999, Framing.NORMAL, Channel.SYSTEM)
    studies = tuple(
        make_study(
            (_spec(index, framing, Channel.USER), anchor),
            Assistant.NEMOTRON_3_5_LIGHTNING,
            1,
        )
        for index, framing in enumerate(Framing)
    )
    monkeypatch.setattr(optimization, "materialize_specs", _fake_materialize)
    monkeypatch.setattr(optimization, "audit_messages", _fake_audit)
    optimization.evaluate_prompt_brief(
        studies,
        brief=BASELINE_AUTHORING_BRIEF,
        output=tmp_path,
        client=None,
        manual_messages=ManualMessageLibrary(tmp_path / "manual"),
        probe_scorer=_Probe(),
        assistant=Assistant.NEMOTRON_3_5_LIGHTNING,
        authors=(Author.NEMOTRON_3_5_LIGHTNING,),
        pair_ids={str(spec.instruction): "pilot-pair" for study in studies for spec in study.inputs},
        routing=CollectionRouting(allow_paid=True),
    )
    report = json.loads((tmp_path / "probe_qa_report.json").read_text())
    assert len(report["scores"]) == 32


def test_paid_guard_runs_before_materialization_and_failure_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def unexpected(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("provider stage reached before paid guard")

    monkeypatch.setattr(optimization, "materialize_specs", unexpected)
    with pytest.raises(ValueError, match="allow-paid"):
        optimization.evaluate_prompt_brief(
            _studies(),
            brief=BASELINE_AUTHORING_BRIEF,
            output=tmp_path,
            client=None,
            manual_messages=ManualMessageLibrary(tmp_path / "manual"),
            probe_scorer=_Probe(),
            assistant=Assistant.NEMOTRON_3_5_LIGHTNING,
            authors=(Author.NEMOTRON_3_5_LIGHTNING,),
            pair_ids=_pair_ids(_studies()),
            routing=CollectionRouting(allow_paid=False),
        )
    assert not called
    failures = (tmp_path / "failures.jsonl").read_text()
    assert "allow-paid" in failures
    assert json.loads((tmp_path / "manifest.json").read_text())["status"] == "failed"


def test_probe_preflight_runs_before_materialization_and_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailingProbe(_Probe):
        def preflight(self, assistants: tuple[Assistant, ...]) -> None:
            raise RuntimeError("invalid probe bundle")

    called = False

    def unexpected(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("authoring must wait for probe preflight")

    monkeypatch.setattr(optimization, "materialize_specs", unexpected)
    with pytest.raises(RuntimeError, match="invalid probe"):
        _run(tmp_path, monkeypatch, probe=FailingProbe())
    assert not called
    assert "probe_preflight" in (tmp_path / "failures.jsonl").read_text()


def test_candidate_directory_must_be_fresh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "old.txt").write_text("old")
    with pytest.raises(ValueError, match="fresh"):
        _run(tmp_path, monkeypatch)

    file_output = tmp_path / "file-output"
    file_output.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ValueError, match="not a directory"):
        _run(file_output, monkeypatch)


def test_compare_outputs_reports_pair_and_judge_denominators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    _run(baseline_dir, monkeypatch, brief=BASELINE_AUTHORING_BRIEF)

    def failed_audit(messages, qa_cache, client, *, routing, prefer_batch=True):
        verdicts = tuple(
            MessageQaVerdict(
                message.spec,
                message.content,
                index != 0,
                () if index else (QaIssue.parse("candidate failure"),),
                {"id": f"qa-{index}"},
            )
            for index, message in enumerate(messages)
        )
        return MessageQaRunResult(verdicts, Natural.parse(0))

    _run(candidate_dir, monkeypatch, brief=REASONESE_NATURAL_AUTHORING_BRIEF, audit=failed_audit)
    comparison = optimization.compare_prompt_outputs(baseline_dir, candidate_dir)
    rows = cast(list[dict[str, object]], comparison["rows"])
    assert len(rows) == 2
    qa = next(row for row in rows if cast(str, row["judge"]).startswith("message-QA"))
    qa_baseline = cast(dict[str, int], qa["baseline"])
    assert qa_baseline["pass_numerator"] == 16
    qa_candidate = cast(dict[str, int], qa["candidate"])
    assert qa_candidate["pass_numerator"] == 15
    assert qa_baseline["pass_denominator"] == 16
    markdown = cast(str, comparison["markdown"])
    assert "passed/eligible" in markdown
    assert "16/16 (100.0%)" in markdown
    assert "15/16 (93.8%)" in markdown
    assert "| Overall |" in markdown
    overall = cast(dict[str, object], comparison["overall"])
    overall_baseline = cast(dict[str, object], overall["baseline"])
    overall_probe = cast(dict[str, int], overall_baseline["probe"])
    assert overall_probe["descriptive"] == 4
    descriptive = cast(dict[str, object], comparison["descriptive_probe"])
    assert descriptive["baseline"] == {"pilot-pair": [0.5] * 4}
    assert comparison["descriptive_probe_counts"] == {
        "baseline": {"pilot-pair": 4},
        "candidate": {"pilot-pair": 4},
    }


def test_comparison_markdown_uses_na_for_empty_pass_denominator() -> None:
    empty = {"pass_numerator": 0, "pass_denominator": 0, "missing": 0, "descriptive": 0}
    markdown = optimization._format_comparison_markdown(
        [],
        "baseline",
        "candidate",
        {
            "baseline": {"message_qa": empty, "probe": empty},
            "candidate": {"message_qa": empty, "probe": empty},
        },
    )

    assert markdown.count("0/0 (N/A)") == 4


def test_compare_rejects_mismatched_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _run(first, monkeypatch)
    _run(second, monkeypatch, brief=REASONESE_NATURAL_AUTHORING_BRIEF)
    manifest_path = second / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["message_qa"]["rubric_sha256"] = "changed"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="rubrics"):
        optimization.compare_prompt_outputs(first, second)


def test_validation_and_pair_annotation_fail_closed(tmp_path: Path) -> None:
    studies = _studies()
    with pytest.raises(ValueError, match="distinct"):
        optimization._validate_suite(
            (studies[0], studies[0]),
            assistant=Assistant.NEMOTRON_3_5_LIGHTNING,
            authors=(Author.NEMOTRON_3_5_LIGHTNING,),
        )
    spec = _spec(1, Framing.NORMAL, Channel.USER)
    with pytest.raises(ValueError, match="non-empty"):
        optimization._bind_pair_ids((spec,), {str(spec.instruction): ""})

    report = {
        "inputs": [{"instruction": "a"}, {}, "ignored"],
        "scores": [{"instruction": "a"}, {}, "ignored"],
        "comparisons": [
            {"inputs": [{"instruction": "a"}, {}, "ignored"]},
            {"inputs": "ignored"},
            "ignored",
        ],
    }
    annotated = optimization._add_pair_ids(cast(dict[str, object], report), {"a": "pair"})
    assert cast(dict[str, object], cast(list[object], annotated["comparisons"])[0])["pair_ids"] == [
        "pair"
    ]


def test_study_cannot_mix_instruction_pair_ids() -> None:
    studies = _studies()
    first, second = studies[0].inputs
    with pytest.raises(ValueError, match="one instruction pair"):
        optimization._validate_study_pair_ids(
            (studies[0],),
            {str(first.instruction): "pair-a", str(second.instruction): "pair-b"},
        )


def test_probe_identity_records_config_and_probe_digests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import reasonese.local_probe_qa as local_probe_qa

    config = tmp_path / "bundles.json"
    probe = tmp_path / "probe.npz"
    config.write_text("{}", encoding="utf-8")
    probe.write_bytes(b"probe")
    bundle = SimpleNamespace(
        assistant=Assistant.NEMOTRON_3_5_LIGHTNING,
        adapter=SimpleNamespace(name="role-probe"),
        checkpoint=tmp_path / "checkpoint",
        probe_path=probe,
    )
    monkeypatch.setattr(local_probe_qa, "load_probe_bundles", lambda path: (bundle,))
    identity = optimization._probe_identity(config)
    assert identity["config"] == {
        "path": str(config.resolve()),
        "sha256": optimization._sha256(config),
    }
    bundles = cast(list[dict[str, object]], identity["bundles"])
    assert bundles[0]["probe_sha256"] == optimization._sha256(probe)
    assert identity["capture_policy"] == local_probe_qa.CAPTURE_POLICY


def test_json_and_judge_helpers_reject_malformed_rows(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="invalid"):
        optimization._read_json(missing)
    invalid = tmp_path / "invalid.json"
    invalid.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="object"):
        optimization._read_json(invalid)

    report = tmp_path / "authoring_report.json"
    report.write_text(json.dumps({"inputs": "bad"}), encoding="utf-8")
    with pytest.raises(ValueError, match="list"):
        optimization._judge_rows(tmp_path, "authoring_report.json", "inputs")
    report.write_text(json.dumps({"inputs": ["bad"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="non-object"):
        optimization._judge_rows(tmp_path, "authoring_report.json", "inputs")

    row = {"pair_id": "pair", "instruction": "a", "complies": True}
    report.write_text(json.dumps({"inputs": [row, row]}), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        optimization._judge_rows(tmp_path, "authoring_report.json", "inputs")
    for bad_row, message in (
        ({"instruction": "a"}, "pair_id"),
        ({"pair_id": "pair"}, "instruction"),
    ):
        report.write_text(json.dumps({"inputs": [bad_row]}), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            optimization._judge_rows(tmp_path, "authoring_report.json", "inputs")


def test_judge_totals_separate_missing_and_descriptive_values() -> None:
    expected = {("pair", "one"), ("pair", "two")}
    totals = optimization._judge_totals(
        {("pair", "one"): True, ("pair", "other"): None},
        "pair",
        expected,
        allow_descriptive=True,
    )
    assert totals == {
        "pass_numerator": 1,
        "pass_denominator": 1,
        "missing": 1,
        "descriptive": 1,
    }
    with pytest.raises(ValueError, match="boolean"):
        optimization._judge_totals(
            {("pair", "one"): None}, "pair", expected, allow_descriptive=False
        )
    with pytest.raises(ValueError, match="boolean"):
        optimization._judge_totals(
            {("pair", "one"): "yes"}, "pair", expected, allow_descriptive=True
        )


def _write_expected_probe_report(root: Path, **changes: object) -> None:
    first = optimization.prompt_spec_to_dict(_spec(1, Framing.NORMAL, Channel.USER))
    second = optimization.prompt_spec_to_dict(_spec(2, Framing.CASUAL, Channel.SYSTEM))
    comparison: dict[str, object] = {
        "study_id": "study",
        "inputs": [first, second],
        "input_pair_ids": ["pair", "pair"],
    }
    comparison.update(changes)
    (root / "authoring_report.json").write_text(
        json.dumps({"comparisons": [comparison]}), encoding="utf-8"
    )


def test_expected_probe_coordinates_are_order_and_position_complete(tmp_path: Path) -> None:
    _write_expected_probe_report(tmp_path)
    expected = optimization._expected_probe_rows(tmp_path)
    assert len(expected) == 4
    coordinates = [json.loads(key) for _, key in expected]
    assert {(row["permutation"], row["position"]) for row in coordinates} == {
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
    }


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({}, "comparisons"),
        (None, "objects"),
        ({"inputs": []}, "two inputs"),
        ({"input_pair_ids": ["pair"]}, "two pair IDs"),
        ({"study_id": ""}, "study ID"),
        ({"inputs": ["bad", {"instruction": "b"}]}, "input is invalid"),
        ({"input_pair_ids": ["", "pair"]}, "bind an input pair ID"),
    ],
)
def test_expected_probe_coordinates_reject_malformed_reports(
    tmp_path: Path, changes: object, message: str
) -> None:
    if changes == {}:
        (tmp_path / "authoring_report.json").write_text("{}", encoding="utf-8")
    elif changes is None:
        (tmp_path / "authoring_report.json").write_text(
            json.dumps({"comparisons": ["bad"]}), encoding="utf-8"
        )
    else:
        _write_expected_probe_report(tmp_path, **cast(dict[str, object], changes))
    with pytest.raises(ValueError, match=message):
        optimization._expected_probe_rows(tmp_path)

    _write_expected_probe_report(tmp_path)
    with (tmp_path / "authoring_report.json").open("r+", encoding="utf-8") as handle:
        report = json.load(handle)
        report["comparisons"].append(report["comparisons"][0])
        handle.seek(0)
        handle.truncate()
        json.dump(report, handle)
    with pytest.raises(ValueError, match="duplicate"):
        optimization._expected_probe_rows(tmp_path)


def test_evaluation_records_probe_report_failures_and_setup_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_build_trials = optimization.build_trials
    monkeypatch.setattr(
        optimization,
        "build_trials",
        lambda study: original_build_trials(study)[:1],
    )
    with pytest.raises(ValueError, match="ordered setups"):
        _run(tmp_path / "setup", monkeypatch)
    assert "exactly two ordered setups" in (tmp_path / "setup" / "failures.jsonl").read_text()
    monkeypatch.setattr(optimization, "build_trials", original_build_trials)

    def missing_comparisons(*args, **kwargs):
        return {"counts": {}, "scores": []}

    monkeypatch.setattr(optimization, "probe_qa_report", missing_comparisons)
    with pytest.raises(ValueError, match="comparison rows"):
        _run(tmp_path / "report", monkeypatch)
    assert "probe report lacks comparison rows" in (
        tmp_path / "report" / "failures.jsonl"
    ).read_text()


@pytest.mark.parametrize("field", ["suite", "assistant", "routing", "probe", "instruction_pairs"])
def test_compare_rejects_changed_top_level_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _run(first, monkeypatch)
    _run(second, monkeypatch, brief=REASONESE_NATURAL_AUTHORING_BRIEF)
    manifest_path = second / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = {"changed": True}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=field):
        optimization.compare_prompt_outputs(first, second)


def test_compare_rejects_changed_probe_capture_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _run(first, monkeypatch)
    _run(second, monkeypatch, brief=REASONESE_NATURAL_AUTHORING_BRIEF)
    manifest_path = second / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["probe"]["capture_policy"] = "legacy_full_context_v0"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="probe identities differ"):
        optimization.compare_prompt_outputs(first, second)


def test_compare_rejects_incomplete_or_ambiguous_candidate_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first"
    _run(first, monkeypatch)
    for name, update, message in (
        ("status", {"status": "running"}, "completed"),
        ("qa-shape", {"message_qa": {}}, "message-QA"),
        ("candidate-shape", {"candidate": "bad"}, "candidate identities"),
        ("candidate-name", {"candidate": {"name": 1}}, "candidates must have names"),
        ("same-name", {"candidate": {"name": "baseline"}}, "names must differ"),
    ):
        other = tmp_path / name
        _run(other, monkeypatch, brief=REASONESE_NATURAL_AUTHORING_BRIEF)
        manifest_path = other / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.update(update)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            optimization.compare_prompt_outputs(first, other)


def test_compare_rejects_malformed_reports_and_coordinate_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first"
    _run(first, monkeypatch)

    malformed_qa = tmp_path / "malformed-qa"
    _run(malformed_qa, monkeypatch, brief=REASONESE_NATURAL_AUTHORING_BRIEF)
    manifest_path = malformed_qa / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["message_qa"] = "bad"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="message-QA identity"):
        optimization.compare_prompt_outputs(first, malformed_qa)

    mismatch_qa = tmp_path / "mismatch-qa"
    _run(mismatch_qa, monkeypatch, brief=REASONESE_NATURAL_AUTHORING_BRIEF)
    report_path = mismatch_qa / "authoring_report.json"
    report = json.loads(report_path.read_text())
    report["inputs"].pop()
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="message-QA report coordinates"):
        optimization.compare_prompt_outputs(first, mismatch_qa)

    planned_qa_mismatch = tmp_path / "planned-qa-mismatch"
    _run(planned_qa_mismatch, monkeypatch, brief=REASONESE_NATURAL_AUTHORING_BRIEF)
    planned_qa_path = planned_qa_mismatch / "authoring_report.json"
    planned_qa = json.loads(planned_qa_path.read_text())
    planned_qa["comparisons"][0]["input_pair_ids"][0] = "different-pair"
    planned_qa_path.write_text(json.dumps(planned_qa), encoding="utf-8")
    with pytest.raises(ValueError, match="message-QA coordinates differ"):
        optimization.compare_prompt_outputs(first, planned_qa_mismatch)

    mismatch_probe = tmp_path / "mismatch-probe"
    _run(mismatch_probe, monkeypatch, brief=REASONESE_NATURAL_AUTHORING_BRIEF)
    probe_path = mismatch_probe / "probe_qa_report.json"
    probe_report = json.loads(probe_path.read_text())
    probe_report["scores"].pop()
    probe_path.write_text(json.dumps(probe_report), encoding="utf-8")
    with pytest.raises(ValueError, match="probe-QA report coordinates"):
        optimization.compare_prompt_outputs(first, mismatch_probe)

    planned_probe_mismatch = tmp_path / "planned-probe-mismatch"
    _run(planned_probe_mismatch, monkeypatch, brief=REASONESE_NATURAL_AUTHORING_BRIEF)
    planned_probe_path = planned_probe_mismatch / "authoring_report.json"
    planned_probe = json.loads(planned_probe_path.read_text())
    planned_probe["comparisons"][0]["study_id"] = "different-study"
    planned_probe_path.write_text(json.dumps(planned_probe), encoding="utf-8")
    with pytest.raises(ValueError, match="probe-QA coordinates differ"):
        optimization.compare_prompt_outputs(first, planned_probe_mismatch)


def test_compare_main_writes_offline_comparison(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _run(baseline, monkeypatch)
    _run(candidate, monkeypatch, brief=REASONESE_NATURAL_AUTHORING_BRIEF)
    output = tmp_path / "comparison.json"
    assert optimization.compare_main(
        ["--baseline", str(baseline), "--candidate", str(candidate), "--output", str(output)]
    ) == 0
    result = json.loads(output.read_text())
    assert "Overall" in result["markdown"]


def test_optimize_main_reports_missing_suite_without_provider_imports(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        optimization.main(
            [
                "--pairs",
                str(tmp_path / "pairs.yaml"),
                "--suite",
                str(tmp_path / "missing-suite.yaml"),
                "--output",
                str(tmp_path / "out"),
                "--brief",
                "baseline",
                "--role-probes",
                str(tmp_path / "probe-bundles.json"),
            ]
        )


def test_optimize_main_wires_bound_identity_and_selected_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import reasonese.local_probe_qa as local_probe_qa

    studies = _studies()
    pair_ids = _pair_ids(studies)
    monkeypatch.setattr(optimization, "load_study_suite", lambda path: studies)
    monkeypatch.setattr(
        optimization,
        "_pair_identity",
        lambda path: (pair_ids, {"path": str(path)}),
    )
    monkeypatch.setattr(optimization, "_probe_identity", lambda path: {"path": str(path)})
    monkeypatch.setattr(
        optimization,
        "evaluate_prompt_brief",
        lambda studies, **kwargs: {"candidate": kwargs["brief"].name},
    )

    class FakeScorer:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    monkeypatch.setattr(local_probe_qa, "LocalProbeQaScorer", FakeScorer)
    assert optimization.main(
        [
            "--pairs",
            str(tmp_path / "pairs.yaml"),
            "--suite",
            str(tmp_path / "suite.yaml"),
            "--output",
            str(tmp_path / "out"),
            "--brief",
            "reasonese-natural-v1",
            "--role-probes",
            str(tmp_path / "probe-bundles.json"),
            "--author",
            str(Author.NEMOTRON_3_5_LIGHTNING),
            "--assistant",
            str(Assistant.NEMOTRON_3_5_LIGHTNING),
        ]
    ) == 0


def test_compare_main_reports_invalid_input(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        optimization.compare_main(
            ["--baseline", str(tmp_path / "missing-a"), "--candidate", str(tmp_path / "missing-b")]
        )
