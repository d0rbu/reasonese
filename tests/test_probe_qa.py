"""Probe-QA request, verdict, and attrition-report contracts."""

from __future__ import annotations

from dataclasses import replace

import pytest

from reasonese.axes import Assistant, Author, Channel, Framing, Instruction
from reasonese.conversation import GeneratedMessage, GeneratedText, construct_conversation
from reasonese.planning import PromptSpec
from reasonese.probe_qa import (
    ProbeExpectation,
    ProbeQaDiagnosticIssue,
    ProbeQaIssueKind,
    ProbeQaRequest,
    ProbeQaVerdict,
    check_probe_qa,
    probe_expectation,
    probe_qa_report,
    probe_qa_requests,
    resolve_probe_mode,
)
from reasonese.study import Study, build_trials, make_study


def _study(first_framing: Framing = Framing.REASONESE_NORMAL):
    first = PromptSpec(
        Instruction.parse("Write a small Python program."),
        first_framing,
        Channel.USER,
        Author.NEMOTRON_3_5_LIGHTNING,
    )
    second = PromptSpec(
        Instruction.parse("Explain the algorithm in prose."),
        Framing.NORMAL,
        Channel.SYSTEM,
        Author.GEMMA_4_31B_IT,
    )
    return make_study((first, second), Assistant.NEMOTRON_3_5_LIGHTNING, 3)


def _requests(
    first_framing: Framing = Framing.REASONESE_NORMAL,
) -> tuple[Study, tuple[ProbeQaRequest, ...]]:
    study = _study(first_framing)
    contents = {
        study.inputs[0]: GeneratedMessage(
            study.inputs[0], GeneratedText.parse("I need to write the requested program."), None
        ),
        study.inputs[1]: GeneratedMessage(
            study.inputs[1], GeneratedText.parse("Explain the algorithm in clear prose."), None
        ),
    }
    first_trials = tuple(trial for trial in build_trials(study) if int(trial.rollout) == 1)
    setups = tuple(
        construct_conversation(
            trial.matchup,
            tuple(contents[spec] for spec in trial.matchup.inputs),
        )
        for trial in first_trials
    )
    assert len(setups) == 2
    return study, probe_qa_requests(study, (setups[0], setups[1]))


def _verdict(
    request: ProbeQaRequest,
    *,
    complies: bool | None = True,
    reasoning: float | None = None,
) -> ProbeQaVerdict:
    expectation = probe_expectation(request.spec)
    if reasoning is None:
        reasoning = 0.9 if expectation is ProbeExpectation.REASONING else 0.1
    issue = "reasoning probability missed the preregistered gate" if complies is False else None
    if expectation is ProbeExpectation.DESCRIPTIVE:
        complies = None
        issue = None
    return ProbeQaVerdict(
        request=request,
        context_fingerprint=f"context-{request.permutation}-{request.position}",
        role_probabilities=(
            ("system", 0.01),
            ("user", 0.03),
            ("tool", 0.01),
            ("reasoning", reasoning),
            ("assistant", 0.95 - reasoning),
        ),
        reasoning_probability=reasoning,
        expectation=expectation,
        complies=complies,
        issue=issue,
    )


@pytest.mark.parametrize(
    ("framing", "expected"),
    [
        (Framing.NORMAL, ProbeExpectation.NONREASONING),
        (Framing.CASUAL, ProbeExpectation.NONREASONING),
        (Framing.PERSUASIVE, ProbeExpectation.NONREASONING),
        (Framing.SUBAGENT, ProbeExpectation.NONREASONING),
        (Framing.REASONESE_NORMAL, ProbeExpectation.REASONING),
        (Framing.REASONESE_PERSUASIVE, ProbeExpectation.REASONING),
        (Framing.COMPRESSED_NORMAL, ProbeExpectation.DESCRIPTIVE),
        (Framing.COMPRESSED_PERSUASIVE, ProbeExpectation.DESCRIPTIVE),
    ],
)
def test_framing_probe_expectations_are_explicit(
    framing: Framing, expected: ProbeExpectation
) -> None:
    spec = _study(framing).inputs[0]
    assert probe_expectation(spec) is expected


def test_requests_cover_both_inputs_in_both_exact_orders() -> None:
    study, requests = _requests()
    assert [(row.permutation, row.position) for row in requests] == [
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
    ]
    assert [row.spec for row in requests] == [
        study.inputs[0],
        study.inputs[1],
        study.inputs[1],
        study.inputs[0],
    ]
    assert requests[0].setup.messages != requests[2].setup.messages


def test_request_builder_rejects_wrong_ordered_contexts() -> None:
    study, requests = _requests()
    with pytest.raises(ValueError, match="ordered permutations"):
        probe_qa_requests(study, (requests[2].setup, requests[0].setup))


def test_probe_request_rejects_invalid_coordinates_and_empty_study_identity() -> None:
    _, requests = _requests()
    with pytest.raises(ValueError, match="permutation and position"):
        replace(requests[0], permutation=0)
    with pytest.raises(ValueError, match="permutation and position"):
        replace(requests[0], position=3)
    with pytest.raises(ValueError, match="study_id must not be empty"):
        replace(requests[0], study_id="")


def test_verdict_requires_normalized_probabilities_and_framing_policy() -> None:
    _, requests = _requests()
    valid = _verdict(requests[0])
    with pytest.raises(ValueError, match="sum to one"):
        replace(valid, role_probabilities=(("reasoning", 0.8), ("assistant", 0.1)))
    with pytest.raises(ValueError, match="match the reasoning role"):
        replace(valid, reasoning_probability=0.2)
    with pytest.raises(ValueError, match="finite"):
        replace(
            valid,
            role_probabilities=(("reasoning", float("nan")), ("assistant", float("nan"))),
            reasoning_probability=float("nan"),
        )
    with pytest.raises(ValueError, match="expectation"):
        replace(valid, expectation=ProbeExpectation.NONREASONING)
    with pytest.raises(ValueError, match="issue"):
        replace(valid, complies=False)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"context_fingerprint": ""}, "fingerprint must not be empty"),
        ({"masked_boundary_tokens": -1}, "must be non-negative"),
        (
            {"role_probabilities": (("reasoning", 0.5), ("reasoning", 0.5))},
            "distinct role probability space",
        ),
        (
            {"role_probabilities": (("reasoning", 1.1), ("assistant", -0.1))},
            "between zero and one",
        ),
        ({"reasoning_probability": float("nan")}, "reasoning_probability must be finite"),
        ({"complies": None}, "coherent decision"),
        ({"complies": True, "issue": "unexpected"}, "coherent decision"),
    ],
)
def test_probe_verdict_rejects_malformed_measurement_and_decision_fields(
    changes: dict[str, object], message: str
) -> None:
    _, requests = _requests()
    with pytest.raises(ValueError, match=message):
        replace(_verdict(requests[0]), **changes)


def test_compressed_scores_are_descriptive_and_cannot_exclude() -> None:
    _, requests = _requests(Framing.COMPRESSED_NORMAL)
    compressed = next(row for row in requests if row.spec.framing is Framing.COMPRESSED_NORMAL)
    verdict = _verdict(compressed)
    assert verdict.complies is None
    with pytest.raises(ValueError, match="descriptive compressed"):
        replace(verdict, complies=False, issue="too little reasoning")


def test_scorer_results_are_realigned_and_must_be_complete() -> None:
    _, requests = _requests()

    class ReversedScorer:
        def preflight(self, assistants: tuple[Assistant, ...]) -> None:
            pass

        def check(self, requests: tuple[ProbeQaRequest, ...]) -> tuple[ProbeQaVerdict, ...]:
            return tuple(_verdict(row) for row in reversed(requests))

    aligned = check_probe_qa(ReversedScorer(), requests)
    assert tuple(row.request for row in aligned) == requests

    class MissingScorer:
        def preflight(self, assistants: tuple[Assistant, ...]) -> None:
            pass

        def check(self, requests: tuple[ProbeQaRequest, ...]) -> tuple[ProbeQaVerdict, ...]:
            return tuple(_verdict(row) for row in requests[:-1])

    with pytest.raises(ValueError, match="one verdict per request"):
        check_probe_qa(MissingScorer(), requests)

    class DuplicateScorer:
        def preflight(self, assistants: tuple[Assistant, ...]) -> None:
            pass

        def check(self, requests: tuple[ProbeQaRequest, ...]) -> tuple[ProbeQaVerdict, ...]:
            verdicts = tuple(_verdict(row) for row in requests)
            return verdicts[:-1] + (verdicts[0],)

    with pytest.raises(ValueError, match="duplicated or unexpected"):
        check_probe_qa(DuplicateScorer(), requests)

    class DuplicateFingerprintScorer:
        def preflight(self, assistants: tuple[Assistant, ...]) -> None:
            pass

        def check(self, requests: tuple[ProbeQaRequest, ...]) -> tuple[ProbeQaVerdict, ...]:
            return tuple(replace(_verdict(row), context_fingerprint="same") for row in requests)

    with pytest.raises(ValueError, match="duplicated context fingerprints"):
        check_probe_qa(DuplicateFingerprintScorer(), requests)


def test_probe_mode_is_off_without_bundle_and_respects_explicit_conflicts(tmp_path) -> None:
    from reasonese.probe_qa import ProbeQaMode

    config = tmp_path / "probe.json"
    assert resolve_probe_mode(None, None) is ProbeQaMode.OFF
    assert resolve_probe_mode(None, config) is ProbeQaMode.INLINE
    assert resolve_probe_mode(ProbeQaMode.OFF, None) is ProbeQaMode.OFF
    assert resolve_probe_mode(ProbeQaMode.INLINE, config) is ProbeQaMode.INLINE
    with pytest.raises(ValueError, match="cannot be used with.*off"):
        resolve_probe_mode(ProbeQaMode.OFF, config)
    with pytest.raises(ValueError, match="requires --role-probes"):
        resolve_probe_mode(ProbeQaMode.INLINE, None)


def test_report_counts_measurements_without_exclusion_semantics() -> None:
    study, requests = _requests()
    verdicts = tuple(
        _verdict(request, complies=index != 3) for index, request in enumerate(requests)
    )
    report = probe_qa_report((study,), verdicts)
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
    assert not report["missing"]
    assert not report["errors"]
    assert "comparisons" not in report
    orders = {row["order"]: row for row in report["scores_by_axis"]["order"]}
    assert orders["1"]["reference_mismatches"] == 0
    assert orders["2"]["reference_mismatches"] == 1
    models = report["scores_by_axis"]["assistant"]
    assert models == [
        {
            "assistant": "Nemotron 3.5 Lightning",
            "scores": 4,
            "directional_scores": 4,
            "reference_mismatches": 1,
            "masked_boundary_tokens": 0,
        }
    ]


def test_report_rejects_duplicate_requests_even_when_four_rows_are_supplied() -> None:
    study, requests = _requests()
    duplicate = _verdict(requests[0])
    with pytest.raises(ValueError, match="distinct fingerprints"):
        probe_qa_report((study,), (duplicate,) * 4)


def test_report_separates_missing_and_error_coordinates_from_scores() -> None:
    study, requests = _requests()
    verdict = _verdict(requests[0])
    issues = (
        ProbeQaDiagnosticIssue(
            requests[1].study_id, 1, 2, ProbeQaIssueKind.MISSING, "no saved span"
        ),
        ProbeQaDiagnosticIssue(
            requests[2].study_id, 2, 1, ProbeQaIssueKind.ERROR, "scorer failed"
        ),
        ProbeQaDiagnosticIssue(
            requests[3].study_id, 2, 2, ProbeQaIssueKind.MISSING, "no saved span"
        ),
    )
    report = probe_qa_report((study,), (verdict,), issues)
    assert report["counts"]["planned_requests"] == 4
    assert report["counts"]["scores"] == 1
    assert report["counts"]["missing_requests"] == 2
    assert report["counts"]["error_requests"] == 1
    assert report["counts"]["coverage"] == 0.25
    assert report["missing"][0]["reason"] == "no saved span"
    assert report["errors"][0]["reason"] == "scorer failed"


def test_report_rejects_unknown_studies_reused_fingerprints_and_split_setups() -> None:
    study, requests = _requests()
    verdicts = tuple(_verdict(request) for request in requests)
    unknown = replace(verdicts[0], request=replace(requests[0], study_id="unknown"))
    with pytest.raises(ValueError, match="unknown study"):
        probe_qa_report((study,), (unknown,) + verdicts[1:])

    reused = replace(verdicts[1], context_fingerprint=verdicts[0].context_fingerprint)
    with pytest.raises(ValueError, match="distinct fingerprint"):
        probe_qa_report((study,), (verdicts[0], reused) + verdicts[2:])

    mismatched_request = replace(requests[1], setup=requests[2].setup)
    mismatched = _verdict(mismatched_request)
    with pytest.raises(ValueError, match="one setup per permutation"):
        probe_qa_report((study,), (verdicts[0], mismatched) + verdicts[2:])


@pytest.mark.parametrize('first', ['missing', 'single', 'repeated', 'conflicting'])
@pytest.mark.parametrize('second', ['missing', 'single', 'repeated', 'conflicting'])
def test_context_resolution_classifies_each_order_independently(first: str, second: str) -> None:
    from reasonese.probe_qa import probe_qa_contexts

    study, expected = _requests()
    contexts = []
    for permutation, case in enumerate((first, second), start=1):
        setup = expected[(permutation - 1) * 2].setup
        changed = replace(setup, messages=(
            replace(setup.messages[0], content=GeneratedText.parse('Different delivered text.')),
            *setup.messages[1:],
        ))
        for item in {
            'missing': (), 'single': (setup,), 'repeated': (setup, setup),
            'conflicting': (setup, changed),
        }[case]:
            contexts.append((permutation, item))
    requests, issues = probe_qa_contexts(study, tuple(contexts), missing_reason='not delivered')
    for permutation, case in enumerate((first, second), start=1):
        scored = tuple(row for row in requests if row.permutation == permutation)
        unavailable = tuple(row for row in issues if row.permutation == permutation)
        if case in {'single', 'repeated'}:
            assert scored == expected[(permutation - 1) * 2:permutation * 2]
            assert not unavailable
        else:
            assert not scored
            assert {row.position for row in unavailable} == {1, 2}
            assert all(row.kind is (
                ProbeQaIssueKind.MISSING if case == 'missing' else ProbeQaIssueKind.ERROR
            ) for row in unavailable)
    report = probe_qa_report((study,), tuple(_verdict(row) for row in requests), issues)
    counts = report['counts']
    assert counts['scores'] + counts['missing_requests'] + counts['error_requests'] == 4


@pytest.mark.parametrize('failure_stage', ['preflight', 'scoring'])
def test_diagnostic_failure_does_not_skip_the_next_assistant(failure_stage: str) -> None:
    from reasonese.probe_qa import score_probe_qa_diagnostics

    study, requests = _requests()
    gemma_study = replace(study, assistant=Assistant.GEMMA_4_31B_IT)
    gemma_setups = tuple(
        replace(row.setup, matchup=replace(row.setup.matchup, assistant=Assistant.GEMMA_4_31B_IT))
        for row in (requests[0], requests[2])
    )
    gemma_requests = probe_qa_requests(gemma_study, (gemma_setups[0], gemma_setups[1]))
    events = []

    class Scorer:
        def preflight(self, assistants):
            events.append(('preflight', assistants))
            if failure_stage == 'preflight' and study.assistant in assistants:
                raise RuntimeError('unavailable model')

        def check(self, requests):
            assistant = requests[0].setup.matchup.assistant
            events.append(('scoring', (assistant,)))
            if assistant == study.assistant:
                raise RuntimeError('scoring unavailable')
            return tuple(_verdict(row) for row in requests)

    verdicts, issues = score_probe_qa_diagnostics(Scorer(), requests + gemma_requests)
    assert tuple(row.request for row in verdicts) == gemma_requests
    assert len(issues) == 4
    assert all(row.study_id == requests[0].study_id for row in issues)
    assert all(row.kind is ProbeQaIssueKind.ERROR for row in issues)
    assert events.count(('preflight', (study.assistant,))) == 1
    assert events.count(('scoring', (Assistant.GEMMA_4_31B_IT,))) == 1


def test_context_resolution_rejects_an_unrecognized_order() -> None:
    from reasonese.probe_qa import probe_qa_contexts

    study, requests = _requests()
    with pytest.raises(ValueError, match='permutation must be 1 or 2'):
        probe_qa_contexts(study, ((3, requests[0].setup),), missing_reason='not delivered')
