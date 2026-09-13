"""Probe-QA request, verdict, and attrition-report contracts."""

from __future__ import annotations

from dataclasses import replace

import pytest

from reasonese.axes import Assistant, Author, Channel, Framing, Instruction
from reasonese.conversation import GeneratedMessage, GeneratedText, construct_conversation
from reasonese.planning import PromptSpec
from reasonese.probe_qa import (
    ProbeExpectation,
    ProbeQaRequest,
    ProbeQaVerdict,
    check_probe_qa,
    probe_expectation,
    probe_qa_report,
    probe_qa_requests,
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


def test_report_counts_order_and_axis_failures_and_excludes_whole_comparison() -> None:
    study, requests = _requests()
    verdicts = tuple(
        _verdict(request, complies=index != 3) for index, request in enumerate(requests)
    )
    report = probe_qa_report((study,), verdicts)
    assert report["counts"] == {
        "scores": 4,
        "enforced_scores": 4,
        "descriptive_scores": 0,
        "failed_scores": 1,
        "masked_boundary_tokens": 0,
        "planned_comparisons": 1,
        "excluded_comparisons": 1,
        "planned_trials": 6,
        "excluded_trials": 6,
    }
    assert report["comparisons"][0]["excluded"] is True
    assert report["comparisons"][0]["failed_requests"] == [
        {
            "permutation": 2,
            "position": 2,
            "framing": "reasonese-normal",
            "channel": "user message",
            "author": "Nemotron 3.5 Lightning",
            "reasoning_probability": 0.9,
            "issue": "reasoning probability missed the preregistered gate",
        }
    ]
    orders = {row["order"]: row for row in report["scores_by_axis"]["order"]}
    assert orders["1"]["failed_scores"] == 0
    assert orders["2"]["failed_scores"] == 1
    models = report["scores_by_axis"]["assistant"]
    assert models == [
        {
            "assistant": "Nemotron 3.5 Lightning",
            "scores": 4,
            "enforced_scores": 4,
            "failed_scores": 1,
            "masked_boundary_tokens": 0,
        }
    ]


def test_report_rejects_duplicate_requests_even_when_four_rows_are_supplied() -> None:
    study, requests = _requests()
    duplicate = _verdict(requests[0])
    with pytest.raises(ValueError, match="four distinct requests"):
        probe_qa_report((study,), (duplicate,) * 4)


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
