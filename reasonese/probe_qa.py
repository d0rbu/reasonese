"""Model-native activation-probe checks for materialized instructions."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from beartype import beartype

from reasonese.axes import Assistant, Framing
from reasonese.conversation import ConversationSetup
from reasonese.matchup import prompt_spec_to_dict
from reasonese.observations import cell_id
from reasonese.openrouter import JsonObject
from reasonese.planning import PromptSpec
from reasonese.study import Cell, Study, build_trials, study_fingerprint


class ProbeExpectation(StrEnum):
    """Whether a framing has an enforced latent-role direction."""

    REASONING = "reasoning"
    NONREASONING = "nonreasoning"
    DESCRIPTIVE = "descriptive"


@beartype
def probe_expectation(spec: PromptSpec) -> ProbeExpectation:
    """Map framing semantics to the preregistered probe policy."""
    if spec.framing in {Framing.REASONESE_NORMAL, Framing.REASONESE_PERSUASIVE}:
        return ProbeExpectation.REASONING
    if spec.framing in {Framing.COMPRESSED_NORMAL, Framing.COMPRESSED_PERSUASIVE}:
        return ProbeExpectation.DESCRIPTIVE
    if spec.framing in {
        Framing.NORMAL,
        Framing.CASUAL,
        Framing.PERSUASIVE,
        Framing.SUBAGENT,
    }:
        return ProbeExpectation.NONREASONING
    raise ValueError(f"unhandled framing for probe QA: {spec.framing}")


@beartype
@dataclass(frozen=True, slots=True)
class ProbeQaRequest:
    """One input span in one exact ordered, assistant-specific context."""

    study_id: str
    permutation: int
    position: int
    setup: ConversationSetup

    def __post_init__(self) -> None:
        if self.permutation not in {1, 2} or self.position not in {1, 2}:
            raise ValueError("probe request permutation and position must be 1 or 2")
        if self.study_id == "":
            raise ValueError("probe request study_id must not be empty")

    @property
    def spec(self) -> PromptSpec:
        return self.setup.matchup.inputs[self.position - 1]


@beartype
@dataclass(frozen=True, slots=True)
class ProbeQaVerdict:
    """Measured role probabilities and an optional enforced decision."""

    request: ProbeQaRequest
    context_fingerprint: str
    role_probabilities: tuple[tuple[str, float], ...]
    reasoning_probability: float
    expectation: ProbeExpectation
    complies: bool | None
    issue: str | None

    def __post_init__(self) -> None:
        if not self.context_fingerprint:
            raise ValueError("probe context fingerprint must not be empty")
        roles = tuple(role for role, _ in self.role_probabilities)
        probabilities = tuple(value for _, value in self.role_probabilities)
        if len(roles) < 2 or len(set(roles)) != len(roles) or "reasoning" not in roles:
            raise ValueError("probe verdict must contain a distinct role probability space")
        if any(not math.isfinite(value) or value < 0 or value > 1 for value in probabilities):
            raise ValueError("probe role probabilities must be finite and lie between zero and one")
        if not math.isfinite(self.reasoning_probability):
            raise ValueError("reasoning_probability must be finite")
        if abs(sum(probabilities) - 1) > 1e-6:
            raise ValueError("probe role probabilities must sum to one")
        measured = dict(self.role_probabilities)["reasoning"]
        if abs(measured - self.reasoning_probability) > 1e-12:
            raise ValueError("reasoning_probability must match the reasoning role")
        expected = probe_expectation(self.request.spec)
        if self.expectation is not expected:
            raise ValueError("probe expectation does not match the input framing")
        if expected is ProbeExpectation.DESCRIPTIVE:
            if self.complies is not None or self.issue is not None:
                raise ValueError("descriptive compressed scores cannot pass or fail QA")
        elif self.complies is None or (self.complies and self.issue is not None):
            raise ValueError("enforced probe verdict must contain a coherent decision")
        elif not self.complies and not self.issue:
            raise ValueError("failed probe verdict must contain an issue")


@runtime_checkable
class ProbeQaScorer(Protocol):
    """Assistant-specific scorer that owns local extraction and its cache."""

    def preflight(self, assistants: tuple[Assistant, ...]) -> None: ...

    def check(self, requests: tuple[ProbeQaRequest, ...]) -> tuple[ProbeQaVerdict, ...]: ...


@beartype
def probe_qa_requests(
    study: Study,
    setups_by_permutation: tuple[ConversationSetup, ConversationSetup],
) -> tuple[ProbeQaRequest, ...]:
    """Build both positions in both orderings exactly once per study."""
    expected = tuple(trial.matchup for trial in build_trials(study) if int(trial.rollout) == 1)
    if tuple(setup.matchup for setup in setups_by_permutation) != expected:
        raise ValueError("probe setups do not match the study's ordered permutations")
    study_id = study_fingerprint(study)
    return tuple(
        ProbeQaRequest(study_id, permutation, position, setup)
        for permutation, setup in enumerate(setups_by_permutation, start=1)
        for position in (1, 2)
    )


@beartype
def check_probe_qa(
    scorer: ProbeQaScorer,
    requests: tuple[ProbeQaRequest, ...],
) -> tuple[ProbeQaVerdict, ...]:
    """Run the scorer and reject missing, duplicated, or misaligned results."""
    verdicts = scorer.check(requests)
    if len(verdicts) != len(requests):
        raise ValueError("probe scorer must return one verdict per request")
    by_request = {verdict.request: verdict for verdict in verdicts}
    if len(by_request) != len(verdicts) or set(by_request) != set(requests):
        raise ValueError("probe scorer returned duplicated or unexpected requests")
    return tuple(by_request[request] for request in requests)


def _axis_rows(
    verdicts: tuple[ProbeQaVerdict, ...],
    axis: str,
) -> list[JsonObject]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for verdict in verdicts:
        if axis == "assistant":
            value = str(verdict.request.setup.matchup.assistant)
        elif axis == "order":
            value = str(verdict.request.permutation)
        else:
            value = str(getattr(verdict.request.spec, axis))
        row = counts[value]
        row[0] += 1
        row[1] += int(verdict.complies is not None)
        row[2] += int(verdict.complies is False)
    return [
        {
            axis: value,
            "scores": row[0],
            "enforced_scores": row[1],
            "failed_scores": row[2],
        }
        for value, row in sorted(counts.items())
    ]


@beartype
def probe_qa_report(
    studies: tuple[Study, ...],
    verdicts: tuple[ProbeQaVerdict, ...],
) -> JsonObject:
    """Report scores and whole-comparison exclusions across requested axes."""
    by_study: dict[str, list[ProbeQaVerdict]] = defaultdict(list)
    known = {study_fingerprint(study): study for study in studies}
    for verdict in verdicts:
        if verdict.request.study_id not in known:
            raise ValueError("probe verdict references an unknown study")
        by_study[verdict.request.study_id].append(verdict)
    for study_id, study in known.items():
        rows = by_study[study_id]
        if len(rows) != 4 or len({row.request for row in rows}) != 4:
            raise ValueError("probe report requires four distinct requests for every study")
        if len({row.context_fingerprint for row in rows}) != 4:
            raise ValueError("probe report requires a distinct fingerprint for every target span")
        setups: list[ConversationSetup] = []
        for permutation in (1, 2):
            matching = [row.request.setup for row in rows if row.request.permutation == permutation]
            if len(matching) != 2 or matching[0] != matching[1]:
                raise ValueError("probe report requests do not share one setup per permutation")
            setups.append(matching[0])
        expected = probe_qa_requests(study, (setups[0], setups[1]))
        if {row.request for row in rows} != set(expected):
            raise ValueError("probe report requests do not match the study positions and orders")

    comparisons: list[JsonObject] = []
    for study_id, study in known.items():
        rows = by_study[study_id]
        failures = [row for row in rows if row.complies is False]
        trials = build_trials(study)
        comparisons.append(
            {
                "study_id": study_id,
                "assistant": str(study.assistant),
                "inputs": [prompt_spec_to_dict(spec) for spec in study.inputs],
                "cell_ids": [str(cell_id(Cell(spec, study.assistant))) for spec in study.inputs],
                "trial_ids": [str(trial.trial_id) for trial in trials],
                "excluded": bool(failures),
                "failed_requests": [
                    {
                        "permutation": row.request.permutation,
                        "position": row.request.position,
                        "framing": str(row.request.spec.framing),
                        "channel": str(row.request.spec.channel),
                        "author": str(row.request.spec.author),
                        "reasoning_probability": row.reasoning_probability,
                        "issue": row.issue,
                    }
                    for row in failures
                ],
            }
        )
    return {
        "counts": {
            "scores": len(verdicts),
            "enforced_scores": sum(row.complies is not None for row in verdicts),
            "descriptive_scores": sum(row.complies is None for row in verdicts),
            "failed_scores": sum(row.complies is False for row in verdicts),
            "planned_comparisons": len(studies),
            "excluded_comparisons": sum(row["excluded"] for row in comparisons),
            "planned_trials": sum(len(row["trial_ids"]) for row in comparisons),
            "excluded_trials": sum(len(row["trial_ids"]) for row in comparisons if row["excluded"]),
        },
        "scores_by_axis": {
            axis: _axis_rows(verdicts, axis)
            for axis in ("assistant", "framing", "channel", "author", "order")
        },
        "scores": [
            {
                "study_id": row.request.study_id,
                "assistant": str(row.request.setup.matchup.assistant),
                "permutation": row.request.permutation,
                "position": row.request.position,
                **prompt_spec_to_dict(row.request.spec),
                "context_fingerprint": row.context_fingerprint,
                "role_probabilities": dict(row.role_probabilities),
                "reasoning_probability": row.reasoning_probability,
                "expectation": str(row.expectation),
                "complies": row.complies,
                "issue": row.issue,
            }
            for row in verdicts
        ],
        "comparisons": comparisons,
    }
