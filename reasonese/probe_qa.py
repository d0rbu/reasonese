"""Model-native activation-probe checks for materialized instructions."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable

from beartype import beartype

from reasonese.axes import Assistant, Framing
from reasonese.conversation import ConversationSetup
from reasonese.matchup import prompt_spec_to_dict
from reasonese.openrouter import JsonObject
from reasonese.planning import PromptSpec
from reasonese.study import Study, build_trials, study_fingerprint


class ProbeExpectation(StrEnum):
    """Whether a framing has a directional reference for descriptive scoring."""

    REASONING = "reasoning"
    NONREASONING = "nonreasoning"
    DESCRIPTIVE = "descriptive"


class ProbeQaMode(StrEnum):
    """Collection-time probe behavior; probe results are always diagnostic."""

    OFF = "off"
    INLINE = "inline"


@beartype
def resolve_probe_mode(
    requested: ProbeQaMode | None,
    role_probes: Path | None,
) -> ProbeQaMode:
    """Resolve the CLI mode while preserving an explicitly supplied bundle's intent."""
    if requested is None:
        return ProbeQaMode.INLINE if role_probes is not None else ProbeQaMode.OFF
    if requested is ProbeQaMode.OFF and role_probes is not None:
        raise ValueError("--role-probes cannot be used with --probe-mode off")
    if requested is ProbeQaMode.INLINE and role_probes is None:
        raise ValueError("--probe-mode inline requires --role-probes")
    return requested


class ProbeQaIssueKind(StrEnum):
    """Why one planned probe coordinate has no score."""

    MISSING = "missing"
    ERROR = "error"


@beartype
def probe_expectation(spec: PromptSpec) -> ProbeExpectation:
    """Map framing semantics to the frozen descriptive reference direction."""
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
    """Measured role probabilities and an optional directional reference comparison."""

    request: ProbeQaRequest
    context_fingerprint: str
    role_probabilities: tuple[tuple[str, float], ...]
    reasoning_probability: float
    expectation: ProbeExpectation
    complies: bool | None
    issue: str | None
    masked_boundary_tokens: int = 0

    def __post_init__(self) -> None:
        if not self.context_fingerprint:
            raise ValueError("probe context fingerprint must not be empty")
        if self.masked_boundary_tokens < 0:
            raise ValueError("masked_boundary_tokens must be non-negative")
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


@beartype
@dataclass(frozen=True, slots=True)
class ProbeQaDiagnosticIssue:
    """One missing or failed diagnostic score, without an eligibility decision."""

    study_id: str
    permutation: int
    position: int
    kind: ProbeQaIssueKind
    reason: str

    def __post_init__(self) -> None:
        if not self.study_id:
            raise ValueError("probe diagnostic issue study_id must not be empty")
        if self.permutation not in {1, 2} or self.position not in {1, 2}:
            raise ValueError("probe diagnostic issue permutation and position must be 1 or 2")
        if not self.reason:
            raise ValueError("probe diagnostic issue reason must not be empty")


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
    return probe_qa_requests_for_setups(study, tuple(enumerate(setups_by_permutation, start=1)))


@beartype
def probe_qa_requests_for_setups(
    study: Study,
    setups_by_permutation: tuple[tuple[int, ConversationSetup], ...],
) -> tuple[ProbeQaRequest, ...]:
    """Build target requests for only the exact ordered setups that are available."""
    expected = tuple(trial.matchup for trial in build_trials(study) if int(trial.rollout) == 1)
    if len(expected) != 2:
        raise ValueError("probe scoring requires exactly two ordered study permutations")
    if len({permutation for permutation, _ in setups_by_permutation}) != len(setups_by_permutation):
        raise ValueError("probe setup permutations must be unique")
    if any(permutation not in {1, 2} for permutation, _ in setups_by_permutation):
        raise ValueError("probe setup permutation must be 1 or 2")
    for permutation, setup in setups_by_permutation:
        if setup.matchup != expected[permutation - 1]:
            raise ValueError("probe setup does not match its study permutation")
    study_id = study_fingerprint(study)
    return tuple(
        ProbeQaRequest(study_id, permutation, position, setup)
        for permutation, setup in setups_by_permutation
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
    if len({verdict.context_fingerprint for verdict in verdicts}) != len(verdicts):
        raise ValueError("probe scorer returned duplicated context fingerprints")
    return tuple(by_request[request] for request in requests)


@beartype
def score_probe_qa_diagnostics(
    scorer: ProbeQaScorer,
    requests: tuple[ProbeQaRequest, ...],
) -> tuple[tuple[ProbeQaVerdict, ...], tuple[ProbeQaDiagnosticIssue, ...]]:
    """Run one batch per assistant and turn scorer failures into diagnostic rows."""
    by_assistant: dict[Assistant, list[ProbeQaRequest]] = defaultdict(list)
    for request in requests:
        by_assistant[request.setup.matchup.assistant].append(request)

    prepared: set[Assistant] = set()
    preflight_errors: dict[Assistant, str] = {}
    logger = logging.getLogger(__name__)
    for assistant in by_assistant:
        try:
            scorer.preflight((assistant,))
        except Exception as error:
            logger.exception("Activation probe diagnostic preflight failed for %s", assistant)
            preflight_errors[assistant] = f"{type(error).__name__}: {error}"
        else:
            prepared.add(assistant)

    verdicts: list[ProbeQaVerdict] = []
    issues: list[ProbeQaDiagnosticIssue] = []
    for assistant, assistant_requests in by_assistant.items():
        if assistant in preflight_errors:
            error_text = f"probe preflight failed: {preflight_errors[assistant]}"
        elif assistant in prepared:
            try:
                verdicts.extend(check_probe_qa(scorer, tuple(assistant_requests)))
                continue
            except Exception as error:
                logger.exception("Activation probe diagnostic scoring failed for %s", assistant)
                error_text = (
                    "probe scoring failed before a complete batch returned: "
                    f"{type(error).__name__}: {error}"
                )
        else:
            continue
        issues.extend(
            ProbeQaDiagnosticIssue(
                request.study_id,
                request.permutation,
                request.position,
                ProbeQaIssueKind.ERROR,
                error_text,
            )
            for request in assistant_requests
        )
    return tuple(verdicts), tuple(issues)


def _axis_rows(
    verdicts: tuple[ProbeQaVerdict, ...],
    axis: str,
) -> list[JsonObject]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
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
        row[3] += verdict.masked_boundary_tokens
    return [
        {
            axis: value,
            "scores": row[0],
            "directional_scores": row[1],
            "reference_mismatches": row[2],
            "masked_boundary_tokens": row[3],
        }
        for value, row in sorted(counts.items())
    ]


@beartype
def probe_qa_report(
    studies: tuple[Study, ...],
    verdicts: tuple[ProbeQaVerdict, ...],
    issues: tuple[ProbeQaDiagnosticIssue, ...] = (),
) -> JsonObject:
    """Report probe measurements, missing coverage, and scoring errors separately."""
    by_study: dict[str, list[ProbeQaVerdict]] = defaultdict(list)
    known = {study_fingerprint(study): study for study in studies}
    for verdict in verdicts:
        if verdict.request.study_id not in known:
            raise ValueError("probe verdict references an unknown study")
        by_study[verdict.request.study_id].append(verdict)
    issue_by_study: dict[str, list[ProbeQaDiagnosticIssue]] = defaultdict(list)
    for issue in issues:
        if issue.study_id not in known:
            raise ValueError("probe diagnostic issue references an unknown study")
        issue_by_study[issue.study_id].append(issue)

    all_coordinates: set[tuple[str, int, int]] = set()
    for study_id, study in known.items():
        study_verdicts = by_study[study_id]
        study_issues = issue_by_study[study_id]
        if len({row.context_fingerprint for row in study_verdicts}) != len(study_verdicts):
            raise ValueError("probe report requires distinct fingerprints for scored target spans")
        coordinates = [
            (row.request.study_id, row.request.permutation, row.request.position)
            for row in study_verdicts
        ] + [
            (row.study_id, row.permutation, row.position)
            for row in study_issues
        ]
        if len(set(coordinates)) != len(coordinates):
            raise ValueError("probe report contains duplicate scored, missing, or error coordinates")
        expected_coordinates = {
            (study_id, permutation, position)
            for permutation in (1, 2)
            for position in (1, 2)
        }
        if set(coordinates) != expected_coordinates:
            raise ValueError("probe report must classify all four coordinates for every study")
        for permutation in (1, 2):
            matching = [
                row.request.setup
                for row in study_verdicts
                if row.request.permutation == permutation
            ]
            if matching and any(setup != matching[0] for setup in matching[1:]):
                raise ValueError("probe report requests do not share one setup per permutation")
            if any(
                row.request.setup.matchup
                != tuple(trial.matchup for trial in build_trials(study) if int(trial.rollout) == 1)[
                    permutation - 1
                ]
                for row in study_verdicts
                if row.request.permutation == permutation
            ):
                raise ValueError("probe report request does not match its study permutation")
        all_coordinates.update(coordinates)
    if len(all_coordinates) != 4 * len(studies):
        raise ValueError("probe report study coordinates are not unique")

    score_count = len(verdicts)
    missing_issues = [issue for issue in issues if issue.kind is ProbeQaIssueKind.MISSING]
    error_issues = [issue for issue in issues if issue.kind is ProbeQaIssueKind.ERROR]
    expected_count = 4 * len(studies)
    return {
        "counts": {
            "planned_requests": expected_count,
            "scores": score_count,
            "directional_scores": sum(row.complies is not None for row in verdicts),
            "descriptive_scores": sum(row.complies is None for row in verdicts),
            "reference_mismatches": sum(row.complies is False for row in verdicts),
            "missing_requests": len(missing_issues),
            "error_requests": len(error_issues),
            "coverage": score_count / expected_count if expected_count else 1.0,
            "masked_boundary_tokens": sum(row.masked_boundary_tokens for row in verdicts),
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
                "masked_boundary_tokens": row.masked_boundary_tokens,
            }
            for row in verdicts
        ],
        "missing": [
            _issue_row(known[issue.study_id], issue)
            for issue in missing_issues
        ],
        "errors": [
            _issue_row(known[issue.study_id], issue)
            for issue in error_issues
        ],
    }


def _issue_row(study: Study, issue: ProbeQaDiagnosticIssue) -> JsonObject:
    input_index = issue.position - 1 if issue.permutation == 1 else 2 - issue.position
    spec = study.inputs[input_index]
    return {
        "study_id": issue.study_id,
        "assistant": str(study.assistant),
        "permutation": issue.permutation,
        "position": issue.position,
        **prompt_spec_to_dict(spec),
        "reason": issue.reason,
    }
