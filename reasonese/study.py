"""Permutation-balanced study definitions and trial planning."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, cast

from beartype import beartype
from phantom import Phantom

from reasonese.axes import Assistant, is_non_empty_trimmed
from reasonese.matchup import (
    Matchup,
    is_matchup_inputs,
    make_matchup,
    prompt_spec_from_dict,
    prompt_spec_to_dict,
)
from reasonese.planning import PromptSpec


def _is_positive_integer(value: int) -> bool:
    return value > 0


class PositiveInteger(
    int,
    Phantom[int],
    predicate=_is_positive_integer,
    bound=int,
):
    """An integer greater than zero."""


def _is_study_inputs(value: tuple[PromptSpec, ...]) -> bool:
    return is_matchup_inputs(value) and len(set(value)) == 2


class StudyInputs(
    tuple[PromptSpec, ...],
    Phantom,
    predicate=_is_study_inputs,
):
    """Exactly two distinct inputs, including an explicit user message."""


class TrialId(str, Phantom[str], predicate=is_non_empty_trimmed, bound=str):
    """A stable non-empty trial identifier."""


@beartype
@dataclass(frozen=True, slots=True)
class Cell:
    """One four-axis datapoint evaluated by one assistant."""

    spec: PromptSpec
    assistant: Assistant


@beartype
@dataclass(frozen=True, slots=True)
class Study:
    """Distinct cells evaluated over every input ordering."""

    inputs: StudyInputs
    assistant: Assistant
    rollouts_per_permutation: PositiveInteger


@beartype
@dataclass(frozen=True, slots=True)
class Trial:
    """One ordered matchup and rollout within a study."""

    trial_id: TrialId
    matchup: Matchup
    permutation: PositiveInteger
    rollout: PositiveInteger


@beartype
def make_study(
    inputs: tuple[PromptSpec, ...],
    assistant: Assistant,
    rollouts_per_permutation: int,
) -> Study:
    """Validate study-wide balance invariants and construct a study."""
    if len(inputs) != 2:
        raise ValueError("a study requires exactly two inputs")
    if len(inputs) != len(set(inputs)):
        raise ValueError("study inputs must be distinct to produce unique permutations")
    if not is_matchup_inputs(inputs):
        raise ValueError("a study requires at least one user message input")
    if rollouts_per_permutation < 1:
        raise ValueError("rollouts per permutation must be at least one")
    return Study(
        StudyInputs.parse(inputs),
        assistant,
        PositiveInteger.parse(rollouts_per_permutation),
    )


@beartype
def study_to_dict(study: Study) -> dict[str, object]:
    """Serialize a study to YAML-compatible data."""
    return {
        "assistant": str(study.assistant),
        "rollouts_per_permutation": int(study.rollouts_per_permutation),
        "inputs": [prompt_spec_to_dict(spec) for spec in study.inputs],
    }


def study_from_dict(raw: object) -> Study:
    """Parse a study from YAML-compatible data."""
    if not isinstance(raw, dict):
        raise ValueError("study must be a mapping")
    data = cast(dict[str, Any], raw)
    expected = {"assistant", "rollouts_per_permutation", "inputs"}
    if set(data) != expected:
        raise ValueError(f"study fields must be {sorted(expected)}")
    raw_inputs = data["inputs"]
    if not isinstance(raw_inputs, list):
        raise ValueError("study inputs must be a list")
    rollouts = data["rollouts_per_permutation"]
    if not isinstance(rollouts, int) or isinstance(rollouts, bool):
        raise ValueError("rollouts per permutation must be an integer")
    return make_study(
        tuple(prompt_spec_from_dict(item) for item in raw_inputs),
        Assistant(data["assistant"]),
        rollouts,
    )


@beartype
def study_fingerprint(study: Study) -> str:
    """Return a short stable identifier for one exact study definition."""
    canonical = json.dumps(study_to_dict(study), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


@beartype
def build_trials(study: Study) -> tuple[Trial, ...]:
    """Enumerate both pair orderings and requested rollouts in stable order."""
    prefix = study_fingerprint(study)
    trials: list[Trial] = []
    first, second = study.inputs
    matchups = (
        make_matchup((first, second), study.assistant),
        make_matchup((second, first), study.assistant),
    )
    for permutation_index, matchup in enumerate(
        matchups, start=1
    ):
        for rollout_index in range(1, int(study.rollouts_per_permutation) + 1):
            permutation = PositiveInteger.parse(permutation_index)
            rollout = PositiveInteger.parse(rollout_index)
            trials.append(
                Trial(
                    TrialId.parse(
                        f"{prefix}-permutation-{permutation_index:06d}-rollout-{rollout_index:04d}"
                    ),
                    matchup,
                    permutation,
                    rollout,
                )
            )
    return tuple(trials)


@beartype
def study_cells(study: Study) -> tuple[Cell, ...]:
    """Return the cells represented in a study."""
    return tuple(Cell(spec, study.assistant) for spec in study.inputs)


@beartype
def trial_count(study: Study) -> PositiveInteger:
    """Return two orderings times rollouts per ordering."""
    return PositiveInteger.parse(2 * int(study.rollouts_per_permutation))


@beartype
def observations_per_cell(study: Study) -> PositiveInteger:
    """Return the balanced number of verdicts collected for every cell."""
    return trial_count(study)


@beartype
def observations_per_cell_position(study: Study) -> PositiveInteger:
    """Return one verdict per rollout for a cell at each of the two positions."""
    return study.rollouts_per_permutation
