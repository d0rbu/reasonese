"""End-to-end checks across the seam this change touches.

Collection itself is unchanged and covered in `test_study_orchestration.py`, so
these tests wire the real bank through planning, sampling, trial construction,
and analysis at real scale, with verdicts synthesized instead of collected. No
provider is contacted.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from functools import cache
from pathlib import Path

import pytest
from phantom.interval import Natural

from reasonese.analysis import analyze_observations
from reasonese.axes import Assistant, Channel, Framing
from reasonese.instructions import InstructionPair, load_instruction_pairs
from reasonese.judging import TraceFingerprint
from reasonese.observations import Observation, cell_id
from reasonese.planning import PairSpecs, PromptSpec, build_pair_specs
from reasonese.sampling import build_sampled_studies, sample_pair_inputs
from reasonese.study import Cell, PositiveInteger, Study, build_trials

BANK = Path("configs/instruction_pairs.yaml")
PAIRINGS = 200


def _count(row: dict[str, object], key: str) -> int:
    """Read an integer table or diagnostic entry, asserting it really is one."""
    value = row[key]
    assert isinstance(value, int), f"{key} should be an integer, got {type(value).__name__}"
    return value


@cache
def _pairs() -> tuple[InstructionPair, ...]:
    return load_instruction_pairs(BANK)


@cache
def _pair_specs() -> tuple[PairSpecs, ...]:
    return build_pair_specs(_pairs())


def _completed(spec: PromptSpec, position: int, first_instructions: frozenset[str]) -> bool:
    """A deterministic verdict rule that yields all three trial outcomes."""
    if str(spec.instruction) in first_instructions:
        return spec.framing in (Framing.NORMAL, Framing.PERSUASIVE, Framing.SUBAGENT)
    return spec.channel is Channel.USER and position == 2


def _observations(
    studies: tuple[Study, ...], first_instructions: frozenset[str]
) -> tuple[Observation, ...]:
    rows: list[Observation] = []
    for study in studies:
        for trial in build_trials(study):
            fingerprint = TraceFingerprint.parse(
                hashlib.sha256(str(trial.trial_id).encode()).hexdigest()
            )
            for position, spec in enumerate(trial.matchup.inputs, start=1):
                rows.append(
                    Observation(
                        trial.trial_id,
                        cell_id(Cell(spec, study.assistant)),
                        spec,
                        study.assistant,
                        trial.permutation,
                        trial.rollout,
                        PositiveInteger.parse(position),
                        _completed(spec, position, first_instructions),
                        fingerprint,
                        f"assistant-{trial.trial_id}",
                        f"judge-{trial.trial_id}-{position}",
                    )
                )
    return tuple(rows)


def test_every_pair_in_the_bank_samples_to_a_connected_covering_design() -> None:
    """The whole bank, not just one pair, satisfies the sampling invariants."""
    for pair_specs in _pair_specs():
        sampled = sample_pair_inputs(
            pair_specs, PositiveInteger.parse(PAIRINGS), Natural.parse(0)
        )
        assert len(sampled) == PAIRINGS
        assert len({frozenset(inputs) for inputs in sampled}) == PAIRINGS

        first_side = set(pair_specs.first)
        second_side = set(pair_specs.second)
        degrees: Counter[PromptSpec] = Counter()
        for inputs in sampled:
            assert inputs[0] in first_side
            assert inputs[1] in second_side
            assert any(spec.channel is Channel.USER for spec in inputs)
            degrees.update(inputs)

        # 179 edges would be the bare spanning minimum, so 200 must cover
        # every one of the 180 cells with room to spare.
        assert len(degrees) == 180


def test_sampled_design_analyses_into_one_component_per_pair_and_assistant() -> None:
    pair_specs = _pair_specs()[:2]
    assistants = (Assistant.INKLING, Assistant.QWEN3_8_FLASH)
    studies = build_sampled_studies(
        pair_specs,
        assistants,
        PositiveInteger.parse(PAIRINGS),
        PositiveInteger.parse(1),
        Natural.parse(0),
    )
    assert len(studies) == len(pair_specs) * len(assistants) * PAIRINGS

    first_instructions = frozenset(str(item.pair.first) for item in pair_specs)
    observations = _observations(studies, first_instructions)
    assert len(observations) == 2 * 2 * len(studies)

    bundle = analyze_observations(
        observations,
        _pairs(),
        1.0,
        bootstrap_samples=0,
        seed=0,
    )

    expected_components = len(pair_specs) * len(assistants)
    assert len(bundle.fit.connected_components) == expected_components
    assert bundle.diagnostics["components_match_pair_assistant"] is True
    assert bundle.diagnostics["comparison_graph_connected"] is False
    assert all(len(component) == 180 for component in bundle.fit.connected_components)
    assert len(bundle.fit.ranking) == expected_components * 180

    # Ranks restart per component and every component self-centres, which is
    # what makes the pooled axis margins comparable.
    by_component: dict[int, list[float]] = {}
    for item in bundle.fit.ranking:
        by_component.setdefault(item.component_index, []).append(item.score)
    assert len(by_component) == expected_components
    for scores in by_component.values():
        assert len(scores) == 180
        assert sum(scores) == pytest.approx(0.0, abs=1e-9)

    # Instruction and assistant no longer appear as Bradley-Terry axes.
    assert {row["axis"] for row in bundle.axis_summary} == {"framing", "channel", "author"}
    assert {row["stratum"] for row in bundle.stratum_summary} == {
        "assistant",
        "skill",
        "conflict",
        "pair",
    }

    # Exclusivity accounts for every trial exactly once, per pair.
    assert len(bundle.pair_exclusivity) == len(pair_specs)
    total_trials = 0
    for row in bundle.pair_exclusivity:
        counted = (
            _count(row, "exactly_one")
            + _count(row, "both_completed")
            + _count(row, "neither_completed")
        )
        assert counted == _count(row, "trials")
        total_trials += counted
    both = _count(bundle.diagnostics, "both_completed_trials")
    neither = _count(bundle.diagnostics, "neither_completed_trials")
    assert total_trials == _count(bundle.diagnostics, "trials") == 2 * len(studies)
    assert bundle.fit.tie_count == both + neither

    # The synthetic verdict rule is meant to exercise all three outcomes.
    assert both > 0
    assert neither > 0


def test_a_pair_design_covers_every_condition_on_both_sides() -> None:
    pair_specs = _pair_specs()[0]
    sampled = sample_pair_inputs(
        pair_specs, PositiveInteger.parse(PAIRINGS), Natural.parse(4)
    )
    seen = {spec for inputs in sampled for spec in inputs}

    assert seen == set(pair_specs.first) | set(pair_specs.second)
    assert {spec.framing for spec in seen} == set(Framing)
    assert {spec.channel for spec in seen} == set(Channel)
