"""Tests for within-pair study sampling.

These run against the real 24-pair bank at its real size: 90 conditions per
instruction side, a 4,500-edge population per pair, and the 720-edge pilot
default. Synthetic pairs are used only where a degenerate channel mix is needed
that the real bank cannot express.
"""

from __future__ import annotations

import itertools
import json
from collections import Counter
from functools import cache
from pathlib import Path
from random import Random

import pytest
import yaml
from phantom.interval import Natural

import reasonese.sample_studies as sample_studies_module
import reasonese.sampling as sampling_module
from reasonese.axes import Assistant, Author, Channel, Framing, Instruction
from reasonese.config import load_study_suite
from reasonese.instructions import (
    ConflictType,
    InstructionPair,
    PairId,
    Rationale,
    Skill,
    load_instruction_pairs,
)
from reasonese.io import write_study_suite
from reasonese.planning import PairSpecs, PromptSpec, build_pair_specs
from reasonese.sample_studies import main as sample_studies
from reasonese.sampling import (
    DEFAULT_PAIRINGS_PER_PAIR,
    _edge_from_rank,
    _proportional_quotas,
    _rank_from_edge,
    _ranks_by_stratum,
    _repair_connectivity,
    _sides,
    _stratum_for_rank,
    build_sampled_studies,
    default_pairing_count,
    minimum_connected_pairings,
    pairing_population_size,
    sample_pair_inputs,
)
from reasonese.study import PositiveInteger, StudyInputs, study_to_dict

BANK = Path("configs/instruction_pairs.yaml")
SIDE_SIZE = 90
POPULATION = 4500
NODES = 2 * SIDE_SIZE


@cache
def _bank() -> tuple[InstructionPair, ...]:
    return load_instruction_pairs(BANK)


@cache
def _bank_specs() -> tuple[PairSpecs, ...]:
    return build_pair_specs(_bank())


def _first_pair() -> PairSpecs:
    return _bank_specs()[0]


def _synthetic_side(
    instruction: Instruction, channels: tuple[Channel, ...]
) -> tuple[PromptSpec, ...]:
    """Build a side whose channels are given, keeping every spec distinct."""
    combinations = tuple(itertools.product(Framing, Author))
    return tuple(
        PromptSpec(instruction, combinations[index][0], channel, combinations[index][1])
        for index, channel in enumerate(channels)
    )


def _synthetic_pair_specs(
    first_channels: tuple[Channel, ...],
    second_channels: tuple[Channel, ...],
    pair_id: str = "probe-pair",
) -> PairSpecs:
    first = Instruction.parse("Do the first thing.")
    second = Instruction.parse("Do the second thing.")
    pair = InstructionPair(
        PairId.parse(pair_id),
        Skill.PYTHON,
        ConflictType.OUTPUT_FORMAT,
        first,
        second,
        Rationale.parse("The two requests cannot both be satisfied."),
    )
    return PairSpecs(
        pair,
        _synthetic_side(first, first_channels),
        _synthetic_side(second, second_channels),
    )


def _brute_force_edges(pair_specs: PairSpecs) -> set[tuple[int, int]]:
    """Enumerate valid edges directly from the definition, ignoring rank maths."""
    return {
        (index, other)
        for index, first in enumerate(pair_specs.first)
        for other, second in enumerate(pair_specs.second)
        if Channel.USER in (first.channel, second.channel)
    }


def _components(pair_specs: PairSpecs, pairings: tuple[StudyInputs, ...]) -> list[set[PromptSpec]]:
    reached: list[set[PromptSpec]] = []
    for inputs in pairings:
        edge = set(inputs)
        touching = [component for component in reached if component & edge]
        merged = set(edge)
        for component in touching:
            merged |= component
            reached.remove(component)
        reached.append(merged)
    covered = {spec for component in reached for spec in component}
    for spec in pair_specs.first + pair_specs.second:
        if spec not in covered:
            reached.append({spec})
    return reached


# --------------------------------------------------------------------------
# Population, bounds, and the rank encoding
# --------------------------------------------------------------------------


def test_every_real_pair_has_the_expected_population_and_bounds() -> None:
    specs = _bank_specs()
    assert len(specs) == 24
    for pair_specs in specs:
        assert len(pair_specs.first) == SIDE_SIZE
        assert len(pair_specs.second) == SIDE_SIZE
        assert int(pairing_population_size(pair_specs)) == POPULATION
        assert int(minimum_connected_pairings(pair_specs)) == NODES - 1
        assert int(default_pairing_count(pair_specs)) == DEFAULT_PAIRINGS_PER_PAIR


def test_population_counts_agree_with_brute_force_enumeration() -> None:
    cases = (
        ((Channel.USER,) * 4, (Channel.SYSTEM,) * 3),
        ((Channel.USER, Channel.SYSTEM, Channel.README), (Channel.USER, Channel.README)),
        ((Channel.SYSTEM, Channel.README), (Channel.USER,) * 5),
        ((Channel.USER,) * 2, (Channel.USER,) * 2),
    )
    for first_channels, second_channels in cases:
        pair_specs = _synthetic_pair_specs(first_channels, second_channels)
        expected = len(_brute_force_edges(pair_specs))
        assert int(pairing_population_size(pair_specs)) == expected


def test_default_pairing_count_is_capped_and_respects_connectivity() -> None:
    small = _synthetic_pair_specs((Channel.USER,) * 3, (Channel.SYSTEM,) * 2)
    # Population 6 is below the 720 default, so the population caps it, but the
    # result still has to reach the four edges that connect five cells.
    assert int(pairing_population_size(small)) == 6
    assert int(default_pairing_count(small)) == 6

    wide = _synthetic_pair_specs((Channel.USER,) * 12, (Channel.USER,) * 12)
    assert int(minimum_connected_pairings(wide)) == 23
    assert int(default_pairing_count(wide)) == 144


def test_rank_encoding_is_a_bijection_onto_the_valid_edges() -> None:
    pair_specs = _first_pair()
    sides = _sides(pair_specs)
    decoded = [_edge_from_rank(sides, rank) for rank in range(POPULATION)]

    assert len(set(decoded)) == POPULATION
    assert set(decoded) == _brute_force_edges(pair_specs)
    for rank, (first_index, second_index) in enumerate(decoded):
        assert _rank_from_edge(sides, first_index, second_index) == rank
        assert Channel.USER in (
            pair_specs.first[first_index].channel,
            pair_specs.second[second_index].channel,
        )


def test_rank_decoding_rejects_ranks_outside_the_population() -> None:
    sides = _sides(_first_pair())
    for rank in (-1, POPULATION, POPULATION + 1):
        with pytest.raises(ValueError, match="outside the valid population"):
            _edge_from_rank(sides, rank)


def test_rank_encoding_rejects_a_pairing_without_a_user_channel() -> None:
    pair_specs = _first_pair()
    sides = _sides(pair_specs)
    first_index = next(
        index
        for index, spec in enumerate(pair_specs.first)
        if spec.channel is not Channel.USER
    )
    second_index = next(
        index
        for index, spec in enumerate(pair_specs.second)
        if spec.channel is not Channel.USER
    )
    with pytest.raises(ValueError, match="must contain a user-channel"):
        _rank_from_edge(sides, first_index, second_index)


# --------------------------------------------------------------------------
# Strata and quotas
# --------------------------------------------------------------------------


def test_strata_partition_the_population_and_match_brute_force() -> None:
    pair_specs = _first_pair()
    sides = _sides(pair_specs)
    grouped = _ranks_by_stratum(sides)

    assert sum(len(ranks) for ranks in grouped.values()) == POPULATION
    assert len({rank for ranks in grouped.values() for rank in ranks}) == POPULATION
    # Five legal channel pairings times framing-differs times author-differs.
    assert len(grouped) == 20

    expected: Counter[tuple[str, str, bool, bool]] = Counter()
    for first_index, second_index in _brute_force_edges(pair_specs):
        first = pair_specs.first[first_index]
        second = pair_specs.second[second_index]
        expected[
            (
                str(first.channel),
                str(second.channel),
                first.framing != second.framing,
                first.author != second.author,
            )
        ] += 1
    actual = {
        (
            str(stratum.channels[0]),
            str(stratum.channels[1]),
            stratum.framing_differs,
            stratum.author_differs,
        ): len(ranks)
        for stratum, ranks in grouped.items()
    }
    assert actual == dict(expected)
    assert all(Channel.USER in stratum.channels for stratum in grouped)


def test_proportional_quotas_sum_to_the_request_and_break_ties_by_remainder() -> None:
    sides = _sides(_first_pair())
    grouped = _ranks_by_stratum(sides)
    counts = {stratum: len(ranks) for stratum, ranks in grouped.items()}

    for requested in (179, 360, 720, 1441, POPULATION - 1):
        quotas = _proportional_quotas(counts, requested, POPULATION)
        assert sum(quotas.values()) == requested
        assert set(quotas) == set(counts)
        for stratum, quota in quotas.items():
            assert quota <= counts[stratum]
            exact = requested * counts[stratum] / POPULATION
            # Largest-remainder rounding never moves a quota by a whole unit.
            assert abs(quota - exact) < 1.0


# --------------------------------------------------------------------------
# Sampling behaviour
# --------------------------------------------------------------------------


@pytest.mark.parametrize("pairings", [179, 180, 360, 720, 1440])
@pytest.mark.parametrize("seed", [0, 1, 17])
def test_samples_are_distinct_valid_connected_and_cover_every_cell(
    pairings: int, seed: int
) -> None:
    pair_specs = _first_pair()
    sampled = sample_pair_inputs(
        pair_specs, PositiveInteger.parse(pairings), Natural.parse(seed)
    )

    assert len(sampled) == pairings
    assert len({frozenset(inputs) for inputs in sampled}) == pairings
    first_side = set(pair_specs.first)
    second_side = set(pair_specs.second)
    for inputs in sampled:
        assert len(inputs) == 2
        assert any(spec.channel is Channel.USER for spec in inputs)
        assert {inputs[0] in first_side, inputs[1] in second_side} == {True}

    components = _components(pair_specs, sampled)
    assert len(components) == 1
    assert len(components[0]) == NODES


def test_minimum_request_produces_a_spanning_tree() -> None:
    pair_specs = _first_pair()
    sampled = sample_pair_inputs(
        pair_specs, PositiveInteger.parse(NODES - 1), Natural.parse(3)
    )
    # 179 edges over 180 connected cells can only be an acyclic spanning tree.
    assert len(sampled) == NODES - 1
    components = _components(pair_specs, sampled)
    assert len(components) == 1
    assert len(components[0]) == NODES


def test_full_population_request_returns_every_valid_edge() -> None:
    pair_specs = _first_pair()
    sampled = sample_pair_inputs(
        pair_specs, PositiveInteger.parse(POPULATION), Natural.parse(11)
    )
    assert len(sampled) == POPULATION
    assert {frozenset(inputs) for inputs in sampled} == {
        frozenset((pair_specs.first[first], pair_specs.second[second]))
        for first, second in _brute_force_edges(pair_specs)
    }


def test_sampling_is_reproducible_and_seed_sensitive() -> None:
    pair_specs = _first_pair()
    pairings = PositiveInteger.parse(720)
    baseline = sample_pair_inputs(pair_specs, pairings, Natural.parse(5))

    assert sample_pair_inputs(pair_specs, pairings, Natural.parse(5)) == baseline
    assert sample_pair_inputs(pair_specs, pairings, Natural.parse(6)) != baseline


def test_a_pair_design_does_not_depend_on_its_position_in_the_bank() -> None:
    """Seeds derive from the pair id, so reordering the bank changes nothing."""
    specs = _bank_specs()
    pairings = PositiveInteger.parse(720)
    direct = sample_pair_inputs(specs[3], pairings, Natural.parse(0))

    reordered = build_sampled_studies(
        (specs[3], specs[0]),
        (Assistant.INKLING,),
        pairings,
        PositiveInteger.parse(1),
        Natural.parse(0),
    )
    assert tuple(study.inputs for study in reordered[: len(direct)]) == direct


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_degree_is_balanced_within_each_channel(seed: int) -> None:
    pair_specs = _first_pair()
    sampled = sample_pair_inputs(
        pair_specs, PositiveInteger.parse(720), Natural.parse(seed)
    )
    degrees: Counter[PromptSpec] = Counter()
    for inputs in sampled:
        degrees.update(inputs)

    by_channel: dict[Channel, list[int]] = {channel: [] for channel in Channel}
    for spec in pair_specs.first + pair_specs.second:
        by_channel[spec.channel].append(degrees[spec])

    # Stratum quotas fix each channel's endpoint total, so the mean degree is
    # exact and identical for every seed. Only its spread is stochastic.
    means = {
        channel: sum(counts) / len(counts) for channel, counts in by_channel.items()
    }
    assert means == {
        Channel.USER: 14.4,
        Channel.SYSTEM: 4.8,
        Channel.README: 4.8,
    }

    # A greedy best-of-eight choice keeps the spread far below the roughly
    # 15-wide tail an unbalanced Poisson draw over 60 cells would produce.
    tolerances = {Channel.USER: 5, Channel.SYSTEM: 3, Channel.README: 3}
    for channel, counts in by_channel.items():
        assert len(counts) == 60
        assert min(counts) >= 1, f"{channel} left a cell uncovered"
        assert max(counts) - min(counts) <= tolerances[channel], (
            f"{channel} degrees are lopsided: {sorted(counts)}"
        )

    # User-channel cells reach every partner while others only reach user-channel
    # partners, so their degrees stay proportional rather than equal.
    assert min(by_channel[Channel.USER]) > max(by_channel[Channel.SYSTEM])
    assert min(by_channel[Channel.USER]) > max(by_channel[Channel.README])


# --------------------------------------------------------------------------
# Connectivity repair
# --------------------------------------------------------------------------


def _ranks_touching_second_below(sides: sampling_module._Sides, limit: int) -> set[int]:
    return {
        rank
        for rank in range(sides.population)
        if _edge_from_rank(sides, rank)[1] < limit
    }


def _ranks_touching_first_below(sides: sampling_module._Sides, limit: int) -> set[int]:
    return {
        rank
        for rank in range(sides.population)
        if _edge_from_rank(sides, rank)[0] < limit
    }


def test_repair_reconnects_isolated_second_side_cells() -> None:
    """Isolated cells opposite the anchor attach to it directly."""
    sides = _sides(_first_pair())
    selected = _ranks_touching_second_below(sides, 10)
    assert len(selected) == 600

    repaired = _repair_connectivity(sides, set(selected), Random(0))

    assert len(repaired) == len(selected)
    forest = sampling_module._DisjointSet(sides.node_count)
    for rank in repaired:
        forest.union(*sampling_module._edge_nodes(sides, rank))
    assert len({forest.find(node) for node in range(sides.node_count)}) == 1


def test_repair_reconnects_isolated_first_side_cells() -> None:
    """Isolated cells on the anchor's own side route through the opposite side."""
    sides = _sides(_first_pair())
    selected = _ranks_touching_first_below(sides, 10)
    assert len(selected) == 600

    repaired = _repair_connectivity(sides, set(selected), Random(1))

    assert len(repaired) == len(selected)
    forest = sampling_module._DisjointSet(sides.node_count)
    for rank in repaired:
        forest.union(*sampling_module._edge_nodes(sides, rank))
    assert len({forest.find(node) for node in range(sides.node_count)}) == 1


def test_repair_leaves_an_already_connected_selection_untouched() -> None:
    pair_specs = _first_pair()
    sides = _sides(pair_specs)
    connected = {
        _rank_from_edge(sides, sides.user_first[0], index)
        for index in range(len(pair_specs.second))
    } | {
        _rank_from_edge(sides, index, sides.user_second[0])
        for index in range(len(pair_specs.first))
    }
    repaired = _repair_connectivity(sides, set(connected), Random(0))
    assert repaired == connected


def test_repair_prefers_removing_an_edge_from_the_bridge_stratum() -> None:
    sides = _sides(_first_pair())
    selected = _ranks_touching_second_below(sides, 10)
    repaired = _repair_connectivity(sides, set(selected), Random(7))

    added = repaired - selected
    removed = selected - repaired
    assert len(added) == len(removed)
    added_strata = Counter(_stratum_for_rank(sides, rank) for rank in added)
    removed_strata = Counter(_stratum_for_rank(sides, rank) for rank in removed)
    # Every bridge stratum that had a spare cycle edge gave one up, so the
    # stratum profile of the design is preserved wherever it could be.
    assert sum((added_strata & removed_strata).values()) >= len(added) // 2


def _second_side_anchor_pair(first_channels: tuple[Channel, ...], users: int) -> PairSpecs:
    """Build a pair whose only user-channel cells sit on the second side."""
    return _synthetic_pair_specs(first_channels, (Channel.USER,) * users, "anchor-probe")


def test_repair_reconnects_when_the_anchor_sits_on_the_second_side() -> None:
    """With no user-channel cell on the first side the anchor flips sides."""
    pair_specs = _second_side_anchor_pair((Channel.SYSTEM, Channel.README), 3)
    sides = _sides(pair_specs)
    assert sides.user_first == ()
    assert sampling_module._anchor_node(sides) == len(pair_specs.first)

    # Leave the last second-side cell isolated, on the anchor's own side.
    selected = {
        _rank_from_edge(sides, first, second) for first in range(2) for second in range(2)
    }
    assert len(selected) == 4

    repaired = _repair_connectivity(sides, set(selected), Random(0))

    assert len(repaired) == len(selected)
    forest = sampling_module._DisjointSet(sides.node_count)
    for rank in repaired:
        forest.union(*sampling_module._edge_nodes(sides, rank))
    assert len({forest.find(node) for node in range(sides.node_count)}) == 1


def test_repair_reconnects_an_isolated_first_side_cell_opposite_the_anchor() -> None:
    pair_specs = _second_side_anchor_pair((Channel.SYSTEM, Channel.README, Channel.SYSTEM), 4)
    sides = _sides(pair_specs)
    assert sides.user_first == ()

    # Every edge of the first two first-side cells, leaving the third isolated.
    selected = {
        _rank_from_edge(sides, first, second) for first in range(2) for second in range(4)
    }
    assert len(selected) == 8

    repaired = _repair_connectivity(sides, set(selected), Random(2))

    assert len(repaired) == len(selected)
    forest = sampling_module._DisjointSet(sides.node_count)
    for rank in repaired:
        forest.union(*sampling_module._edge_nodes(sides, rank))
    assert len({forest.find(node) for node in range(sides.node_count)}) == 1


def test_sampling_works_end_to_end_with_a_second_side_anchor() -> None:
    pair_specs = _second_side_anchor_pair((Channel.SYSTEM, Channel.README, Channel.SYSTEM), 4)
    sampled = sample_pair_inputs(pair_specs, PositiveInteger.parse(9), Natural.parse(0))

    assert len(sampled) == 9
    components = _components(pair_specs, sampled)
    assert len(components) == 1
    assert len(components[0]) == 7


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------


def test_single_rejects_a_ragged_design() -> None:
    assert sample_studies_module._single((4500, 4500), "pairing population") == 4500
    with pytest.raises(ValueError, match="must share the same pairing population"):
        sample_studies_module._single((4500, 900), "pairing population")


@pytest.mark.parametrize(
    ("first_channels", "second_channels", "error"),
    [
        ((), (Channel.USER,), "at least one specification"),
        ((Channel.USER,), (), "at least one specification"),
        ((Channel.SYSTEM,), (Channel.README,), "user message channel"),
        (
            (Channel.USER, Channel.SYSTEM),
            (Channel.SYSTEM, Channel.README),
            "when the second side offers none",
        ),
        (
            (Channel.SYSTEM, Channel.README),
            (Channel.USER, Channel.SYSTEM),
            "when the first side offers none",
        ),
    ],
)
def test_sides_reject_unusable_channel_mixes(
    first_channels: tuple[Channel, ...],
    second_channels: tuple[Channel, ...],
    error: str,
) -> None:
    pair_specs = _synthetic_pair_specs(first_channels, second_channels)
    with pytest.raises(ValueError, match=error):
        _sides(pair_specs)


def test_sides_reject_duplicate_and_overlapping_specifications() -> None:
    pair_specs = _first_pair()
    duplicated = PairSpecs(
        pair_specs.pair,
        pair_specs.first + (pair_specs.first[0],),
        pair_specs.second,
    )
    with pytest.raises(ValueError, match="unique within each side"):
        _sides(duplicated)

    overlapping = PairSpecs(pair_specs.pair, pair_specs.first, pair_specs.first)
    with pytest.raises(ValueError, match="must not share specifications"):
        _sides(overlapping)


def test_sampling_rejects_requests_outside_the_valid_range() -> None:
    pair_specs = _first_pair()
    with pytest.raises(ValueError, match="at least 179"):
        sample_pair_inputs(pair_specs, PositiveInteger.parse(178), Natural.parse(0))
    with pytest.raises(ValueError, match="cannot exceed the valid population"):
        sample_pair_inputs(
            pair_specs, PositiveInteger.parse(POPULATION + 1), Natural.parse(0)
        )


# --------------------------------------------------------------------------
# Suite construction
# --------------------------------------------------------------------------


def test_studies_share_one_design_across_assistants() -> None:
    specs = _bank_specs()[:2]
    assistants = (Assistant.INKLING, Assistant.QWEN3_8_FLASH)
    studies = build_sampled_studies(
        specs,
        assistants,
        PositiveInteger.parse(200),
        PositiveInteger.parse(1),
        Natural.parse(0),
    )

    assert len(studies) == 2 * 2 * 200
    by_assistant: dict[Assistant, list[StudyInputs]] = {}
    for study in studies:
        by_assistant.setdefault(study.assistant, []).append(study.inputs)
    assert set(by_assistant) == set(assistants)
    designs = [tuple(inputs) for inputs in by_assistant.values()]
    assert designs[0] == designs[1]


def test_full_pilot_design_has_the_expected_size() -> None:
    studies = build_sampled_studies(
        _bank_specs(),
        tuple(Assistant),
        PositiveInteger.parse(DEFAULT_PAIRINGS_PER_PAIR),
        PositiveInteger.parse(1),
        Natural.parse(0),
    )
    assert len(studies) == 24 * DEFAULT_PAIRINGS_PER_PAIR * len(Assistant)
    assert len(studies) == 69_120
    assert len(set(studies)) == len(studies)
    # Two orderings per study, one rollout each.
    assert 2 * len(studies) == 138_240


@pytest.mark.parametrize(
    ("pair_specs", "assistants", "error"),
    [
        ((), (Assistant.INKLING,), "at least one instruction pair"),
        (None, (), "at least one assistant"),
        (None, (Assistant.INKLING, Assistant.INKLING), "assistants must be unique"),
    ],
)
def test_suite_construction_rejects_invalid_inputs(
    pair_specs: tuple[PairSpecs, ...] | None,
    assistants: tuple[Assistant, ...],
    error: str,
) -> None:
    specs = _bank_specs()[:1] if pair_specs is None else pair_specs
    with pytest.raises(ValueError, match=error):
        build_sampled_studies(
            specs,
            assistants,
            PositiveInteger.parse(200),
            PositiveInteger.parse(1),
            Natural.parse(0),
        )


def test_suite_construction_rejects_duplicate_pair_ids() -> None:
    duplicated = (_first_pair(), _first_pair())
    with pytest.raises(ValueError, match="pair ids must be unique"):
        build_sampled_studies(
            duplicated,
            (Assistant.INKLING,),
            PositiveInteger.parse(200),
            PositiveInteger.parse(1),
            Natural.parse(0),
        )


def test_study_suite_round_trip_and_validation(tmp_path: Path) -> None:
    studies = build_sampled_studies(
        _bank_specs()[:1],
        (Assistant.INKLING,),
        PositiveInteger.parse(200),
        PositiveInteger.parse(1),
        Natural.parse(0),
    )
    path = tmp_path / "suite.yaml"
    write_study_suite(path, studies)
    assert load_study_suite(path) == studies

    with pytest.raises(ValueError, match="at least one"):
        write_study_suite(path, ())
    with pytest.raises(ValueError, match="distinct"):
        write_study_suite(path, (studies[0], studies[0]))

    path.write_text("wrong: []\n")
    with pytest.raises(ValueError, match="exactly one"):
        load_study_suite(path)
    path.write_text("studies: []\n")
    with pytest.raises(ValueError, match="at least one"):
        load_study_suite(path)
    path.write_text(yaml.safe_dump({"studies": [study_to_dict(studies[0])] * 2}))
    with pytest.raises(ValueError, match="distinct"):
        load_study_suite(path)


# --------------------------------------------------------------------------
# Command line interface
# --------------------------------------------------------------------------


def test_sample_studies_cli_writes_a_loadable_suite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "suite.yaml"
    assert (
        sample_studies(
            [
                "--pairs",
                str(BANK),
                "--output",
                str(output),
                "--pairings-per-pair",
                "200",
                "--assistant",
                "Inkling",
                "--author",
                "Inkling",
                "--author",
                "Inkling Small",
            ]
        )
        == 0
    )

    summary = json.loads(capsys.readouterr().out)
    assert summary["instruction_pairs"] == 24
    assert summary["pairings_per_pair"] == 200
    assert summary["studies"] == 24 * 200
    assert summary["trials"] == 2 * 24 * 200
    assert summary["authors"] == ["Inkling", "Inkling Small"]
    # Two authors leaves 6 framings x 3 channels x 2 authors per side.
    assert summary["specs"] == 24 * 2 * 36
    assert summary["pairing_population_per_pair"] == 36 * 36 - 24 * 24
    assert summary["minimum_connected_pairings_per_pair"] == 71

    studies = load_study_suite(output)
    assert len(studies) == 24 * 200
    for study in studies:
        assert study.assistant is Assistant.INKLING
        assert {spec.author for spec in study.inputs} <= {
            Author.INKLING,
            Author.INKLING_SMALL,
        }


def test_sample_studies_cli_uses_the_pilot_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "suite.yaml"
    assert (
        sample_studies(
            [
                "--pairs",
                str(BANK),
                "--output",
                str(output),
                "--assistant",
                "Inkling",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["pairings_per_pair"] == DEFAULT_PAIRINGS_PER_PAIR
    assert summary["pairing_population_per_pair"] == POPULATION
    assert summary["studies"] == 24 * DEFAULT_PAIRINGS_PER_PAIR


@pytest.mark.parametrize(
    "arguments",
    [
        ["--pairings-per-pair", "0"],
        ["--pairings-per-pair", "178"],
        ["--pairings-per-pair", "4501"],
        ["--rollouts-per-permutation", "0"],
        ["--seed", "-1"],
        ["--author", "Inkling", "--author", "Inkling"],
    ],
)
def test_sample_studies_cli_reports_invalid_values(
    tmp_path: Path, arguments: list[str]
) -> None:
    with pytest.raises(SystemExit, match="2"):
        sample_studies(
            ["--pairs", str(BANK), "--output", str(tmp_path / "suite.yaml"), *arguments]
        )


def test_sample_studies_cli_reports_a_missing_bank(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="2"):
        sample_studies(
            [
                "--pairs",
                str(tmp_path / "missing.yaml"),
                "--output",
                str(tmp_path / "suite.yaml"),
            ]
        )
