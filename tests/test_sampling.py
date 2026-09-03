from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from phantom.interval import Natural

from reasonese.axes import Assistant, Author, Channel, Framing, Instruction
from reasonese.config import load_study_suite
from reasonese.io import write_study_suite
from reasonese.planning import PromptSpec, build_prompt_specs
from reasonese.sample_studies import main as sample_studies
from reasonese.sampling import (
    build_sampled_studies,
    default_pairing_count,
    minimum_connected_pairings,
    pairing_population_size,
    sample_study_inputs,
)
from reasonese.study import PositiveInteger, StudyInputs, study_to_dict


def _spec(name: str, channel: Channel) -> PromptSpec:
    return PromptSpec(
        Instruction.parse(name),
        Framing.NORMAL,
        channel,
        Author.INKLING,
    )


def _specs() -> tuple[PromptSpec, ...]:
    return (
        _spec("A", Channel.USER),
        _spec("B", Channel.USER),
        _spec("C", Channel.SYSTEM),
        _spec("D", Channel.README),
        _spec("E", Channel.SYSTEM),
    )


def _reachable(pairings: tuple[StudyInputs, ...]) -> set[PromptSpec]:
    reached = {pairings[0][0]}
    changed = True
    while changed:
        changed = False
        for pairing in pairings:
            edge = set(pairing)
            if edge & reached and not edge <= reached:
                reached.update(edge)
                changed = True
    return reached


def test_pairing_counts_respect_the_user_channel_constraint() -> None:
    specs = _specs()
    assert pairing_population_size(specs) == 7
    assert minimum_connected_pairings(specs) == 4

    all_user = (_spec("A", Channel.USER), _spec("B", Channel.USER), _spec("C", Channel.USER))
    assert pairing_population_size(all_user) == 3
    assert minimum_connected_pairings(all_user) == 2


def test_twenty_instruction_design_size() -> None:
    instructions = tuple(Instruction.parse(f"Task {index}.") for index in range(20))
    specs = build_prompt_specs(instructions)
    assert len(specs) == 1_800
    assert pairing_population_size(specs) == 899_700
    assert minimum_connected_pairings(specs) == 1_799
    assert default_pairing_count(specs) == 20_000


def test_default_pairing_count_is_capped_and_preserves_connectivity() -> None:
    assert default_pairing_count(_specs()) == pairing_population_size(_specs())

    instructions = tuple(Instruction.parse(f"Task {index}.") for index in range(250))
    specs = build_prompt_specs(instructions)
    assert minimum_connected_pairings(specs) == 22_499
    assert default_pairing_count(specs) == 22_499


def test_seeded_sample_is_unique_connected_and_order_independent() -> None:
    specs = _specs()
    first = sample_study_inputs(specs, PositiveInteger.parse(5), Natural.parse(7))
    repeated = sample_study_inputs(
        tuple(reversed(specs)), PositiveInteger.parse(5), Natural.parse(7)
    )
    different_seed = sample_study_inputs(specs, PositiveInteger.parse(5), Natural.parse(8))

    assert first == repeated
    assert first != different_seed
    assert len(first) == len({frozenset(pairing) for pairing in first}) == 5
    assert _reachable(first) == set(specs)
    assert all(any(spec.channel is Channel.USER for spec in pairing) for pairing in first)


def test_minimum_and_exhaustive_samples_cover_edge_cases() -> None:
    specs = _specs()
    minimum = sample_study_inputs(specs, PositiveInteger.parse(4), Natural.parse(0))
    exhaustive = sample_study_inputs(specs, PositiveInteger.parse(7), Natural.parse(99))
    assert _reachable(minimum) == set(specs)
    assert {frozenset(pairing) for pairing in exhaustive} == {
        frozenset((first, second))
        for index, first in enumerate(specs)
        for second in specs[index + 1 :]
        if first.channel is Channel.USER or second.channel is Channel.USER
    }

    one_user = (_spec("A", Channel.USER), _spec("B", Channel.SYSTEM))
    assert sample_study_inputs(
        one_user, PositiveInteger.parse(1), Natural.parse(0)
    ) == (StudyInputs.parse(one_user),)

    all_user = tuple(_spec(name, Channel.USER) for name in ("A", "B", "C", "D"))
    all_user_pairs = sample_study_inputs(
        all_user, PositiveInteger.parse(6), Natural.parse(0)
    )
    assert {frozenset(pairing) for pairing in all_user_pairs} == {
        frozenset((first, second))
        for index, first in enumerate(all_user)
        for second in all_user[index + 1 :]
    }


@pytest.mark.parametrize(
    ("specs", "error"),
    [
        ((_spec("A", Channel.USER),), "at least two"),
        (
            (_spec("A", Channel.USER), _spec("A", Channel.USER)),
            "must be unique",
        ),
        (
            (_spec("A", Channel.SYSTEM), _spec("B", Channel.README)),
            "user message",
        ),
    ],
)
def test_sampling_rejects_invalid_spec_sets(
    specs: tuple[PromptSpec, ...], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        pairing_population_size(specs)


def test_sampling_rejects_disconnected_or_oversized_requests() -> None:
    specs = _specs()
    with pytest.raises(ValueError, match="at least 4"):
        sample_study_inputs(specs, PositiveInteger.parse(3), Natural.parse(0))
    with pytest.raises(ValueError, match="cannot exceed"):
        sample_study_inputs(specs, PositiveInteger.parse(8), Natural.parse(0))


def test_sampled_studies_share_pairings_across_assistants() -> None:
    specs = _specs()
    studies = build_sampled_studies(
        specs,
        (Assistant.INKLING, Assistant.INKLING_SMALL),
        PositiveInteger.parse(5),
        PositiveInteger.parse(2),
        Natural.parse(4),
    )
    assert len(studies) == 10
    assert tuple(study.inputs for study in studies[:5]) == tuple(
        study.inputs for study in studies[5:]
    )
    assert {study.assistant for study in studies} == {
        Assistant.INKLING,
        Assistant.INKLING_SMALL,
    }
    assert all(study.rollouts_per_permutation == 2 for study in studies)

    with pytest.raises(ValueError, match="at least one assistant"):
        build_sampled_studies(
            specs,
            (),
            PositiveInteger.parse(4),
            PositiveInteger.parse(1),
            Natural.parse(0),
        )
    with pytest.raises(ValueError, match="must be unique"):
        build_sampled_studies(
            specs,
            (Assistant.INKLING, Assistant.INKLING),
            PositiveInteger.parse(4),
            PositiveInteger.parse(1),
            Natural.parse(0),
        )


def test_study_suite_round_trip_and_validation(tmp_path: Path) -> None:
    studies = build_sampled_studies(
        _specs(),
        (Assistant.INKLING,),
        PositiveInteger.parse(4),
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
    path.write_text(
        yaml.safe_dump({"studies": [study_to_dict(studies[0])] * 2})
    )
    with pytest.raises(ValueError, match="distinct"):
        load_study_suite(path)


def test_sample_studies_cli_writes_filtered_suite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    instructions = tmp_path / "instructions.toml"
    instructions.write_text('instructions = ["Do the task."]\n')
    output = tmp_path / "suite.yaml"

    assert (
        sample_studies(
            [
                "--instructions",
                str(instructions),
                "--output",
                str(output),
                "--pairings-per-assistant",
                "17",
                "--rollouts-per-permutation",
                "2",
                "--seed",
                "11",
                "--author",
                str(Author.USER),
                "--assistant",
                str(Assistant.INKLING),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    studies = load_study_suite(output)
    assert summary == {
        "assistants": ["Inkling"],
        "authors": ["user"],
        "instructions": 1,
        "minimum_connected_pairings_per_assistant": 17,
        "output": str(output),
        "pairing_population_per_assistant": 87,
        "pairings_per_assistant": 17,
        "rollouts_per_permutation": 2,
        "seed": 11,
        "specs": 18,
        "studies": 17,
        "trials": 68,
    }
    assert len(studies) == 17
    assert {study.assistant for study in studies} == {Assistant.INKLING}


def test_sample_studies_cli_uses_capped_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    instructions = tmp_path / "instructions.toml"
    instructions.write_text('instructions = ["Do the task."]\n')
    output = tmp_path / "suite.yaml"

    assert (
        sample_studies(
            [
                "--instructions",
                str(instructions),
                "--output",
                str(output),
                "--author",
                str(Author.USER),
                "--assistant",
                str(Assistant.INKLING),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["pairing_population_per_assistant"] == 87
    assert summary["pairings_per_assistant"] == 87
    assert len(load_study_suite(output)) == 87


def test_sample_studies_cli_reports_invalid_values(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    instructions = tmp_path / "instructions.toml"
    instructions.write_text('instructions = ["Do the task."]\n')
    common = [
        "--instructions",
        str(instructions),
        "--output",
        str(tmp_path / "suite.yaml"),
        "--pairings-per-assistant",
        "1",
    ]
    with pytest.raises(SystemExit, match="2"):
        sample_studies(common)
    assert "pairings must be at least" in capsys.readouterr().err

    with pytest.raises(SystemExit, match="2"):
        sample_studies(
            common
            + [
                "--author",
                str(Author.USER),
                "--author",
                str(Author.USER),
            ]
        )
    assert "authors must be unique" in capsys.readouterr().err
