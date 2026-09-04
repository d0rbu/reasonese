from __future__ import annotations

import itertools
from pathlib import Path
from typing import cast

import pytest
from beartype.roar import BeartypeCallHintParamViolation
from phantom.interval import Natural

from reasonese.axes import Author, Channel, Framing, Instruction
from reasonese.instructions import (
    ConflictType,
    InstructionPair,
    PairId,
    Rationale,
    Skill,
    load_instruction_pairs,
)
from reasonese.planning import (
    PromptSpec,
    build_pair_specs,
    build_prompt_specs,
    specs_per_instruction,
)


def _pair(pair_id: str, first: str, second: str) -> InstructionPair:
    return InstructionPair(
        PairId.parse(pair_id),
        Skill.PYTHON,
        ConflictType.OUTPUT_FORMAT,
        Instruction.parse(first),
        Instruction.parse(second),
        Rationale.parse("The two requests cannot both be satisfied."),
    )


def test_specs_per_instruction_is_a_non_negative_integer() -> None:
    count = specs_per_instruction()
    assert count == 90
    assert isinstance(count, Natural)


def test_plan_is_the_complete_cartesian_product() -> None:
    instruction = Instruction.parse("Write a program.")
    specs = build_prompt_specs((instruction,))

    assert len(specs) == specs_per_instruction()
    assert {(spec.framing, spec.channel, spec.author) for spec in specs} == set(
        itertools.product(Framing, Channel, Author)
    )
    assert all(spec.instruction == instruction for spec in specs)
    assert specs[0] == PromptSpec(
        instruction,
        Framing.NORMAL,
        Channel.SYSTEM,
        Author.USER,
    )


def test_plan_requires_unique_instructions() -> None:
    instruction = Instruction.parse("Write a program.")
    with pytest.raises(ValueError, match="at least one"):
        build_prompt_specs(())
    with pytest.raises(ValueError, match="unique"):
        build_prompt_specs((instruction, instruction))


def test_pair_specs_enumerate_both_sides_of_the_real_bank() -> None:
    pairs = load_instruction_pairs(Path("configs/instruction_pairs.yaml"))
    pair_specs = build_pair_specs(pairs)

    assert len(pair_specs) == len(pairs) == 24
    for pair, specs in zip(pairs, pair_specs, strict=True):
        assert specs.pair is pair
        assert len(specs.first) == len(specs.second) == specs_per_instruction()
        assert {spec.instruction for spec in specs.first} == {pair.first}
        assert {spec.instruction for spec in specs.second} == {pair.second}
        assert not set(specs.first) & set(specs.second)
        for side in (specs.first, specs.second):
            assert {(spec.framing, spec.channel, spec.author) for spec in side} == set(
                itertools.product(Framing, Channel, Author)
            )

    every_spec = [spec for item in pair_specs for spec in item.first + item.second]
    assert len(every_spec) == 24 * 2 * 90
    assert len(set(every_spec)) == len(every_spec)


def test_pair_specs_reject_an_instruction_shared_by_two_pairs() -> None:
    shared = "Compute the answer."
    pairs = (
        _pair("first-pair", shared, "Do something else."),
        _pair("second-pair", shared, "Do a third thing."),
    )
    with pytest.raises(ValueError, match="appears in more than one pair"):
        build_pair_specs(pairs)


def test_pair_specs_require_at_least_one_pair() -> None:
    with pytest.raises(ValueError, match="at least one instruction pair"):
        build_pair_specs(())


def test_prompt_spec_is_runtime_type_checked() -> None:
    with pytest.raises(BeartypeCallHintParamViolation):
        PromptSpec(
            Instruction.parse("Write a program."),
            cast(Framing, "normal"),
            Channel.SYSTEM,
            Author.USER,
        )
