from __future__ import annotations

import itertools
from typing import cast

import pytest
from beartype.roar import BeartypeCallHintParamViolation
from phantom.interval import Natural

from reasonese.axes import Author, Channel, Framing, Instruction
from reasonese.planning import PromptSpec, build_prompt_specs, specs_per_instruction


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


def test_prompt_spec_is_runtime_type_checked() -> None:
    with pytest.raises(BeartypeCallHintParamViolation):
        PromptSpec(
            Instruction.parse("Write a program."),
            cast(Framing, "normal"),
            Channel.SYSTEM,
            Author.USER,
        )
