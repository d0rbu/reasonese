from __future__ import annotations

import itertools
from typing import cast

import pytest
from beartype.roar import BeartypeCallHintParamViolation
from phantom.interval import Natural

from reasonese.axes import Author, Channel, Framing, Instruction, author_framings
from reasonese.planning import PromptSpec, build_prompt_specs, specs_per_instruction


def test_specs_per_instruction_is_a_non_negative_integer() -> None:
    count = specs_per_instruction()
    assert count == 81
    assert isinstance(count, Natural)


def test_plan_is_the_cartesian_product_restricted_to_author_framings() -> None:
    instruction = Instruction.parse("Write a program.")
    specs = build_prompt_specs((instruction,))

    assert len(specs) == specs_per_instruction()
    assert {(spec.framing, spec.channel, spec.author) for spec in specs} == {
        (framing, channel, author)
        for framing, channel, author in itertools.product(Framing, Channel, Author)
        if framing in author_framings(author)
    }
    assert {spec.framing for spec in specs if spec.author is Author.USER} == {
        Framing.NORMAL,
        Framing.CASUAL,
        Framing.PERSUASIVE,
    }
    assert {spec.framing for spec in specs if spec.author is Author.INKLING} == set(Framing)
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


def test_prompt_spec_rejects_model_only_framings_for_the_user_author() -> None:
    instruction = Instruction.parse("Write a program.")
    for framing in (Framing.SUBAGENT, Framing.REASONESE_NORMAL, Framing.REASONESE_PERSUASIVE):
        with pytest.raises(ValueError, match="user author does not write"):
            PromptSpec(instruction, framing, Channel.USER, Author.USER)
        assert PromptSpec(instruction, framing, Channel.USER, Author.INKLING).framing is framing


def test_prompt_spec_is_runtime_type_checked() -> None:
    with pytest.raises(BeartypeCallHintParamViolation):
        PromptSpec(
            Instruction.parse("Write a program."),
            cast(Framing, "normal"),
            Channel.SYSTEM,
            Author.USER,
        )
