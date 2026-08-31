from __future__ import annotations

import itertools
from dataclasses import replace
from typing import cast

import pytest

from reasonese.axes import Author, Channel, Framing, Instruction
from reasonese.planning import PromptSpec, build_prompt_specs, specs_per_instruction


def _instruction(identifier: str = "write_program") -> Instruction:
    return Instruction(id=identifier, text="Write a program that prints hello.")


def test_plan_is_the_complete_cartesian_product() -> None:
    instruction = _instruction()
    specs = build_prompt_specs((instruction,))

    assert specs_per_instruction() == 90
    assert len(specs) == 90
    assert len({spec.spec_id for spec in specs}) == 90
    assert {(spec.framing, spec.channel, spec.author) for spec in specs} == set(
        itertools.product(Framing, Channel, Author)
    )
    assert all(spec.instruction_id == instruction.id for spec in specs)
    assert specs[0].framing is Framing.NORMAL
    assert specs[0].channel is Channel.SYSTEM
    assert specs[0].author is Author.USER


def test_plan_scales_independently_with_instructions() -> None:
    first = _instruction()
    second = Instruction(id="find_information", text="Find information about pathlib.")
    specs = build_prompt_specs((first, second))

    assert len(specs) == 180
    assert sum(spec.instruction_id == first.id for spec in specs) == 90
    assert sum(spec.instruction_id == second.id for spec in specs) == 90


def test_spec_ids_are_deterministic_and_content_addressed() -> None:
    instruction = _instruction()
    baseline = PromptSpec.create(instruction, Framing.NORMAL, Channel.USER, Author.USER)
    repeated = PromptSpec.create(instruction, Framing.NORMAL, Channel.USER, Author.USER)
    changed_author = PromptSpec.create(instruction, Framing.NORMAL, Channel.USER, Author.INKLING)
    changed_text = PromptSpec.create(
        Instruction(id=instruction.id, text="Write a different program."),
        Framing.NORMAL,
        Channel.USER,
        Author.USER,
    )

    assert baseline == repeated
    assert len(baseline.spec_id) == 20
    assert baseline.spec_id != changed_author.spec_id
    assert baseline.spec_id != changed_text.spec_id


def test_prompt_spec_round_trips_through_strict_wire_format() -> None:
    spec = PromptSpec.create(
        _instruction(), Framing.REASONESE_PERSUASIVE, Channel.README, Author.INKLING_SMALL
    )
    assert PromptSpec.from_dict(spec.to_dict()) == spec
    assert spec.to_dict() == {
        "schema_version": 1,
        "spec_id": spec.spec_id,
        "instruction_id": "write_program",
        "instruction": "Write a program that prints hello.",
        "framing": "reasonese-persuasive",
        "channel": "readme",
        "author": "inkling_small",
    }


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"extra": True}, "extra=\\['extra'\\]"),
        ({"remove": "author"}, "missing=\\['author'\\]"),
    ],
)
def test_prompt_spec_rejects_unknown_or_missing_keys(
    mutation: dict[str, object], match: str
) -> None:
    payload: dict[str, object] = dict(
        PromptSpec.create(_instruction(), Framing.NORMAL, Channel.USER, Author.USER).to_dict()
    )
    if "remove" in mutation:
        payload.pop(str(mutation["remove"]))
    else:
        payload.update(mutation)

    with pytest.raises(ValueError, match=match):
        PromptSpec.from_dict(payload)


def test_prompt_spec_rejects_invalid_schema_id_and_enum() -> None:
    spec = PromptSpec.create(_instruction(), Framing.NORMAL, Channel.USER, Author.USER)
    with pytest.raises(ValueError, match="unsupported schema version"):
        replace(spec, schema_version=2)
    with pytest.raises(ValueError, match="spec_id must be"):
        replace(spec, spec_id="tampered")

    payload = spec.to_dict()
    payload["framing"] = "unknown"
    with pytest.raises(ValueError, match="unknown"):
        PromptSpec.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("schema_version", "1", "schema_version must be an integer"),
        ("spec_id", 1, "spec_id must be a string"),
        ("instruction_id", 1, "instruction_id must be a string"),
        ("instruction", 1, "instruction must be a string"),
        ("framing", 1, "framing must be a string"),
        ("channel", 1, "channel must be a string"),
        ("author", 1, "author must be a string"),
    ],
)
def test_prompt_spec_rejects_non_string_wire_values(field: str, value: object, match: str) -> None:
    spec = PromptSpec.create(_instruction(), Framing.NORMAL, Channel.USER, Author.USER)
    payload: dict[str, object] = dict(spec.to_dict())
    payload[field] = value
    with pytest.raises(ValueError, match=match):
        PromptSpec.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "match"),
    [
        ("framing", "framing must be a Framing value"),
        ("channel", "channel must be a Channel value"),
        ("author", "author must be an Author value"),
    ],
)
def test_direct_prompt_spec_construction_validates_enum_types(field: str, match: str) -> None:
    spec = PromptSpec.create(_instruction(), Framing.NORMAL, Channel.USER, Author.USER)
    with pytest.raises(ValueError, match=match):
        replace(spec, **{field: cast(object, "invalid")})


def test_plan_rejects_empty_or_duplicate_instruction_sets() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_prompt_specs(())
    with pytest.raises(ValueError, match="unique"):
        build_prompt_specs((_instruction(), _instruction()))
