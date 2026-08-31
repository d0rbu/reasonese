from __future__ import annotations

from typing import cast

import pytest

from reasonese.axes import (
    AUTHOR_LABELS,
    CHANNEL_LABELS,
    FRAMING_DESCRIPTIONS,
    Author,
    Channel,
    Framing,
    Instruction,
    axis_manifest,
    validate_identifier,
)


def test_axis_values_and_order_are_explicit() -> None:
    assert [framing.value for framing in Framing] == [
        "normal",
        "casual",
        "persuasive",
        "subagent",
        "reasonese-normal",
        "reasonese-persuasive",
    ]
    assert [channel.value for channel in Channel] == ["system", "user", "readme"]
    assert [author.value for author in Author] == [
        "user",
        "qwen3_8_flash",
        "qwen3_8_2_4t",
        "inkling",
        "inkling_small",
    ]


def test_human_readable_axis_metadata_is_complete() -> None:
    assert set(FRAMING_DESCRIPTIONS) == set(Framing)
    assert CHANNEL_LABELS[Channel.README] == "README.md"
    assert AUTHOR_LABELS[Author.QWEN3_8_2_4T] == "Qwen3.8 2.4T"

    manifest = axis_manifest()
    assert manifest["instruction"] == {
        "description": "An author-independent base task.",
        "source": "configured",
    }
    assert [entry["id"] for entry in manifest["framing"]] == [framing.value for framing in Framing]
    assert [entry["label"] for entry in manifest["channel"]] == [
        CHANNEL_LABELS[channel] for channel in Channel
    ]
    assert [entry["label"] for entry in manifest["author"]] == [
        AUTHOR_LABELS[author] for author in Author
    ]


def test_instruction_accepts_a_stable_id_and_trimmed_text() -> None:
    instruction = Instruction(id="write_parser_2", text="Write a parser.")
    assert instruction.id == "write_parser_2"
    assert validate_identifier("valid_id") == "valid_id"


@pytest.mark.parametrize("identifier", ["", "UPPER", "two-words", "2start", "space id"])
def test_instruction_rejects_invalid_ids(identifier: str) -> None:
    with pytest.raises(ValueError, match="instruction id must match"):
        Instruction(id=identifier, text="Do the task.")


@pytest.mark.parametrize("text", ["", " leading", "trailing ", "\nmultiline\n"])
def test_instruction_rejects_empty_or_untrimmed_text(text: str) -> None:
    with pytest.raises(ValueError, match="instruction text"):
        Instruction(id="valid", text=text)


def test_identifier_error_names_the_requested_field() -> None:
    with pytest.raises(ValueError, match="custom field must match"):
        validate_identifier("not-valid", field="custom field")


def test_instruction_rejects_non_string_boundary_values() -> None:
    with pytest.raises(ValueError, match="instruction id must be a string"):
        Instruction(id=cast(str, 1), text="Do the task.")
    with pytest.raises(ValueError, match="instruction text must not be empty"):
        Instruction(id="task", text=cast(str, 1))
