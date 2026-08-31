from __future__ import annotations

import pytest

from reasonese.axes import Author, Channel, Framing, Instruction, axis_manifest


def test_axis_values_are_their_display_strings() -> None:
    assert list(Framing) == [
        "normal",
        "casual",
        "persuasive",
        "subagent",
        "reasonese-normal",
        "reasonese-persuasive",
    ]
    assert list(Channel) == ["system prompt", "user message", "README.md"]
    assert list(Author) == [
        "user",
        "Qwen3.8 Flash",
        "Qwen3.8 2.4T",
        "Inkling",
        "Inkling Small",
    ]


def test_axis_manifest_uses_enum_values_directly() -> None:
    assert axis_manifest() == {
        "instruction": "configured base prompts",
        "framing": list(Framing),
        "channel": list(Channel),
        "author": list(Author),
    }


def test_instruction_is_a_trimmed_non_empty_phantom_string() -> None:
    instruction = Instruction.parse("Write a program.")
    assert instruction == "Write a program."
    assert isinstance(instruction, Instruction)

    for invalid in ("", " leading", "trailing "):
        with pytest.raises(TypeError):
            Instruction.parse(invalid)
