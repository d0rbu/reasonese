from __future__ import annotations

import json
from pathlib import Path

import pytest

from reasonese.axes import Author, Channel, Framing, Instruction
from reasonese.config import load_instructions
from reasonese.io import write_prompt_specs
from reasonese.planning import PromptSpec


def test_example_instructions_load_as_phantom_strings() -> None:
    instructions = load_instructions(Path("configs/example_instructions.toml"))
    assert len(instructions) == 2
    assert all(isinstance(instruction, Instruction) for instruction in instructions)


@pytest.mark.parametrize(
    ("contents", "error"),
    [
        ("other = []\n", "TOML array"),
        ("instructions = 'bad'\n", "TOML array"),
        ("instructions = []\n", "at least one"),
        ("instructions = ['']\n", "Could not parse"),
    ],
)
def test_invalid_instruction_configs_are_rejected(
    tmp_path: Path, contents: str, error: str
) -> None:
    path = tmp_path / "instructions.toml"
    path.write_text(contents)
    with pytest.raises((TypeError, ValueError), match=error):
        load_instructions(path)


def test_writer_emits_only_the_four_axes(tmp_path: Path) -> None:
    spec = PromptSpec(
        Instruction.parse("Write a program."),
        Framing.NORMAL,
        Channel.SYSTEM,
        Author.USER,
    )
    output = tmp_path / "nested" / "specs.jsonl"
    write_prompt_specs(output, (spec,))

    assert json.loads(output.read_text()) == {
        "instruction": "Write a program.",
        "framing": "normal",
        "channel": "system prompt",
        "author": "user",
    }
