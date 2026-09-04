from __future__ import annotations

import json
from pathlib import Path

from reasonese.axes import Author, Channel, Framing, Instruction
from reasonese.io import write_prompt_specs
from reasonese.planning import PromptSpec


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
