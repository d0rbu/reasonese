from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import cast

import pytest

from reasonese.axes import Author, Channel, Framing, Instruction
from reasonese.config import InstructionSet, load_instruction_set
from reasonese.io import read_prompt_specs, write_prompt_specs
from reasonese.planning import PromptSpec, build_prompt_specs


def test_example_instruction_set_is_valid() -> None:
    instruction_set = load_instruction_set(Path("configs/example_instructions.toml"))
    assert [instruction.id for instruction in instruction_set.instructions] == [
        "write_word_counter",
        "find_pathlib_documentation",
    ]


def test_instruction_set_rejects_empty_and_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="at least one"):
        InstructionSet(instructions=())
    instruction = Instruction(id="duplicate", text="Do something.")
    with pytest.raises(ValueError, match="unique"):
        InstructionSet(instructions=(instruction, instruction))


@pytest.mark.parametrize(
    ("contents", "match"),
    [
        ("schema_version = 2\ninstructions = []\n", "unsupported schema"),
        ("schema_version = 1\n", "missing=\\['instructions'\\]"),
        ("schema_version = 1\ninstructions = 'bad'\n", "array of tables"),
        (
            "schema_version = 1\n[[instructions]]\nid = 'task'\ntext = 'Do it.'\nextra = 1\n",
            "extra=\\['extra'\\]",
        ),
        (
            "schema_version = 1\ninstructions = [1]\n",
            "instruction 0 must be a table",
        ),
        (
            "schema_version = 1\n[[instructions]]\nid = 'task'\n",
            "missing=\\['text'\\]",
        ),
        (
            "schema_version = 1\n[[instructions]]\nid = 1\ntext = 'Do it.'\n",
            "id and text must be strings",
        ),
        ("schema_version = 1\ninstructions = []\n", "at least one"),
    ],
)
def test_instruction_config_is_strict(tmp_path: Path, contents: str, match: str) -> None:
    config = tmp_path / "instructions.toml"
    config.write_text(contents)
    with pytest.raises(ValueError, match=match):
        load_instruction_set(config)


def test_prompt_specs_round_trip_as_jsonl(tmp_path: Path) -> None:
    specs = build_prompt_specs((Instruction(id="task", text="Do the task."),))
    output = tmp_path / "nested" / "specs.jsonl"
    write_prompt_specs(output, specs)

    assert read_prompt_specs(output) == specs
    assert len(output.read_text().splitlines()) == 90
    assert output.read_text().endswith("\n")


@pytest.mark.parametrize(
    ("contents", "match"),
    [
        ("\n", "blank line"),
        ("{bad json}\n", "invalid JSON"),
        ("[]\n", "must be an object"),
        (json.dumps({"schema_version": 1}) + "\n", "invalid prompt spec keys"),
    ],
)
def test_prompt_spec_reader_rejects_invalid_lines(
    tmp_path: Path, contents: str, match: str
) -> None:
    source = tmp_path / "specs.jsonl"
    source.write_text(contents)
    with pytest.raises(ValueError, match=match):
        read_prompt_specs(source)


def test_prompt_spec_reader_accepts_an_empty_file(tmp_path: Path) -> None:
    source = tmp_path / "empty.jsonl"
    source.touch()
    assert read_prompt_specs(source) == ()


def test_atomic_writer_preserves_target_and_cleans_temp_on_failure(tmp_path: Path) -> None:
    class BrokenSpec:
        def to_dict(self) -> dict[str, object]:
            raise RuntimeError("serialization failed")

    output = tmp_path / "specs.jsonl"
    output.write_text("existing\n")
    broken = cast(Iterable[PromptSpec], [BrokenSpec()])

    with pytest.raises(RuntimeError, match="serialization failed"):
        write_prompt_specs(output, broken)

    assert output.read_text() == "existing\n"
    assert list(tmp_path.glob(".specs.jsonl.*.tmp")) == []


def test_single_prompt_spec_can_be_written(tmp_path: Path) -> None:
    spec = PromptSpec.create(
        Instruction(id="task", text="Do the task."),
        Framing.SUBAGENT,
        Channel.SYSTEM,
        Author.QWEN3_8_FLASH,
    )
    output = tmp_path / "one.jsonl"
    write_prompt_specs(output, [spec])
    assert read_prompt_specs(output) == (spec,)
