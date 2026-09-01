"""Filesystem-backed messages written manually by the user author."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from beartype import beartype

from reasonese.axes import Author, Framing, Instruction
from reasonese.conversation import ConversationSetup, GeneratedText
from reasonese.planning import PromptSpec

_PLACEHOLDER_PREFIX = "TODO:"
_SOURCE_FILE = "instruction.txt"


@beartype
@dataclass(frozen=True, slots=True)
class ManualMessageLibrary:
    """Manual framing variants organized as one directory per base instruction."""

    root: Path

    def _instruction_directories(self) -> dict[Instruction, Path]:
        if not self.root.is_dir():
            raise ValueError(f"manual message directory does not exist: {self.root}")
        required = {_SOURCE_FILE, *(f"{framing}.txt" for framing in Framing)}
        by_instruction: dict[Instruction, Path] = {}
        for directory in sorted(
            (entry for entry in self.root.iterdir() if entry.is_dir()),
            key=lambda entry: entry.name,
        ):
            missing = required - {entry.name for entry in directory.iterdir() if entry.is_file()}
            if missing:
                raise ValueError(
                    f"manual message directory {directory} is missing: {sorted(missing)}"
                )
            instruction = Instruction.parse(
                (directory / _SOURCE_FILE).read_text(encoding="utf-8").strip()
            )
            if instruction in by_instruction:
                raise ValueError(
                    f"manual message instruction appears more than once: {instruction}"
                )
            by_instruction[instruction] = directory
        return by_instruction

    @beartype
    def message_for(self, spec: PromptSpec) -> GeneratedText:
        """Load the selected manual framing for one user-authored datapoint."""
        if spec.author is not Author.USER:
            raise ValueError("manual messages are only defined for the user author")
        directory = self._instruction_directories().get(spec.instruction)
        if directory is None:
            raise ValueError(f"no manual message directory matches instruction: {spec.instruction}")
        path = directory / f"{spec.framing}.txt"
        content = path.read_text(encoding="utf-8").strip()
        if content.startswith(_PLACEHOLDER_PREFIX):
            raise ValueError(f"manual message is still a placeholder: {path}")
        return GeneratedText.parse(content)

    @beartype
    def matches(self, setup: ConversationSetup) -> bool:
        """Return whether cached user-authored text still matches the source files."""
        return all(
            spec.author is not Author.USER
            or self.message_for(spec) == setup.content_for_input(index)
            for index, spec in enumerate(setup.matchup.inputs)
        )
