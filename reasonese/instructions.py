"""Instruction pairs: two base prompts that share a scenario but cannot both be completed."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import yaml
from beartype import beartype
from phantom import Phantom

from reasonese.axes import Framing, Instruction, is_non_empty_trimmed


def is_pair_id(value: str) -> bool:
    """Return whether text is a lowercase, hyphen-separated identifier."""
    return re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value) is not None


class PairId(str, Phantom[str], predicate=is_pair_id, bound=str):
    """A lowercase, hyphen-separated identifier for one instruction pair."""


class Rationale(str, Phantom[str], predicate=is_non_empty_trimmed, bound=str):
    """A non-empty explanation of why one response cannot satisfy both instructions."""


class Skill(StrEnum):
    """The agentic capability an instruction is meant to exercise."""

    PYTHON = "python"
    BASH = "bash"
    WEB_SEARCH = "web search"
    PYTHON_AND_WEB_SEARCH = "python and web search"


class ConflictType(StrEnum):
    """The mechanism that makes two instructions mutually exclusive."""

    OUTPUT_FORMAT = "output format"
    PROCESS = "process"
    DELIVERABLE = "deliverable"
    TOOL_CHOICE = "tool choice"
    SOURCE_POLICY = "source policy"
    LANGUAGE = "language"
    LENGTH = "length"
    SCOPE = "scope"
    CONTENT = "content"
    REGISTER = "register"


@beartype
@dataclass(frozen=True, slots=True)
class InstructionPair:
    """Two self-contained base instructions that one response cannot both complete."""

    pair_id: PairId
    skill: Skill
    conflict: ConflictType
    first: Instruction
    second: Instruction
    rationale: Rationale

    def __post_init__(self) -> None:
        if self.first == self.second:
            raise ValueError("an instruction pair must contain two different instructions")

    @property
    def instructions(self) -> tuple[Instruction, Instruction]:
        """Return both instructions in pair order."""
        return (self.first, self.second)


_PAIR_FIELDS = frozenset({"id", "skill", "conflict", "first", "second", "rationale"})


@beartype
def pair_to_dict(pair: InstructionPair) -> dict[str, str]:
    """Serialize one instruction pair to YAML-compatible data."""
    return {
        "id": str(pair.pair_id),
        "skill": str(pair.skill),
        "conflict": str(pair.conflict),
        "first": str(pair.first),
        "second": str(pair.second),
        "rationale": str(pair.rationale),
    }


def pair_from_dict(raw: object) -> InstructionPair:
    """Parse one instruction pair from YAML-compatible data."""
    if not isinstance(raw, dict):
        raise ValueError("each instruction pair must be a mapping")
    data = cast(dict[str, Any], raw)
    if set(data) != _PAIR_FIELDS:
        raise ValueError(f"instruction pair fields must be {sorted(_PAIR_FIELDS)}")
    for field in ("id", "first", "second", "rationale"):
        if not isinstance(data[field], str):
            raise ValueError(f"instruction pair {field} must be text")
    return InstructionPair(
        PairId.parse(data["id"]),
        Skill(data["skill"]),
        ConflictType(data["conflict"]),
        Instruction.parse(data["first"]),
        Instruction.parse(data["second"]),
        Rationale.parse(data["rationale"]),
    )


@beartype
def load_instruction_pairs(path: Path) -> tuple[InstructionPair, ...]:
    """Load a non-empty list of instruction pairs with unique identifiers."""
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or set(raw) != {"pairs"} or not isinstance(raw["pairs"], list):
        raise ValueError(f"{path} must contain one 'pairs' list")
    pairs = tuple(pair_from_dict(item) for item in raw["pairs"])
    if not pairs:
        raise ValueError("at least one instruction pair is required")
    ids = [pair.pair_id for pair in pairs]
    if len(ids) != len(set(ids)):
        raise ValueError("instruction pair ids must be unique")
    return pairs


_TOKEN = re.compile(r"[a-z0-9]+")


@beartype
def lexical_similarity(first: str, second: str) -> float:
    """Return the Jaccard similarity of two texts' lowercase word sets."""
    first_tokens = set(_TOKEN.findall(first.lower()))
    second_tokens = set(_TOKEN.findall(second.lower()))
    if not first_tokens and not second_tokens:
        return 1.0
    return len(first_tokens & second_tokens) / len(first_tokens | second_tokens)


@beartype
@dataclass(frozen=True, slots=True)
class SimilarityRow:
    """The most lexically similar instruction found in a different pair."""

    pair_id: PairId
    slot: str
    nearest_pair_id: PairId
    similarity: float


@beartype
def cross_pair_similarities(pairs: tuple[InstructionPair, ...]) -> tuple[SimilarityRow, ...]:
    """For every instruction, find its nearest instruction in any other pair."""
    rows: list[SimilarityRow] = []
    for pair in pairs:
        for slot, instruction in (("first", pair.first), ("second", pair.second)):
            best: tuple[PairId, float] | None = None
            for other in pairs:
                if other.pair_id == pair.pair_id:
                    continue
                for candidate in other.instructions:
                    score = lexical_similarity(instruction, candidate)
                    if best is None or score > best[1]:
                        best = (other.pair_id, score)
            if best is not None:
                rows.append(SimilarityRow(pair.pair_id, slot, best[0], best[1]))
    return tuple(sorted(rows, key=lambda row: (-row.similarity, str(row.pair_id), row.slot)))


@beartype
def coverage(pairs: tuple[InstructionPair, ...]) -> dict[tuple[str, str], int]:
    """Count pairs for every skill and conflict-type combination present."""
    return dict(Counter((str(pair.skill), str(pair.conflict)) for pair in pairs))


_SOURCE_FILE = "instruction.txt"


@beartype
def scaffold_manual_variants(
    root: Path, pairs: tuple[InstructionPair, ...]
) -> tuple[Path, ...]:
    """Create placeholder user-authored variant directories for new instructions."""
    root.mkdir(parents=True, exist_ok=True)
    existing: set[str] = set()
    for directory in root.iterdir():
        source = directory / _SOURCE_FILE
        if directory.is_dir() and source.is_file():
            existing.add(source.read_text(encoding="utf-8").strip())
    created: list[Path] = []
    for pair in pairs:
        for slot, instruction in (("a", pair.first), ("b", pair.second)):
            if str(instruction) in existing:
                continue
            directory = root / f"{pair.pair_id}-{slot}"
            if directory.exists():
                raise ValueError(f"manual variant directory holds different text: {directory}")
            directory.mkdir()
            (directory / _SOURCE_FILE).write_text(f"{instruction}\n", encoding="utf-8")
            for framing in Framing:
                (directory / f"{framing}.txt").write_text(
                    f"TODO: Write the {framing} version of {_SOURCE_FILE}.\n",
                    encoding="utf-8",
                )
            existing.add(str(instruction))
            created.append(directory)
    return tuple(created)
