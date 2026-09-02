"""Valid combinations of entry datapoints and an assistant model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from beartype import beartype
from phantom import Phantom

from reasonese.axes import Assistant, Author, Channel, Framing, Instruction
from reasonese.planning import PromptSpec


def is_matchup_inputs(value: tuple[PromptSpec, ...]) -> bool:
    """Return whether inputs form one valid pairwise matchup."""
    return (
        len(value) == 2
        and all(isinstance(item, PromptSpec) for item in value)
        and any(item.channel is Channel.USER for item in value)
    )


class MatchupInputs(
    tuple[PromptSpec, ...],
    Phantom,
    predicate=is_matchup_inputs,
):
    """Exactly two inputs, including at least one explicit user message."""


@beartype
@dataclass(frozen=True, slots=True)
class Matchup:
    """Ordered entry datapoints evaluated by one assistant model."""

    inputs: MatchupInputs
    assistant: Assistant


@beartype
def make_matchup(inputs: tuple[PromptSpec, ...], assistant: Assistant) -> Matchup:
    """Validate matchup-wide invariants and construct the refined type."""
    if len(inputs) != 2:
        raise ValueError("a matchup requires exactly two inputs")
    if not any(item.channel is Channel.USER for item in inputs):
        raise ValueError("a matchup requires at least one user message input")
    return Matchup(MatchupInputs.parse(inputs), assistant)


@beartype
def prompt_spec_to_dict(spec: PromptSpec) -> dict[str, str]:
    """Serialize an entry datapoint."""
    return {
        "instruction": str(spec.instruction),
        "framing": str(spec.framing),
        "channel": str(spec.channel),
        "author": str(spec.author),
    }


def prompt_spec_from_dict(raw: object) -> PromptSpec:
    """Parse one entry datapoint from YAML-compatible data."""
    if not isinstance(raw, dict):
        raise ValueError("each matchup input must be a mapping")
    data = cast(dict[str, Any], raw)
    expected = {"instruction", "framing", "channel", "author"}
    if set(data) != expected:
        raise ValueError(f"matchup input fields must be {sorted(expected)}")
    return PromptSpec(
        Instruction.parse(data["instruction"]),
        Framing(data["framing"]),
        Channel(data["channel"]),
        Author(data["author"]),
    )


@beartype
def matchup_to_dict(matchup: Matchup) -> dict[str, object]:
    """Serialize a matchup."""
    return {
        "assistant": str(matchup.assistant),
        "inputs": [prompt_spec_to_dict(item) for item in matchup.inputs],
    }


def matchup_from_dict(raw: object) -> Matchup:
    """Parse a matchup from YAML-compatible data."""
    if not isinstance(raw, dict):
        raise ValueError("matchup must be a mapping")
    data = cast(dict[str, Any], raw)
    if set(data) != {"assistant", "inputs"}:
        raise ValueError("matchup fields must be ['assistant', 'inputs']")
    raw_inputs = data["inputs"]
    if not isinstance(raw_inputs, list):
        raise ValueError("matchup inputs must be a list")
    inputs = tuple(prompt_spec_from_dict(item) for item in raw_inputs)
    return make_matchup(inputs, Assistant(data["assistant"]))
