"""Analysis-ready observation rows emitted by data collection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from beartype import beartype
from phantom import Phantom

from reasonese.axes import Assistant, Author, Channel, Framing, Instruction
from reasonese.conversation import ConversationTrace
from reasonese.judging import FingerprintedTrace, Judgment, TraceFingerprint
from reasonese.matchup import prompt_spec_to_dict
from reasonese.openrouter import JsonObject
from reasonese.planning import PromptSpec
from reasonese.study import Cell, PositiveInteger, Trial, TrialId


def _is_cell_id(value: str) -> bool:
    return len(value) == 16 and all(character in "0123456789abcdef" for character in value)


class CellId(str, Phantom[str], predicate=_is_cell_id, bound=str):
    """Stable identifier for a four-axis datapoint and assistant."""


@beartype
@dataclass(frozen=True, slots=True)
class Observation:
    """One cell's binary completion result in one ordered trial."""

    trial_id: TrialId
    cell_id: CellId
    spec: PromptSpec
    assistant: Assistant
    permutation: PositiveInteger
    rollout: PositiveInteger
    position: PositiveInteger
    completed: bool
    trace_fingerprint: TraceFingerprint
    assistant_response_id: str | None
    judge_response_id: str | None


def _observation_from_validated(
    trial_id: TrialId,
    cell: CellId,
    spec: PromptSpec,
    assistant: Assistant,
    permutation: PositiveInteger,
    rollout: PositiveInteger,
    position: PositiveInteger,
    completed: bool,
    fingerprint: TraceFingerprint,
    assistant_response_id: str | None,
    judge_response_id: str | None,
) -> Observation:
    """Construct a row after its enclosing batch and relationships were validated."""
    observation = object.__new__(Observation)
    object.__setattr__(observation, "trial_id", trial_id)
    object.__setattr__(observation, "cell_id", cell)
    object.__setattr__(observation, "spec", spec)
    object.__setattr__(observation, "assistant", assistant)
    object.__setattr__(observation, "permutation", permutation)
    object.__setattr__(observation, "rollout", rollout)
    object.__setattr__(observation, "position", position)
    object.__setattr__(observation, "completed", completed)
    object.__setattr__(observation, "trace_fingerprint", fingerprint)
    object.__setattr__(observation, "assistant_response_id", assistant_response_id)
    object.__setattr__(observation, "judge_response_id", judge_response_id)
    return observation


_POSITIONS = (PositiveInteger.parse(1), PositiveInteger.parse(2))


@beartype
def cell_id(cell: Cell) -> CellId:
    """Hash a cell's exact coordinates into a compact stable identifier."""
    canonical = json.dumps(
        {"input": prompt_spec_to_dict(cell.spec), "assistant": str(cell.assistant)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return CellId.parse(hashlib.sha256(canonical.encode()).hexdigest()[:16])


def _response_id(response: JsonObject) -> str | None:
    response_id = response.get("id")
    return response_id if isinstance(response_id, str) else None


@beartype
def observations_from_trial(
    trial: Trial,
    trace: ConversationTrace,
    judgment: Judgment,
) -> tuple[Observation, ...]:
    """Join one trial trace and judgment into ordered cell observations."""
    fingerprinted = FingerprintedTrace(trace)
    cell_ids = {
        (spec, trial.matchup.assistant): cell_id(Cell(spec, trial.matchup.assistant))
        for spec in trial.matchup.inputs
    }
    return _observations_from_trial(trial, fingerprinted, judgment, cell_ids)


def _observations_from_trial(
    trial: Trial,
    fingerprinted: FingerprintedTrace,
    judgment: Judgment,
    cell_ids: dict[tuple[PromptSpec, Assistant], CellId],
) -> tuple[Observation, ...]:
    trace = fingerprinted.trace
    if trace.setup.matchup != trial.matchup:
        raise ValueError("trace matchup must equal the trial matchup")
    if judgment.matchup != trial.matchup:
        raise ValueError("judgment matchup must equal the trial matchup")
    if judgment.trace_fingerprint != fingerprinted.fingerprint:
        raise ValueError("judgment fingerprint must equal the concrete trial trace")
    return tuple(
        _observation_from_validated(
            trial.trial_id,
            cell_ids[(spec, trial.matchup.assistant)],
            spec,
            trial.matchup.assistant,
            trial.permutation,
            trial.rollout,
            position,
            verdict.completed,
            fingerprinted.fingerprint,
            _response_id(trace.response),
            _response_id(verdict.response),
        )
        for position, spec, verdict in zip(
            _POSITIONS,
            trial.matchup.inputs,
            judgment.verdicts,
            strict=True,
        )
    )


@beartype
def observations_from_trials(
    trials: tuple[Trial, ...],
    traces: tuple[FingerprintedTrace, ...],
    judgments: tuple[Judgment, ...],
) -> tuple[Observation, ...]:
    """Join many aligned trial records while reusing their derived cell identifiers."""
    if len(trials) != len(traces) or len(trials) != len(judgments):
        raise ValueError("trials, traces, and judgments must have equal lengths")
    cell_keys = tuple(
        dict.fromkeys(
            (spec, trial.matchup.assistant)
            for trial in trials
            for spec in trial.matchup.inputs
        )
    )
    cell_ids = {
        key: cell_id(Cell(*key))
        for key in cell_keys
    }
    return tuple(
        observation
        for trial, trace, judgment in zip(trials, traces, judgments, strict=True)
        for observation in _observations_from_trial(trial, trace, judgment, cell_ids)
    )


@beartype
def observation_to_dict(observation: Observation) -> dict[str, object]:
    """Serialize one flat analysis-ready observation."""
    return {
        "trial_id": observation.trial_id,
        "cell_id": str(observation.cell_id),
        **prompt_spec_to_dict(observation.spec),
        "assistant": str(observation.assistant),
        "permutation": int(observation.permutation),
        "rollout": int(observation.rollout),
        "position": int(observation.position),
        "completed": observation.completed,
        "trace_fingerprint": str(observation.trace_fingerprint),
        "assistant_response_id": observation.assistant_response_id,
        "judge_response_id": observation.judge_response_id,
    }


def observation_from_dict(raw: object) -> Observation:
    """Parse one flat observation and verify its derived cell identifier."""
    if not isinstance(raw, dict):
        raise ValueError("observation must be a mapping")
    data = cast(dict[str, Any], raw)
    expected = {
        "trial_id",
        "cell_id",
        "instruction",
        "framing",
        "channel",
        "author",
        "assistant",
        "permutation",
        "rollout",
        "position",
        "completed",
        "trace_fingerprint",
        "assistant_response_id",
        "judge_response_id",
    }
    if set(data) != expected:
        raise ValueError("observation has invalid fields")
    completed = data["completed"]
    if not isinstance(completed, bool):
        raise ValueError("observation completed field must be a boolean")
    integer_fields = ("permutation", "rollout", "position")
    if any(
        not isinstance(data[field], int) or isinstance(data[field], bool)
        for field in integer_fields
    ):
        raise ValueError("observation permutation, rollout, and position must be integers")
    for field in ("assistant_response_id", "judge_response_id"):
        if data[field] is not None and not isinstance(data[field], str):
            raise ValueError("observation response IDs must be strings or null")
    spec = PromptSpec(
        Instruction.parse(data["instruction"]),
        Framing(data["framing"]),
        Channel(data["channel"]),
        Author(data["author"]),
    )
    assistant = Assistant(data["assistant"])
    parsed_cell_id = CellId.parse(data["cell_id"])
    if parsed_cell_id != cell_id(Cell(spec, assistant)):
        raise ValueError("observation cell_id does not match its coordinates")
    return Observation(
        TrialId.parse(data["trial_id"]),
        parsed_cell_id,
        spec,
        assistant,
        PositiveInteger.parse(data["permutation"]),
        PositiveInteger.parse(data["rollout"]),
        PositiveInteger.parse(data["position"]),
        completed,
        TraceFingerprint.parse(data["trace_fingerprint"]),
        data["assistant_response_id"],
        data["judge_response_id"],
    )


@beartype
def load_observations(path: Path) -> tuple[Observation, ...]:
    """Load analysis-ready observations from JSONL."""
    observations: list[Observation] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                observations.append(observation_from_dict(raw))
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid observation at {path}:{line_number}: {error}") from error
    if not observations:
        raise ValueError(f"{path} contains no observations")
    return tuple(observations)


@beartype
def write_observations(path: Path, observations: tuple[Observation, ...]) -> None:
    """Write flat observation rows as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for observation in observations:
            handle.write(json.dumps(observation_to_dict(observation), ensure_ascii=False) + "\n")
