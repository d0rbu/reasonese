"""Analysis-ready observation rows emitted by data collection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from beartype import beartype
from phantom import Phantom

from reasonese.axes import Assistant
from reasonese.conversation import ConversationTrace
from reasonese.judging import Judgment, TraceFingerprint, trace_fingerprint
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
    if trace.setup.matchup != trial.matchup:
        raise ValueError("trace matchup must equal the trial matchup")
    if judgment.matchup != trial.matchup:
        raise ValueError("judgment matchup must equal the trial matchup")
    fingerprint = trace_fingerprint(trace)
    if judgment.trace_fingerprint != fingerprint:
        raise ValueError("judgment fingerprint must equal the concrete trial trace")
    return tuple(
        Observation(
            trial.trial_id,
            cell_id(Cell(spec, trial.matchup.assistant)),
            spec,
            trial.matchup.assistant,
            trial.permutation,
            trial.rollout,
            PositiveInteger.parse(index),
            verdict.completed,
            fingerprint,
            _response_id(trace.response),
            _response_id(verdict.response),
        )
        for index, (spec, verdict) in enumerate(
            zip(trial.matchup.inputs, judgment.verdicts, strict=True), start=1
        )
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


@beartype
def write_observations(path: Path, observations: tuple[Observation, ...]) -> None:
    """Write flat observation rows as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for observation in observations:
            handle.write(json.dumps(observation_to_dict(observation), ensure_ascii=False) + "\n")
