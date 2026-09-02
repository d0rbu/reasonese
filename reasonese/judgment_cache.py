"""Readable YAML cache for independent per-instruction judgments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml
from beartype import beartype

from reasonese.conversation import ConversationTrace
from reasonese.judging import (
    InstructionVerdict,
    InstructionVerdicts,
    Judgment,
    TraceFingerprint,
    trace_fingerprint,
)
from reasonese.matchup import (
    Matchup,
    matchup_from_dict,
    matchup_to_dict,
    prompt_spec_from_dict,
    prompt_spec_to_dict,
)
from reasonese.openrouter import JsonObject
from reasonese.planning import PromptSpec


def _response(raw: object) -> JsonObject:
    if not isinstance(raw, dict):
        raise ValueError("cached judge response must be a mapping")
    return cast(JsonObject, raw)


def judgment_from_dict(
    raw: object,
    expected_matchup: Matchup | None = None,
) -> Judgment:
    """Parse one judgment through its fully checked public constructors."""
    matchup, fingerprint, fields = _judgment_fields(raw, expected_matchup)
    verdicts = InstructionVerdicts.parse(
        tuple(InstructionVerdict(spec, completed, response) for spec, completed, response in fields)
    )
    return Judgment(matchup, fingerprint, verdicts)


def _judgment_fields(
    raw: object,
    expected_matchup: Matchup | None,
) -> tuple[Matchup, TraceFingerprint, tuple[tuple[PromptSpec, bool, JsonObject], ...]]:
    if not isinstance(raw, dict):
        raise ValueError("cached judgment must be a mapping")
    data = cast(dict[str, Any], raw)
    if set(data) != {"matchup", "trace_fingerprint", "verdicts"}:
        raise ValueError("cached judgment has invalid fields")
    raw_verdicts = data["verdicts"]
    if not isinstance(raw_verdicts, list):
        raise ValueError("cached judgment verdicts must be a list")
    if expected_matchup is None:
        matchup = matchup_from_dict(data["matchup"])
    else:
        if data["matchup"] != matchup_to_dict(expected_matchup):
            raise ValueError("cached judgment matchup does not match expected trial")
        matchup = expected_matchup
    fields: list[tuple[PromptSpec, bool, JsonObject]] = []
    for index, raw_verdict in enumerate(raw_verdicts):
        if not isinstance(raw_verdict, dict):
            raise ValueError("cached instruction verdict must be a mapping")
        verdict = cast(dict[str, Any], raw_verdict)
        if set(verdict) != {"input", "completed", "response"}:
            raise ValueError("cached instruction verdict has invalid fields")
        completed = verdict["completed"]
        if not isinstance(completed, bool):
            raise ValueError("cached instruction verdict must contain a boolean")
        if expected_matchup is None:
            spec = prompt_spec_from_dict(verdict["input"])
        else:
            if index >= len(expected_matchup.inputs):
                raise ValueError("cached judgment has too many verdicts")
            spec = expected_matchup.inputs[index]
            if verdict["input"] != prompt_spec_to_dict(spec):
                raise ValueError("cached judgment input does not match expected trial")
        fields.append((spec, completed, _response(verdict["response"])))
    return (
        matchup,
        TraceFingerprint.parse(data["trace_fingerprint"]),
        tuple(fields),
    )


def _judgment_from_validated(
    matchup: Matchup,
    fingerprint: TraceFingerprint,
    verdicts: InstructionVerdicts,
) -> Judgment:
    judgment = object.__new__(Judgment)
    object.__setattr__(judgment, "matchup", matchup)
    object.__setattr__(judgment, "trace_fingerprint", fingerprint)
    object.__setattr__(judgment, "verdicts", verdicts)
    return judgment


def _instruction_verdict_from_validated(
    spec: PromptSpec,
    completed: bool,
    response: JsonObject,
) -> InstructionVerdict:
    verdict = object.__new__(InstructionVerdict)
    object.__setattr__(verdict, "spec", spec)
    object.__setattr__(verdict, "completed", completed)
    object.__setattr__(verdict, "response", response)
    return verdict


@beartype
def judgments_from_dicts(
    raws: tuple[object, ...],
    expected_matchups: tuple[Matchup, ...],
) -> tuple[Judgment, ...]:
    """Decode aligned cache rows after explicit payload and relationship validation."""
    if len(raws) != len(expected_matchups):
        raise ValueError("cached judgments and expected matchups must have equal lengths")
    judgments: list[Judgment] = []
    for raw, expected_matchup in zip(raws, expected_matchups, strict=True):
        matchup, fingerprint, fields = _judgment_fields(raw, expected_matchup)
        verdicts = InstructionVerdicts.parse(
            tuple(
                _instruction_verdict_from_validated(spec, completed, response)
                for spec, completed, response in fields
            )
        )
        judgments.append(_judgment_from_validated(matchup, fingerprint, verdicts))
    return tuple(judgments)


@beartype
def judgment_to_dict(judgment: Judgment) -> dict[str, object]:
    return {
        "matchup": matchup_to_dict(judgment.matchup),
        "trace_fingerprint": str(judgment.trace_fingerprint),
        "verdicts": [
            {
                "input": prompt_spec_to_dict(verdict.spec),
                "completed": verdict.completed,
                "response": verdict.response,
            }
            for verdict in judgment.verdicts
        ],
    }


@beartype
@dataclass(frozen=True, slots=True)
class YamlJudgmentCache:
    """Judgments keyed by the complete matchup and concrete trace fingerprint."""

    path: Path

    def load(self) -> tuple[Judgment, ...]:
        if not self.path.exists():
            return ()
        with self.path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if raw is None:
            return ()
        if not isinstance(raw, dict) or set(raw) != {"judgments"}:
            raise ValueError(f"{self.path} must contain one 'judgments' list")
        raw_judgments = raw["judgments"]
        if not isinstance(raw_judgments, list):
            raise ValueError(f"{self.path} must contain one 'judgments' list")
        return tuple(judgment_from_dict(item) for item in raw_judgments)

    def get(self, trace: ConversationTrace) -> Judgment | None:
        fingerprint = trace_fingerprint(trace)
        return next(
            (
                judgment
                for judgment in self.load()
                if judgment.matchup == trace.setup.matchup
                and judgment.trace_fingerprint == fingerprint
            ),
            None,
        )

    def put(self, judgment: Judgment) -> None:
        self.put_many((judgment,))

    def put_many(self, judgments: tuple[Judgment, ...]) -> None:
        """Insert or replace multiple judgments with one cache write."""
        by_key = {(cached.matchup, cached.trace_fingerprint): cached for cached in self.load()}
        by_key.update(
            {(judgment.matchup, judgment.trace_fingerprint): judgment for judgment in judgments}
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                {"judgments": [judgment_to_dict(item) for item in by_key.values()]},
                handle,
                sort_keys=False,
                allow_unicode=True,
                width=100,
            )
