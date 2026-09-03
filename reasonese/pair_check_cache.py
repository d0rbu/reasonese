"""Readable YAML cache for instruction-pair audits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml
from beartype import beartype

from reasonese.instructions import InstructionPair, pair_from_dict, pair_to_dict
from reasonese.openrouter import JsonObject
from reasonese.pair_checks import (
    CheckIssue,
    Difficulty,
    InstructionAssessment,
    PairCheck,
)


def _assessment_from_dict(raw: object, slot: str) -> InstructionAssessment:
    if not isinstance(raw, dict):
        raise ValueError(f"cached pair check {slot} assessment must be a mapping")
    data = cast(dict[str, Any], raw)
    if set(data) != {"feasible", "requires_tools", "difficulty"}:
        raise ValueError(f"cached pair check {slot} assessment has invalid fields")
    if not isinstance(data["feasible"], bool) or not isinstance(data["requires_tools"], bool):
        raise ValueError(f"cached pair check {slot} booleans are invalid")
    difficulty = data["difficulty"]
    if not isinstance(difficulty, int) or isinstance(difficulty, bool):
        raise ValueError(f"cached pair check {slot} difficulty must be an integer")
    return InstructionAssessment(
        data["feasible"], data["requires_tools"], Difficulty.parse(difficulty)
    )


def _assessment_to_dict(assessment: InstructionAssessment) -> dict[str, object]:
    return {
        "feasible": assessment.feasible,
        "requires_tools": assessment.requires_tools,
        "difficulty": int(assessment.difficulty),
    }


def _check_from_dict(raw: object) -> PairCheck:
    if not isinstance(raw, dict):
        raise ValueError("cached pair check must be a mapping")
    data = cast(dict[str, Any], raw)
    expected = {"pair", "first", "second", "mutually_exclusive", "issues", "response"}
    if set(data) != expected:
        raise ValueError("cached pair check has invalid fields")
    mutually_exclusive = data["mutually_exclusive"]
    raw_issues = data["issues"]
    response = data["response"]
    if not isinstance(mutually_exclusive, bool):
        raise ValueError("cached pair check mutually_exclusive must be a boolean")
    if not isinstance(raw_issues, list):
        raise ValueError("cached pair check issues must be a list")
    if not isinstance(response, dict):
        raise ValueError("cached pair check response must be a mapping")
    return PairCheck(
        pair_from_dict(data["pair"]),
        _assessment_from_dict(data["first"], "first"),
        _assessment_from_dict(data["second"], "second"),
        mutually_exclusive,
        tuple(CheckIssue.parse(issue) for issue in raw_issues),
        cast(JsonObject, response),
    )


@beartype
def _check_to_dict(check: PairCheck) -> dict[str, object]:
    return {
        "pair": pair_to_dict(check.pair),
        "first": _assessment_to_dict(check.first),
        "second": _assessment_to_dict(check.second),
        "mutually_exclusive": check.mutually_exclusive,
        "issues": [str(issue) for issue in check.issues],
        "response": check.response,
    }


@beartype
@dataclass(frozen=True, slots=True)
class YamlPairCheckCache:
    """Pair audits keyed by pair identifier and exact instruction texts."""

    path: Path

    def load(self) -> tuple[PairCheck, ...]:
        if not self.path.exists():
            return ()
        with self.path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if raw is None:
            return ()
        if not isinstance(raw, dict) or set(raw) != {"pair_checks"}:
            raise ValueError(f"{self.path} must contain one 'pair_checks' list")
        raw_checks = raw["pair_checks"]
        if not isinstance(raw_checks, list):
            raise ValueError(f"{self.path} must contain one 'pair_checks' list")
        return tuple(_check_from_dict(item) for item in raw_checks)

    @beartype
    def get(self, pair: InstructionPair) -> PairCheck | None:
        return next((check for check in self.load() if check.matches(pair)), None)

    @beartype
    def put_many(self, checks: tuple[PairCheck, ...]) -> None:
        by_id = {check.pair.pair_id: check for check in self.load()}
        by_id.update({check.pair.pair_id: check for check in checks})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                {"pair_checks": [_check_to_dict(check) for check in by_id.values()]},
                handle,
                sort_keys=False,
                allow_unicode=True,
                width=100,
            )
