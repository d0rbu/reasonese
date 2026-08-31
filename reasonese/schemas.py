"""Validated records exchanged by the experiment pipeline."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Self

SCHEMA_VERSION = 1
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")
_RESPONSE_CODE = re.compile(r"^[A-Z][A-Z0-9]*$")


def _require_keys(data: dict[str, Any], expected: set[str], record: str) -> None:
    actual = set(data)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"invalid {record} keys: missing={missing}, extra={extra}")


def _require_schema_version(value: object) -> None:
    if value != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {value!r}")


def validate_identifier(value: str, field_name: str) -> None:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{field_name} must match {_IDENTIFIER.pattern}: {value!r}")


def validate_response_code(value: str) -> None:
    if not _RESPONSE_CODE.fullmatch(value):
        raise ValueError(f"response code must match {_RESPONSE_CODE.pattern}: {value!r}")


@dataclass(frozen=True)
class Condition:
    """One representation or social-framing treatment."""

    id: str
    family: str
    template: str
    description: str

    def __post_init__(self) -> None:
        validate_identifier(self.id, "condition id")
        validate_identifier(self.family, "condition family")
        if self.template.count("{target}") != 1:
            raise ValueError("condition template must contain {target} exactly once")
        if not self.description.strip():
            raise ValueError("condition description must not be empty")

    def render(self, target: str) -> str:
        validate_response_code(target)
        return self.template.format(target=target)


@dataclass(frozen=True)
class Trial:
    """A fully rendered, counterbalanced pairwise conflict."""

    REQUIRED_KEYS: ClassVar[set[str]] = {
        "schema_version",
        "design_id",
        "trial_id",
        "experiment",
        "pair_id",
        "code_pair_id",
        "repetition",
        "first_condition",
        "second_condition",
        "first_target",
        "second_target",
        "system_prompt",
        "user_prompt",
    }

    schema_version: int
    design_id: str
    trial_id: str
    experiment: str
    pair_id: str
    code_pair_id: str
    repetition: int
    first_condition: str
    second_condition: str
    first_target: str
    second_target: str
    system_prompt: str
    user_prompt: str

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        for value, name in (
            (self.experiment, "experiment"),
            (self.first_condition, "first_condition"),
            (self.second_condition, "second_condition"),
        ):
            validate_identifier(value, name)
        if self.first_condition == self.second_condition:
            raise ValueError("trial conditions must be distinct")
        validate_response_code(self.first_target)
        validate_response_code(self.second_target)
        if self.first_target == self.second_target:
            raise ValueError("trial response codes must be distinct")
        if self.repetition < 0:
            raise ValueError("repetition must be non-negative")
        if not self.design_id or not self.trial_id or not self.pair_id or not self.code_pair_id:
            raise ValueError("trial identifiers must not be empty")
        if not self.system_prompt.strip() or not self.user_prompt.strip():
            raise ValueError("trial prompts must not be empty")

    @property
    def condition_to_target(self) -> dict[str, str]:
        return {
            self.first_condition: self.first_target,
            self.second_condition: self.second_target,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        _require_keys(data, cls.REQUIRED_KEYS, "trial")
        return cls(**data)


@dataclass(frozen=True)
class ResponseRecord:
    """One model response to one trial."""

    REQUIRED_KEYS: ClassVar[set[str]] = {
        "schema_version",
        "trial_id",
        "model_id",
        "source",
        "response_text",
    }

    schema_version: int
    trial_id: str
    model_id: str
    source: str
    response_text: str

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        validate_identifier(self.source, "source")
        if not self.trial_id:
            raise ValueError("trial_id must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        _require_keys(data, cls.REQUIRED_KEYS, "response")
        return cls(**data)


@dataclass(frozen=True)
class ScoredOutcome:
    """A decisive pairwise outcome or a retained invalid response."""

    REQUIRED_KEYS: ClassVar[set[str]] = {
        "schema_version",
        "trial_id",
        "model_id",
        "source",
        "first_condition",
        "second_condition",
        "status",
        "winner",
        "loser",
        "matched_target",
    }

    schema_version: int
    trial_id: str
    model_id: str
    source: str
    first_condition: str
    second_condition: str
    status: str
    winner: str | None
    loser: str | None
    matched_target: str | None

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        validate_identifier(self.source, "source")
        validate_identifier(self.first_condition, "first_condition")
        validate_identifier(self.second_condition, "second_condition")
        if self.first_condition == self.second_condition:
            raise ValueError("outcome conditions must be distinct")
        if self.status not in {"decisive", "invalid"}:
            raise ValueError(f"unsupported outcome status: {self.status!r}")
        values = (self.winner, self.loser, self.matched_target)
        if self.status == "decisive":
            if any(value is None for value in values):
                raise ValueError("decisive outcomes require winner, loser, and matched_target")
            assert self.winner is not None and self.loser is not None
            validate_identifier(self.winner, "winner")
            validate_identifier(self.loser, "loser")
            if self.winner == self.loser:
                raise ValueError("winner and loser must be distinct")
            assert self.matched_target is not None
            validate_response_code(self.matched_target)
        elif any(value is not None for value in values):
            raise ValueError("invalid outcomes must not name a winner, loser, or target")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        _require_keys(data, cls.REQUIRED_KEYS, "outcome")
        return cls(**data)
