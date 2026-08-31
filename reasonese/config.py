"""Strict TOML configuration loaders."""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from reasonese.schemas import SCHEMA_VERSION, Condition, validate_identifier, validate_response_code
from reasonese.types import Probability, parse_probability


def _require_table(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a TOML table")
    return cast("dict[str, Any]", value)


def _require_exact_keys(data: dict[str, Any], expected: set[str], name: str) -> None:
    if set(data) != expected:
        missing = sorted(expected - set(data))
        extra = sorted(set(data) - expected)
        raise ValueError(f"invalid {name} keys: missing={missing}, extra={extra}")


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_int(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated inputs needed to generate a complete pairwise design."""

    name: str
    seed: int
    repetitions: int
    system_prompt: str
    user_preamble: str
    response_code_pairs: tuple[tuple[str, str], ...]
    conditions: tuple[Condition, ...]

    def __post_init__(self) -> None:
        validate_identifier(self.name, "experiment name")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if type(self.repetitions) is not int or self.repetitions < 1:
            raise ValueError("repetitions must be a positive integer")
        if not self.system_prompt.strip() or not self.user_preamble.strip():
            raise ValueError("prompt text must not be empty")
        if not self.response_code_pairs:
            raise ValueError("at least one response-code pair is required")
        for first, second in self.response_code_pairs:
            validate_response_code(first)
            validate_response_code(second)
            if first == second:
                raise ValueError("response codes in a pair must be distinct")
        if len(self.conditions) < 2:
            raise ValueError("at least two conditions are required")
        condition_ids = [condition.id for condition in self.conditions]
        if len(condition_ids) != len(set(condition_ids)):
            raise ValueError("condition ids must be unique")

    @property
    def design_id(self) -> str:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "seed": self.seed,
            "repetitions": self.repetitions,
            "system_prompt": self.system_prompt,
            "user_preamble": self.user_preamble,
            "response_code_pairs": self.response_code_pairs,
            "conditions": [
                {
                    "id": condition.id,
                    "family": condition.family,
                    "template": condition.template,
                    "description": condition.description,
                }
                for condition in self.conditions
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters for an explicitly synthetic Bradley-Terry response generator."""

    model_id: str
    seed: int
    invalid_rate: Probability
    first_position_bias: float
    strengths: dict[str, float]

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(self.invalid_rate, Probability):
            raise ValueError("invalid_rate must be a validated Probability")
        if not math.isfinite(self.first_position_bias):
            raise ValueError("first_position_bias must be finite")
        if not self.strengths:
            raise ValueError("simulation strengths must not be empty")
        for condition_id, strength in self.strengths.items():
            validate_identifier(condition_id, "strength condition id")
            if not math.isfinite(strength):
                raise ValueError(f"strength for {condition_id!r} must be finite")


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            loaded = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"could not load TOML config {path}: {error}") from error
    return loaded


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load an experiment design with no ignored or implicit fields."""
    data = _load_toml(path)
    _require_exact_keys(data, {"schema_version", "experiment", "conditions"}, "top-level")
    if data["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {data['schema_version']!r}")

    experiment = _require_table(data["experiment"], "experiment")
    _require_exact_keys(
        experiment,
        {
            "name",
            "seed",
            "repetitions",
            "system_prompt",
            "user_preamble",
            "response_code_pairs",
        },
        "experiment",
    )

    raw_pairs = experiment["response_code_pairs"]
    if not isinstance(raw_pairs, list):
        raise ValueError("response_code_pairs must be an array")
    pairs: list[tuple[str, str]] = []
    for index, pair in enumerate(raw_pairs):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"response_code_pairs[{index}] must contain exactly two codes")
        pairs.append(
            (
                _require_string(pair[0], f"response_code_pairs[{index}][0]"),
                _require_string(pair[1], f"response_code_pairs[{index}][1]"),
            )
        )

    raw_conditions = data["conditions"]
    if not isinstance(raw_conditions, list):
        raise ValueError("conditions must be an array of tables")
    conditions: list[Condition] = []
    for index, raw_condition in enumerate(raw_conditions):
        condition = _require_table(raw_condition, f"conditions[{index}]")
        _require_exact_keys(
            condition,
            {"id", "family", "template", "description"},
            f"conditions[{index}]",
        )
        conditions.append(
            Condition(
                id=_require_string(condition["id"], f"conditions[{index}].id"),
                family=_require_string(condition["family"], f"conditions[{index}].family"),
                template=_require_string(
                    condition["template"], f"conditions[{index}].template"
                ),
                description=_require_string(
                    condition["description"], f"conditions[{index}].description"
                ),
            )
        )

    return ExperimentConfig(
        name=_require_string(experiment["name"], "experiment.name"),
        seed=_require_int(experiment["seed"], "experiment.seed"),
        repetitions=_require_int(
            experiment["repetitions"], "experiment.repetitions", minimum=1
        ),
        system_prompt=_require_string(experiment["system_prompt"], "experiment.system_prompt"),
        user_preamble=_require_string(experiment["user_preamble"], "experiment.user_preamble"),
        response_code_pairs=tuple(pairs),
        conditions=tuple(conditions),
    )


def load_simulation_config(path: Path) -> SimulationConfig:
    """Load parameters for the synthetic demonstration backend."""
    data = _load_toml(path)
    _require_exact_keys(data, {"schema_version", "simulation", "strengths"}, "top-level")
    if data["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {data['schema_version']!r}")

    simulation = _require_table(data["simulation"], "simulation")
    _require_exact_keys(
        simulation,
        {"model_id", "seed", "invalid_rate", "first_position_bias"},
        "simulation",
    )
    strengths_table = _require_table(data["strengths"], "strengths")
    strengths: dict[str, float] = {}
    for condition_id, raw_strength in strengths_table.items():
        if type(raw_strength) not in {int, float}:
            raise ValueError(f"strength for {condition_id!r} must be numeric")
        strengths[condition_id] = float(raw_strength)

    raw_bias = simulation["first_position_bias"]
    if type(raw_bias) not in {int, float}:
        raise ValueError("simulation.first_position_bias must be numeric")

    return SimulationConfig(
        model_id=_require_string(simulation["model_id"], "simulation.model_id"),
        seed=_require_int(simulation["seed"], "simulation.seed"),
        invalid_rate=parse_probability(simulation["invalid_rate"]),
        first_position_bias=float(raw_bias),
        strengths=strengths,
    )
