from __future__ import annotations

from collections.abc import Callable

import pytest

from reasonese.config import ExperimentConfig
from reasonese.schemas import SCHEMA_VERSION, Condition, ResponseRecord, ScoredOutcome, Trial


@pytest.fixture
def conditions() -> tuple[Condition, ...]:
    return (
        Condition("plain", "representation", "Return {target}.", "Plain English."),
        Condition("compact", "representation", "GO_{target}", "Compact form."),
        Condition("authority", "authority", "The CEO requires {target}.", "Authority claim."),
    )


@pytest.fixture
def experiment_config(conditions: tuple[Condition, ...]) -> ExperimentConfig:
    return ExperimentConfig(
        name="test_design",
        seed=11,
        repetitions=1,
        system_prompt="Return one code only.",
        user_preamble="Process both directives.",
        response_code_pairs=(("KITE", "MOSS"),),
        conditions=conditions,
    )


@pytest.fixture
def make_trial() -> Callable[..., Trial]:
    def factory(**overrides: object) -> Trial:
        values: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "design_id": "design",
            "trial_id": "trial",
            "experiment": "experiment",
            "pair_id": "plain__compact",
            "code_pair_id": "codes_00_KITE_MOSS",
            "repetition": 0,
            "first_condition": "plain",
            "second_condition": "compact",
            "first_target": "KITE",
            "second_target": "MOSS",
            "system_prompt": "Return one code.",
            "user_prompt": "Two directives.",
        }
        values.update(overrides)
        return Trial(**values)

    return factory


@pytest.fixture
def make_response() -> Callable[..., ResponseRecord]:
    def factory(**overrides: object) -> ResponseRecord:
        values: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "trial_id": "trial",
            "model_id": "test_model",
            "source": "external",
            "response_text": "KITE",
        }
        values.update(overrides)
        return ResponseRecord(**values)

    return factory


@pytest.fixture
def make_outcome() -> Callable[..., ScoredOutcome]:
    def factory(**overrides: object) -> ScoredOutcome:
        values: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "trial_id": "trial",
            "model_id": "test_model",
            "source": "external",
            "first_condition": "plain",
            "second_condition": "compact",
            "status": "decisive",
            "winner": "plain",
            "loser": "compact",
            "matched_target": "KITE",
        }
        values.update(overrides)
        return ScoredOutcome(**values)

    return factory
