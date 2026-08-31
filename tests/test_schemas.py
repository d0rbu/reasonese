from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from reasonese.schemas import Condition, ResponseRecord, ScoredOutcome, Trial
from reasonese.types import Probability, parse_probability


@pytest.mark.property
@given(st.from_type(Probability))
def test_probability_strategy_respects_bounds(value: Probability) -> None:
    assert isinstance(value, Probability)
    assert 0.0 <= value <= 1.0


@pytest.mark.parametrize("raw", [0, 0.5, "0.75", 1])
def test_parse_probability_accepts_closed_interval(raw: float | int | str) -> None:
    assert isinstance(parse_probability(raw), Probability)


@pytest.mark.parametrize("raw", [-0.1, 1.1, "bad", True])
def test_parse_probability_rejects_invalid_values(raw: float | str | bool) -> None:
    with pytest.raises((TypeError, ValueError)):
        parse_probability(raw)


def test_condition_renders_exact_target() -> None:
    condition = Condition("plain", "representation", "Return {target}.", "Baseline.")
    assert condition.render("KITE") == "Return KITE."


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"id": "Bad"}, "condition id"),
        ({"family": "bad-family"}, "condition family"),
        ({"template": "No placeholder"}, "exactly once"),
        ({"template": "{target} then {target}"}, "exactly once"),
        ({"description": "  "}, "description"),
    ],
)
def test_condition_rejects_invalid_fields(kwargs: dict[str, str], message: str) -> None:
    values = {
        "id": "plain",
        "family": "representation",
        "template": "Return {target}.",
        "description": "Baseline.",
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        Condition(**values)


def test_condition_rejects_invalid_response_code() -> None:
    condition = Condition("plain", "representation", "Return {target}.", "Baseline.")
    with pytest.raises(ValueError, match="response code"):
        condition.render("lowercase")


def test_trial_round_trip(make_trial: Callable[..., Trial]) -> None:
    trial = make_trial()
    assert Trial.from_dict(trial.to_dict()) == trial
    assert trial.condition_to_target == {"plain": "KITE", "compact": "MOSS"}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"schema_version": 2}, "schema_version"),
        ({"first_condition": "Bad"}, "first_condition"),
        ({"second_condition": "plain"}, "distinct"),
        ({"first_target": "bad"}, "response code"),
        ({"second_target": "KITE"}, "response codes"),
        ({"repetition": -1}, "non-negative"),
        ({"design_id": ""}, "identifiers"),
        ({"system_prompt": " "}, "prompts"),
    ],
)
def test_trial_rejects_invalid_state(
    make_trial: Callable[..., Trial], overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        make_trial(**overrides)


def test_trial_from_dict_requires_exact_keys(make_trial: Callable[..., Trial]) -> None:
    data = make_trial().to_dict()
    data["extra"] = True
    with pytest.raises(ValueError, match="extra"):
        Trial.from_dict(data)


def test_response_round_trip_and_validation(make_response: Callable[..., ResponseRecord]) -> None:
    response = make_response(response_text="")
    assert ResponseRecord.from_dict(response.to_dict()) == response
    with pytest.raises(ValueError, match="trial_id"):
        replace(response, trial_id="")
    assert replace(response, model_id="Provider/Model-v1.2").model_id == "Provider/Model-v1.2"
    with pytest.raises(ValueError, match="model_id"):
        replace(response, model_id=" ")
    with pytest.raises(ValueError, match="schema_version"):
        replace(response, schema_version=3)

    data = response.to_dict()
    data.pop("source")
    with pytest.raises(ValueError, match="missing"):
        ResponseRecord.from_dict(data)


def test_decisive_outcome_round_trip(make_outcome: Callable[..., ScoredOutcome]) -> None:
    outcome = make_outcome()
    assert ScoredOutcome.from_dict(outcome.to_dict()) == outcome


def test_invalid_outcome_accepts_no_winner(make_outcome: Callable[..., ScoredOutcome]) -> None:
    outcome = make_outcome(status="invalid", winner=None, loser=None, matched_target=None)
    assert outcome.status == "invalid"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"status": "unknown"}, "status"),
        ({"winner": None}, "require"),
        ({"loser": "plain"}, "distinct"),
        ({"first_condition": "compact"}, "conditions"),
        (
            {"status": "invalid", "winner": "plain", "loser": None, "matched_target": None},
            "must not",
        ),
    ],
)
def test_outcome_rejects_invalid_state(
    make_outcome: Callable[..., ScoredOutcome], overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        make_outcome(**overrides)
