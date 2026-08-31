from __future__ import annotations

from collections.abc import Callable

import pytest

from reasonese.bradley_terry import fit_bradley_terry
from reasonese.schemas import ScoredOutcome


def _pair_outcome(
    first: str,
    second: str,
    winner: str,
    index: int,
    *,
    model_id: str = "test_model",
    source: str = "external",
) -> ScoredOutcome:
    loser = second if winner == first else first
    matched_target = "KITE" if winner == first else "MOSS"
    return ScoredOutcome(
        schema_version=1,
        trial_id=f"trial_{index}",
        model_id=model_id,
        source=source,
        first_condition=first,
        second_condition=second,
        status="decisive",
        winner=winner,
        loser=loser,
        matched_target=matched_target,
    )


def test_fit_recovers_known_order_and_counts_invalid() -> None:
    outcomes: list[ScoredOutcome] = []
    index = 0
    for first, second, winner, count in (
        ("plain", "compact", "plain", 20),
        ("plain", "compact", "compact", 4),
        ("compact", "authority", "compact", 16),
        ("compact", "authority", "authority", 5),
        ("plain", "authority", "plain", 18),
        ("plain", "authority", "authority", 2),
    ):
        for _ in range(count):
            outcomes.append(_pair_outcome(first, second, winner, index))
            index += 1
    outcomes.append(
        ScoredOutcome(
            schema_version=1,
            trial_id="invalid",
            model_id="test_model",
            source="external",
            first_condition="plain",
            second_condition="compact",
            status="invalid",
            winner=None,
            loser=None,
            matched_target=None,
        )
    )

    result = fit_bradley_terry(outcomes, reference_condition="plain", ridge=0.5)
    assert result.converged
    assert result.decisive_trials == 65
    assert result.invalid_trials == 1
    assert [item.condition for item in result.ranking] == ["plain", "compact", "authority"]
    plain = result.ranking[0]
    assert plain.win_probability_vs_reference == pytest.approx(0.5)
    assert plain.wins == 38
    assert plain.losses == 6
    assert plain.invalid_trials == 1
    assert plain.standard_error > 0.0
    assert result.to_dict()["method"] == "l2_penalized_bradley_terry_logit"


def test_balanced_pair_has_equal_scores() -> None:
    outcomes = [
        _pair_outcome("plain", "compact", "plain", 0),
        _pair_outcome("plain", "compact", "compact", 1),
    ]
    result = fit_bradley_terry(outcomes)
    assert [item.log_strength for item in result.ranking] == pytest.approx([0.0, 0.0])
    assert all(item.win_probability_vs_reference == pytest.approx(0.5) for item in result.ranking)


def test_fit_rejects_invalid_options() -> None:
    outcomes = [_pair_outcome("plain", "compact", "plain", 0)]
    with pytest.raises(ValueError, match="ridge"):
        fit_bradley_terry(outcomes, ridge=0.0)
    with pytest.raises(ValueError, match="ridge"):
        fit_bradley_terry(outcomes, ridge=float("inf"))
    with pytest.raises(ValueError, match="tolerance"):
        fit_bradley_terry(outcomes, tolerance=0.0)
    with pytest.raises(ValueError, match="max_iterations"):
        fit_bradley_terry(outcomes, max_iterations=0)
    with pytest.raises(ValueError, match="unknown reference"):
        fit_bradley_terry(outcomes, reference_condition="missing")


def test_fit_rejects_empty_and_all_invalid(
    make_outcome: Callable[..., ScoredOutcome],
) -> None:
    with pytest.raises(ValueError, match="empty"):
        fit_bradley_terry([])
    invalid = make_outcome(status="invalid", winner=None, loser=None, matched_target=None)
    with pytest.raises(ValueError, match="decisive"):
        fit_bradley_terry([invalid])


def test_fit_rejects_duplicate_trial_ids() -> None:
    outcome = _pair_outcome("plain", "compact", "plain", 0)
    with pytest.raises(ValueError, match="unique"):
        fit_bradley_terry([outcome, outcome])


@pytest.mark.parametrize("field", ["model_id", "source"])
def test_fit_rejects_mixed_provenance(field: str) -> None:
    first = _pair_outcome("plain", "compact", "plain", 0)
    kwargs = {field: "different"}
    second = _pair_outcome("plain", "compact", "compact", 1, **kwargs)
    with pytest.raises(ValueError, match="one model_id and one source"):
        fit_bradley_terry([first, second])


def test_fit_rejects_malformed_pair(make_outcome: Callable[..., ScoredOutcome]) -> None:
    malformed = make_outcome(winner="third", loser="compact")
    with pytest.raises(ValueError, match="do not match"):
        fit_bradley_terry([malformed])


def test_fit_rejects_disconnected_decisive_graph(
    make_outcome: Callable[..., ScoredOutcome],
) -> None:
    decisive = _pair_outcome("plain", "compact", "plain", 0)
    invalid = make_outcome(
        trial_id="invalid",
        first_condition="third",
        second_condition="fourth",
        status="invalid",
        winner=None,
        loser=None,
        matched_target=None,
    )
    with pytest.raises(ValueError, match="disconnected"):
        fit_bradley_terry([decisive, invalid])


def test_fit_reports_nonconvergence_when_iteration_budget_is_tiny() -> None:
    outcomes = [_pair_outcome("plain", "compact", "plain", index) for index in range(10)]
    result = fit_bradley_terry(outcomes, max_iterations=1)
    assert not result.converged
    assert result.iterations == 1
