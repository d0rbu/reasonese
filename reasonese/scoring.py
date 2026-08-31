"""Exact-match conversion from raw responses to pairwise outcomes."""

from __future__ import annotations

from reasonese.schemas import SCHEMA_VERSION, ResponseRecord, ScoredOutcome, Trial


def _unique_by_trial_id(records: list[Trial] | list[ResponseRecord], name: str) -> dict[str, object]:
    indexed: dict[str, object] = {}
    for record in records:
        if record.trial_id in indexed:
            raise ValueError(f"duplicate {name} trial_id: {record.trial_id}")
        indexed[record.trial_id] = record
    return indexed


def score_responses(
    trials: list[Trial],
    responses: list[ResponseRecord],
) -> list[ScoredOutcome]:
    """Require a complete one-to-one response set and retain invalid outputs."""
    trial_index = _unique_by_trial_id(trials, "design")
    response_index = _unique_by_trial_id(responses, "response")
    trial_ids = set(trial_index)
    response_ids = set(response_index)
    missing = sorted(trial_ids - response_ids)
    unknown = sorted(response_ids - trial_ids)
    if missing or unknown:
        raise ValueError(f"response/design mismatch: missing={missing}, unknown={unknown}")

    outcomes: list[ScoredOutcome] = []
    for trial in trials:
        response = response_index[trial.trial_id]
        assert isinstance(response, ResponseRecord)
        normalized = response.response_text.strip()
        if normalized == trial.first_target:
            winner = trial.first_condition
            loser = trial.second_condition
            matched_target = trial.first_target
            status = "decisive"
        elif normalized == trial.second_target:
            winner = trial.second_condition
            loser = trial.first_condition
            matched_target = trial.second_target
            status = "decisive"
        else:
            winner = None
            loser = None
            matched_target = None
            status = "invalid"
        outcomes.append(
            ScoredOutcome(
                schema_version=SCHEMA_VERSION,
                trial_id=trial.trial_id,
                model_id=response.model_id,
                source=response.source,
                first_condition=trial.first_condition,
                second_condition=trial.second_condition,
                status=status,
                winner=winner,
                loser=loser,
                matched_target=matched_target,
            )
        )
    return outcomes
