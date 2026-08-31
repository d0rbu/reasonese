"""Penalized Bradley-Terry estimation for condition rankings."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from reasonese.schemas import SCHEMA_VERSION, ScoredOutcome


@dataclass(frozen=True)
class RankedCondition:
    rank: int
    condition: str
    log_strength: float
    standard_error: float
    win_probability_vs_reference: float
    wins: int
    losses: int
    invalid_trials: int


@dataclass(frozen=True)
class BradleyTerryResult:
    schema_version: int
    method: str
    model_id: str
    source: str
    reference_condition: str
    total_trials: int
    decisive_trials: int
    invalid_trials: int
    ridge: float
    converged: bool
    iterations: int
    ranking: tuple[RankedCondition, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ranking"] = [asdict(item) for item in self.ranking]
        return data


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _objective(design: np.ndarray, scores: np.ndarray, ridge: float) -> float:
    logits = design @ scores
    return float(np.sum(-np.logaddexp(0.0, -logits)) - 0.5 * ridge * (scores @ scores))


def _require_connected(conditions: list[str], decisive: list[ScoredOutcome]) -> None:
    adjacency = {condition: set() for condition in conditions}
    for outcome in decisive:
        assert outcome.winner is not None and outcome.loser is not None
        adjacency[outcome.winner].add(outcome.loser)
        adjacency[outcome.loser].add(outcome.winner)

    visited: set[str] = set()
    frontier = [conditions[0]]
    while frontier:
        condition = frontier.pop()
        if condition in visited:
            continue
        visited.add(condition)
        frontier.extend(adjacency[condition] - visited)
    if visited != set(conditions):
        unreachable = sorted(set(conditions) - visited)
        raise ValueError(f"decisive comparison graph is disconnected: {unreachable}")


def fit_bradley_terry(
    outcomes: list[ScoredOutcome],
    *,
    reference_condition: str | None = None,
    ridge: float = 1.0,
    tolerance: float = 1.0e-10,
    max_iterations: int = 100,
) -> BradleyTerryResult:
    """Fit a centered, L2-penalized Bradley-Terry logit model."""
    if not outcomes:
        raise ValueError("cannot fit an empty outcome set")
    if not math.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("ridge must be finite and positive")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    trial_ids = [outcome.trial_id for outcome in outcomes]
    if len(trial_ids) != len(set(trial_ids)):
        raise ValueError("outcome trial_ids must be unique")

    model_ids = {outcome.model_id for outcome in outcomes}
    sources = {outcome.source for outcome in outcomes}
    if len(model_ids) != 1 or len(sources) != 1:
        raise ValueError("fit requires exactly one model_id and one source")

    conditions = sorted(
        {
            condition
            for outcome in outcomes
            for condition in (outcome.first_condition, outcome.second_condition)
        }
    )
    reference = reference_condition or ("plain" if "plain" in conditions else conditions[0])
    if reference not in conditions:
        raise ValueError(f"unknown reference condition: {reference!r}")

    decisive = [outcome for outcome in outcomes if outcome.status == "decisive"]
    if not decisive:
        raise ValueError("at least one decisive outcome is required")
    for outcome in decisive:
        if {outcome.winner, outcome.loser} != {
            outcome.first_condition,
            outcome.second_condition,
        }:
            raise ValueError(f"outcome {outcome.trial_id!r} winner/loser do not match its pair")
    _require_connected(conditions, decisive)

    condition_index = {condition: index for index, condition in enumerate(conditions)}
    design = np.zeros((len(decisive), len(conditions)), dtype=np.float64)
    for row, outcome in enumerate(decisive):
        assert outcome.winner is not None and outcome.loser is not None
        design[row, condition_index[outcome.winner]] = 1.0
        design[row, condition_index[outcome.loser]] = -1.0

    scores = np.zeros(len(conditions), dtype=np.float64)
    converged = False
    iterations = 0
    identity = np.eye(len(conditions), dtype=np.float64)
    for iteration in range(1, max_iterations + 1):
        iterations = iteration
        logits = design @ scores
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -35.0, 35.0)))
        gradient = design.T @ (1.0 - probabilities) - ridge * scores
        weights = probabilities * (1.0 - probabilities)
        information = design.T @ (weights[:, None] * design) + ridge * identity
        step = np.linalg.solve(information, gradient)

        current_objective = _objective(design, scores, ridge)
        scale = 1.0
        while scale > 1.0e-8:
            candidate = scores + scale * step
            candidate -= float(np.mean(candidate))
            if _objective(design, candidate, ridge) >= current_objective:
                break
            scale *= 0.5
        scores = candidate
        if float(np.max(np.abs(scale * step))) < tolerance:
            converged = True
            break

    final_logits = design @ scores
    final_probabilities = 1.0 / (1.0 + np.exp(-np.clip(final_logits, -35.0, 35.0)))
    final_weights = final_probabilities * (1.0 - final_probabilities)
    final_information = design.T @ (final_weights[:, None] * design) + ridge * identity
    standard_errors = np.sqrt(np.diag(np.linalg.inv(final_information)))

    wins = dict.fromkeys(conditions, 0)
    losses = dict.fromkeys(conditions, 0)
    invalid = dict.fromkeys(conditions, 0)
    for outcome in outcomes:
        if outcome.status == "decisive":
            assert outcome.winner is not None and outcome.loser is not None
            wins[outcome.winner] += 1
            losses[outcome.loser] += 1
        else:
            invalid[outcome.first_condition] += 1
            invalid[outcome.second_condition] += 1

    reference_score = float(scores[condition_index[reference]])
    ordered_indices = sorted(range(len(conditions)), key=lambda index: (-scores[index], conditions[index]))
    ranking = tuple(
        RankedCondition(
            rank=rank,
            condition=conditions[index],
            log_strength=float(scores[index]),
            standard_error=float(standard_errors[index]),
            win_probability_vs_reference=_sigmoid(float(scores[index]) - reference_score),
            wins=wins[conditions[index]],
            losses=losses[conditions[index]],
            invalid_trials=invalid[conditions[index]],
        )
        for rank, index in enumerate(ordered_indices, start=1)
    )
    return BradleyTerryResult(
        schema_version=SCHEMA_VERSION,
        method="l2_penalized_bradley_terry_logit",
        model_id=next(iter(model_ids)),
        source=next(iter(sources)),
        reference_condition=reference,
        total_trials=len(outcomes),
        decisive_trials=len(decisive),
        invalid_trials=len(outcomes) - len(decisive),
        ridge=ridge,
        converged=converged,
        iterations=iterations,
        ranking=ranking,
    )
