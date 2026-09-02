"""Bradley-Terry rankings and descriptive axis and position analyses."""

from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
from beartype import beartype
from numpy.typing import NDArray

from reasonese.observations import CellId, Observation
from reasonese.study import Cell

type AxisName = str
type TableRow = dict[str, object]
_AXES = ("instruction", "framing", "channel", "author", "assistant")


def _as_float(value: object) -> float:
    if not isinstance(value, int | float):
        raise TypeError("expected a numeric table value")
    return float(value)


def _as_int(value: object) -> int:
    if not isinstance(value, int):
        raise TypeError("expected an integer table value")
    return value


@beartype
@dataclass(frozen=True, slots=True)
class Comparison:
    """One within-trial Bradley-Terry comparison, with ties represented by 0.5."""

    trial_id: str
    first: CellId
    second: CellId
    outcome: float


@beartype
@dataclass(frozen=True, slots=True)
class RankedCell:
    """One cell's fitted Bradley-Terry result and raw completion summary."""

    rank: int
    cell_id: CellId
    cell: Cell
    score: float
    standard_error: float
    completions: int
    observations: int
    completion_rate: float
    bootstrap_low: float | None
    bootstrap_high: float | None


@beartype
@dataclass(frozen=True, slots=True)
class BradleyTerryFit:
    """A total L2-regularized ordering plus fit diagnostics."""

    ranking: tuple[RankedCell, ...]
    converged: bool
    iterations: int
    objective: float
    comparison_count: int
    tie_count: int
    connected_components: tuple[tuple[CellId, ...], ...]


@beartype
@dataclass(frozen=True, slots=True)
class AnalysisBundle:
    """All tabular and diagnostic analyses emitted together."""

    fit: BradleyTerryFit
    axis_summary: tuple[TableRow, ...]
    axis_comparisons: tuple[TableRow, ...]
    position_summary: tuple[TableRow, ...]
    cell_position_effects: tuple[TableRow, ...]
    axis_position_effects: tuple[TableRow, ...]
    order_sensitivity: tuple[TableRow, ...]
    regularization_sensitivity: tuple[TableRow, ...]
    diagnostics: dict[str, object]


def _axis_value(observation: Observation, axis: AxisName) -> str:
    if axis == "instruction":
        return str(observation.spec.instruction)
    if axis == "framing":
        return str(observation.spec.framing)
    if axis == "channel":
        return str(observation.spec.channel)
    if axis == "author":
        return str(observation.spec.author)
    if axis == "assistant":
        return str(observation.assistant)
    raise ValueError(f"unknown axis {axis!r}")


def _cell(observation: Observation) -> Cell:
    return Cell(observation.spec, observation.assistant)


@beartype
def validate_observations(observations: tuple[Observation, ...]) -> None:
    """Reject incomplete, duplicated, or internally inconsistent trial data."""
    if not observations:
        raise ValueError("at least one observation is required")
    seen_rows: set[tuple[str, CellId]] = set()
    cell_coordinates: dict[CellId, Cell] = {}
    by_trial: dict[str, list[Observation]] = defaultdict(list)
    for observation in observations:
        row_key = (str(observation.trial_id), observation.cell_id)
        if row_key in seen_rows:
            raise ValueError("each trial may contain only one observation per cell")
        seen_rows.add(row_key)
        known = cell_coordinates.setdefault(observation.cell_id, _cell(observation))
        if known != _cell(observation):
            raise ValueError("one cell_id maps to multiple coordinate tuples")
        by_trial[str(observation.trial_id)].append(observation)

    for trial_id, rows in by_trial.items():
        if len(rows) != 2:
            raise ValueError(f"trial {trial_id} must contain exactly two cells")
        positions = sorted(int(row.position) for row in rows)
        if positions != [1, 2]:
            raise ValueError(f"trial {trial_id} positions must be exactly 1 and 2")
        if len({int(row.permutation) for row in rows}) != 1:
            raise ValueError(f"trial {trial_id} has inconsistent permutation metadata")
        if len({int(row.rollout) for row in rows}) != 1:
            raise ValueError(f"trial {trial_id} has inconsistent rollout metadata")
        if len({row.assistant for row in rows}) != 1:
            raise ValueError(f"trial {trial_id} has multiple assistant models")
        if len({row.trace_fingerprint for row in rows}) != 1:
            raise ValueError(f"trial {trial_id} has multiple trace fingerprints")


@beartype
def build_comparisons(observations: tuple[Observation, ...]) -> tuple[Comparison, ...]:
    """Convert each trial's two independent verdicts into one pairwise outcome."""
    validate_observations(observations)
    by_trial: dict[str, list[Observation]] = defaultdict(list)
    for observation in observations:
        by_trial[str(observation.trial_id)].append(observation)
    comparisons: list[Comparison] = []
    for trial_id, rows in by_trial.items():
        first, second = sorted(rows, key=lambda row: str(row.cell_id))
        outcome = 0.5 if first.completed == second.completed else float(first.completed)
        comparisons.append(Comparison(trial_id, first.cell_id, second.cell_id, outcome))
    return tuple(comparisons)


def _connected_components(
    cell_ids: tuple[CellId, ...], comparisons: tuple[Comparison, ...]
) -> tuple[tuple[CellId, ...], ...]:
    neighbors = {cell_id: set() for cell_id in cell_ids}
    for comparison in comparisons:
        neighbors[comparison.first].add(comparison.second)
        neighbors[comparison.second].add(comparison.first)
    remaining = set(cell_ids)
    components: list[tuple[CellId, ...]] = []
    while remaining:
        start = min(remaining)
        stack = [start]
        component: set[CellId] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(neighbors[current] - component)
        remaining -= component
        components.append(tuple(sorted(component)))
    return tuple(sorted(components, key=lambda component: str(component[0])))


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _fit_scores(
    cell_ids: tuple[CellId, ...],
    comparisons: tuple[Comparison, ...],
    l2: float,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> tuple[NDArray[np.float64], NDArray[np.float64], bool, int, float]:
    if l2 <= 0:
        raise ValueError("L2 penalty must be positive")
    index = {cell_id: position for position, cell_id in enumerate(cell_ids)}
    scores = np.zeros(len(cell_ids), dtype=np.float64)
    converged = False
    iterations = 0
    for iteration in range(1, max_iterations + 1):
        iterations = iteration
        gradient = l2 * scores
        hessian = np.eye(len(cell_ids), dtype=np.float64) * l2
        for comparison in comparisons:
            first = index[comparison.first]
            second = index[comparison.second]
            probability = _sigmoid(float(scores[first] - scores[second]))
            residual = probability - comparison.outcome
            curvature = probability * (1.0 - probability)
            gradient[first] += residual
            gradient[second] -= residual
            hessian[first, first] += curvature
            hessian[second, second] += curvature
            hessian[first, second] -= curvature
            hessian[second, first] -= curvature
        step = np.linalg.solve(hessian, gradient)
        scores -= step
        scores -= scores.mean()
        if float(np.max(np.abs(step))) < tolerance:
            converged = True
            break

    final_hessian = np.eye(len(cell_ids), dtype=np.float64) * l2
    objective = 0.5 * l2 * float(scores @ scores)
    for comparison in comparisons:
        first = index[comparison.first]
        second = index[comparison.second]
        difference = float(scores[first] - scores[second])
        probability = _sigmoid(difference)
        curvature = probability * (1.0 - probability)
        final_hessian[first, first] += curvature
        final_hessian[second, second] += curvature
        final_hessian[first, second] -= curvature
        final_hessian[second, first] -= curvature
        objective += float(np.logaddexp(0.0, difference) - comparison.outcome * difference)
    standard_errors = np.sqrt(np.diag(np.linalg.inv(final_hessian)))
    return scores, standard_errors, converged, iterations, objective


def _bootstrap_intervals(
    cell_ids: tuple[CellId, ...],
    comparisons: tuple[Comparison, ...],
    l2: float,
    samples: int,
    seed: int,
) -> dict[CellId, tuple[float, float]]:
    if samples < 0:
        raise ValueError("bootstrap samples must be non-negative")
    if samples == 0:
        return {}
    by_trial: dict[str, list[Comparison]] = defaultdict(list)
    for comparison in comparisons:
        by_trial[comparison.trial_id].append(comparison)
    trial_groups = tuple(by_trial.values())
    generator = np.random.default_rng(seed)
    estimates = np.empty((samples, len(cell_ids)), dtype=np.float64)
    for sample_index in range(samples):
        selected = generator.integers(0, len(trial_groups), size=len(trial_groups))
        resampled = tuple(
            comparison for group_index in selected for comparison in trial_groups[int(group_index)]
        )
        estimates[sample_index] = _fit_scores(cell_ids, resampled, l2)[0]
    lower = np.percentile(estimates, 2.5, axis=0)
    upper = np.percentile(estimates, 97.5, axis=0)
    return {
        cell_id: (float(lower[index]), float(upper[index]))
        for index, cell_id in enumerate(cell_ids)
    }


@beartype
def fit_bradley_terry(
    observations: tuple[Observation, ...],
    l2: float,
    *,
    bootstrap_samples: int = 0,
    seed: int = 0,
) -> BradleyTerryFit:
    """Fit an L2-penalized total ordering with trial-cluster bootstrap intervals."""
    comparisons = build_comparisons(observations)
    cells = {observation.cell_id: _cell(observation) for observation in observations}
    cell_ids = tuple(sorted(cells))
    scores, standard_errors, converged, iterations, objective = _fit_scores(
        cell_ids, comparisons, l2
    )
    intervals = _bootstrap_intervals(cell_ids, comparisons, l2, bootstrap_samples, seed)
    completions = Counter(
        observation.cell_id for observation in observations if observation.completed
    )
    counts = Counter(observation.cell_id for observation in observations)
    index = {cell_id: position for position, cell_id in enumerate(cell_ids)}
    ordered_ids = sorted(cell_ids, key=lambda cell_id: (-scores[index[cell_id]], str(cell_id)))
    ranking = tuple(
        RankedCell(
            rank,
            cell_id,
            cells[cell_id],
            float(scores[index[cell_id]]),
            float(standard_errors[index[cell_id]]),
            completions[cell_id],
            counts[cell_id],
            completions[cell_id] / counts[cell_id],
            intervals.get(cell_id, (None, None))[0],
            intervals.get(cell_id, (None, None))[1],
        )
        for rank, cell_id in enumerate(ordered_ids, start=1)
    )
    return BradleyTerryFit(
        ranking,
        converged,
        iterations,
        objective,
        len(comparisons),
        sum(comparison.outcome == 0.5 for comparison in comparisons),
        _connected_components(cell_ids, comparisons),
    )


def _wilson(successes: int, count: int) -> tuple[float, float]:
    z = 1.959963984540054
    rate = successes / count
    denominator = 1.0 + z * z / count
    center = (rate + z * z / (2.0 * count)) / denominator
    half_width = z * math.sqrt(rate * (1.0 - rate) / count + z * z / (4 * count * count))
    return center - half_width / denominator, center + half_width / denominator


def _correlation(rows: list[Observation]) -> float | None:
    positions = np.array([int(row.position) for row in rows], dtype=np.float64)
    outcomes = np.array([int(row.completed) for row in rows], dtype=np.float64)
    if len(rows) < 2 or float(positions.std()) == 0.0 or float(outcomes.std()) == 0.0:
        return None
    return float(np.corrcoef(positions, outcomes)[0, 1])


def _cell_fields(cell: Cell) -> dict[str, str]:
    return {
        "instruction": str(cell.spec.instruction),
        "framing": str(cell.spec.framing),
        "channel": str(cell.spec.channel),
        "author": str(cell.spec.author),
        "assistant": str(cell.assistant),
    }


def _rate_row(rows: list[Observation]) -> tuple[int, int, float, float, float]:
    count = len(rows)
    successes = sum(row.completed for row in rows)
    low, high = _wilson(successes, count)
    return count, successes, successes / count, low, high


def _axis_tables(
    observations: tuple[Observation, ...], fit: BradleyTerryFit
) -> tuple[tuple[TableRow, ...], tuple[TableRow, ...]]:
    score_by_cell = {ranked.cell_id: ranked.score for ranked in fit.ranking}
    summary: list[TableRow] = []
    comparisons: list[TableRow] = []
    for axis in _AXES:
        groups: dict[str, list[Observation]] = defaultdict(list)
        for observation in observations:
            groups[_axis_value(observation, axis)].append(observation)
        summary_by_value: dict[str, TableRow] = {}
        for value, rows in sorted(groups.items()):
            count, successes, rate, low, high = _rate_row(rows)
            unique_cells = {row.cell_id for row in rows}
            row: TableRow = {
                "axis": axis,
                "value": value,
                "observations": count,
                "completions": successes,
                "completion_rate": rate,
                "wilson_low": low,
                "wilson_high": high,
                "cells": len(unique_cells),
                "mean_bt_score": sum(score_by_cell[item] for item in unique_cells)
                / len(unique_cells),
            }
            summary.append(row)
            summary_by_value[value] = row
        for first, second in itertools.combinations(sorted(groups), 2):
            first_row = summary_by_value[first]
            second_row = summary_by_value[second]
            first_successes = _as_int(first_row["completions"])
            second_successes = _as_int(second_row["completions"])
            first_failures = _as_int(first_row["observations"]) - first_successes
            second_failures = _as_int(second_row["observations"]) - second_successes
            first_odds = (first_successes + 0.5) / (first_failures + 0.5)
            second_odds = (second_successes + 0.5) / (second_failures + 0.5)
            comparisons.append(
                {
                    "axis": axis,
                    "first": first,
                    "second": second,
                    "completion_rate_difference": _as_float(first_row["completion_rate"])
                    - _as_float(second_row["completion_rate"]),
                    "odds_ratio": first_odds / second_odds,
                    "mean_bt_score_difference": _as_float(first_row["mean_bt_score"])
                    - _as_float(second_row["mean_bt_score"]),
                }
            )
    return tuple(summary), tuple(comparisons)


def _position_tables(
    observations: tuple[Observation, ...],
) -> tuple[tuple[TableRow, ...], tuple[TableRow, ...], tuple[TableRow, ...], tuple[TableRow, ...]]:
    by_position: dict[int, list[Observation]] = defaultdict(list)
    by_cell: dict[CellId, list[Observation]] = defaultdict(list)
    for observation in observations:
        by_position[int(observation.position)].append(observation)
        by_cell[observation.cell_id].append(observation)
    position_summary: list[TableRow] = []
    for position, rows in sorted(by_position.items()):
        values = _rate_row(rows)
        position_summary.append(
            {
                "position": position,
                "observations": values[0],
                "completions": values[1],
                "completion_rate": values[2],
                "wilson_low": values[3],
                "wilson_high": values[4],
            }
        )

    cell_position: list[TableRow] = []
    order_sensitivity: list[TableRow] = []
    for cell_identifier, rows in sorted(by_cell.items()):
        overall_rate = sum(row.completed for row in rows) / len(rows)
        correlation = _correlation(rows)
        position_rates: list[float] = []
        for position in sorted({int(row.position) for row in rows}):
            positioned = [row for row in rows if int(row.position) == position]
            count, successes, rate, low, high = _rate_row(positioned)
            position_rates.append(rate)
            cell_position.append(
                {
                    "cell_id": str(cell_identifier),
                    **_cell_fields(_cell(rows[0])),
                    "position": position,
                    "observations": count,
                    "completions": successes,
                    "completion_rate": rate,
                    "wilson_low": low,
                    "wilson_high": high,
                    "delta_from_cell_rate": rate - overall_rate,
                }
            )
        order_sensitivity.append(
            {
                "kind": "cell",
                "cell_id": str(cell_identifier),
                **_cell_fields(_cell(rows[0])),
                "value": str(cell_identifier),
                "observations": len(rows),
                "position_correlation": correlation,
                "min_position_rate": min(position_rates),
                "max_position_rate": max(position_rates),
                "position_rate_range": max(position_rates) - min(position_rates),
            }
        )

    axis_position: list[TableRow] = []
    for axis in _AXES:
        groups: dict[str, list[Observation]] = defaultdict(list)
        for observation in observations:
            groups[_axis_value(observation, axis)].append(observation)
        for value, rows in sorted(groups.items()):
            overall_rate = sum(row.completed for row in rows) / len(rows)
            correlation = _correlation(rows)
            position_rates: list[float] = []
            for position in sorted({int(row.position) for row in rows}):
                positioned = [row for row in rows if int(row.position) == position]
                count, successes, rate, low, high = _rate_row(positioned)
                position_rates.append(rate)
                axis_position.append(
                    {
                        "axis": axis,
                        "value": value,
                        "position": position,
                        "observations": count,
                        "completions": successes,
                        "completion_rate": rate,
                        "wilson_low": low,
                        "wilson_high": high,
                        "delta_from_axis_value_rate": rate - overall_rate,
                        "position_correlation": correlation,
                    }
                )
            order_sensitivity.append(
                {
                    "kind": "axis",
                    "axis": axis,
                    "value": value,
                    "observations": len(rows),
                    "position_correlation": correlation,
                    "min_position_rate": min(position_rates),
                    "max_position_rate": max(position_rates),
                    "position_rate_range": max(position_rates) - min(position_rates),
                }
            )
    order_sensitivity.sort(
        key=lambda row: (-float(row["position_rate_range"]), str(row["kind"]), str(row["value"]))
    )
    return (
        tuple(position_summary),
        tuple(cell_position),
        tuple(axis_position),
        tuple(order_sensitivity),
    )


def _regularization_table(
    observations: tuple[Observation, ...], primary_l2: float
) -> tuple[TableRow, ...]:
    rows: list[TableRow] = []
    for penalty in (primary_l2 / 10.0, primary_l2, primary_l2 * 10.0):
        fit = fit_bradley_terry(observations, penalty)
        for ranked in fit.ranking:
            rows.append(
                {
                    "l2": penalty,
                    "cell_id": str(ranked.cell_id),
                    "rank": ranked.rank,
                    "score": ranked.score,
                }
            )
    return tuple(rows)


def _diagnostics(
    observations: tuple[Observation, ...],
    fit: BradleyTerryFit,
    regularization: tuple[TableRow, ...],
) -> dict[str, object]:
    trials = {str(observation.trial_id) for observation in observations}
    cells = {observation.cell_id for observation in observations}
    counts_by_cell_position = Counter(
        (observation.cell_id, int(observation.position)) for observation in observations
    )
    balance_rows: list[dict[str, object]] = []
    for cell_identifier in sorted(cells):
        position_counts = {
            position: count
            for (candidate, position), count in counts_by_cell_position.items()
            if candidate == cell_identifier
        }
        balance_rows.append(
            {
                "cell_id": str(cell_identifier),
                "positions": sorted(position_counts),
                "count_min": min(position_counts.values()),
                "count_max": max(position_counts.values()),
                "balanced": len(set(position_counts.values())) == 1
                and sorted(position_counts) == list(range(1, max(position_counts) + 1)),
            }
        )

    ranks_by_l2: dict[float, dict[str, int]] = defaultdict(dict)
    for row in regularization:
        ranks_by_l2[_as_float(row["l2"])][str(row["cell_id"])] = _as_int(row["rank"])
    primary_penalty = sorted(ranks_by_l2)[1]
    primary_ranks = ranks_by_l2[primary_penalty]
    sensitivity: list[dict[str, object]] = []
    for penalty, ranks in sorted(ranks_by_l2.items()):
        first = np.array([primary_ranks[cell] for cell in sorted(primary_ranks)], dtype=float)
        second = np.array([ranks[cell] for cell in sorted(primary_ranks)], dtype=float)
        correlation = 1.0 if len(first) == 1 else float(np.corrcoef(first, second)[0, 1])
        sensitivity.append(
            {
                "l2": penalty,
                "rank_correlation_with_primary": correlation,
                "max_absolute_rank_shift": max(
                    abs(ranks[cell] - primary_ranks[cell]) for cell in primary_ranks
                ),
            }
        )
    return {
        "observations": len(observations),
        "trials": len(trials),
        "cells": len(cells),
        "comparison_count": fit.comparison_count,
        "tie_comparisons": fit.tie_count,
        "comparison_graph_components": [
            [str(cell_id) for cell_id in component] for component in fit.connected_components
        ],
        "comparison_graph_connected": len(fit.connected_components) == 1,
        "position_balance": balance_rows,
        "position_balanced": all(bool(row["balanced"]) for row in balance_rows),
        "regularization_sensitivity": sensitivity,
        "fit_converged": fit.converged,
        "fit_iterations": fit.iterations,
        "fit_objective": fit.objective,
    }


@beartype
def analyze_observations(
    observations: tuple[Observation, ...],
    l2: float,
    *,
    bootstrap_samples: int,
    seed: int,
) -> AnalysisBundle:
    """Run ranking, axis, order, balance, and sensitivity analyses."""
    fit = fit_bradley_terry(
        observations,
        l2,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    axis_summary, axis_comparisons = _axis_tables(observations, fit)
    position, cell_position, axis_position, order_sensitivity = _position_tables(observations)
    regularization = _regularization_table(observations, l2)
    diagnostics = _diagnostics(observations, fit, regularization)
    return AnalysisBundle(
        fit,
        axis_summary,
        axis_comparisons,
        position,
        cell_position,
        axis_position,
        order_sensitivity,
        regularization,
        diagnostics,
    )
