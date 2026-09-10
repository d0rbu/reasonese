"""Bradley-Terry rankings and descriptive axis and position analyses."""

from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
from beartype import beartype
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from reasonese.instructions import InstructionPair, PairMembership, instruction_index
from reasonese.observations import CellId, Observation
from reasonese.study import Cell

type AxisName = str
type TableRow = dict[str, object]
# Instruction is not a treatment axis under a bipartite bank. A trial only ever
# holds the two instructions of one pair, so the comparison graph has no edges
# between pairs and an instruction contrast would difference arbitrary
# per-component offsets. These three axes are the only ones that vary within a
# trial, so they are the only ones a Bradley-Terry score can speak to.
_AXES = ("framing", "channel", "author")
# Constant within a trial, so their mean score is zero once components
# self-center. Reported as descriptive strata without a Bradley-Terry column.
_STRATA = ("assistant", "skill", "conflict", "pair")


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
    component_index: int
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
    stratum_summary: tuple[TableRow, ...]
    pair_exclusivity: tuple[TableRow, ...]
    position_summary: tuple[TableRow, ...]
    cell_position_effects: tuple[TableRow, ...]
    axis_position_effects: tuple[TableRow, ...]
    order_sensitivity: tuple[TableRow, ...]
    regularization_sensitivity: tuple[TableRow, ...]
    diagnostics: dict[str, object]


def _axis_value(observation: Observation, axis: AxisName) -> str:
    if axis == "framing":
        return str(observation.spec.framing)
    if axis == "channel":
        return str(observation.spec.channel)
    if axis == "author":
        return str(observation.spec.author)
    raise ValueError(f"unknown axis {axis!r}")


def _stratum_value(
    observation: Observation,
    membership: PairMembership,
    stratum: AxisName,
) -> str:
    if stratum == "assistant":
        return str(observation.assistant)
    if stratum == "skill":
        return str(membership.pair.skill)
    if stratum == "conflict":
        return str(membership.pair.conflict)
    if stratum == "pair":
        return str(membership.pair.pair_id)
    raise ValueError(f"unknown stratum {stratum!r}")


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


@beartype
def pair_memberships(
    observations: tuple[Observation, ...],
    pairs: tuple[InstructionPair, ...],
) -> dict[CellId, PairMembership]:
    """Map every observed cell to its instruction pair and side.

    Rejects instructions absent from the bank, and trials whose two cells are
    not the two opposite sides of one pair. Both would silently break the
    blocking structure that replaces the instruction axis.
    """
    index = instruction_index(pairs)
    memberships: dict[CellId, PairMembership] = {}
    by_trial: dict[str, list[Observation]] = defaultdict(list)
    for observation in observations:
        membership = index.get(observation.spec.instruction)
        if membership is None:
            raise ValueError(
                "observed instruction is absent from the pair bank: "
                f"{observation.spec.instruction}"
            )
        memberships[observation.cell_id] = membership
        by_trial[str(observation.trial_id)].append(observation)

    for trial_id, rows in by_trial.items():
        pair_ids = {memberships[row.cell_id].pair.pair_id for row in rows}
        sides = {memberships[row.cell_id].side for row in rows}
        if len(pair_ids) != 1:
            raise ValueError(f"trial {trial_id} mixes instructions from different pairs")
        if len(sides) != len(rows):
            raise ValueError(f"trial {trial_id} repeats one side of its instruction pair")
    return memberships


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


def _sigmoid_array(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate the logistic function without overflowing on either tail."""
    exponential = np.exp(-np.abs(values))
    return np.where(values >= 0.0, 1.0, exponential) / (1.0 + exponential)


@dataclass(frozen=True, slots=True)
class _Endpoints:
    """Comparison endpoints as index arrays, derived once and reused every step."""

    first: NDArray[np.intp]
    second: NDArray[np.intp]
    outcomes: NDArray[np.float64]
    curvature_rows: NDArray[np.intp]
    curvature_columns: NDArray[np.intp]


def _endpoints(cell_ids: tuple[CellId, ...], comparisons: tuple[Comparison, ...]) -> _Endpoints:
    index = {cell_id: position for position, cell_id in enumerate(cell_ids)}
    count = len(comparisons)
    first = np.fromiter(
        (index[comparison.first] for comparison in comparisons), dtype=np.intp, count=count
    )
    second = np.fromiter(
        (index[comparison.second] for comparison in comparisons), dtype=np.intp, count=count
    )
    outcomes = np.fromiter(
        (comparison.outcome for comparison in comparisons), dtype=np.float64, count=count
    )
    # Each comparison adds curvature to two diagonal entries and subtracts it
    # from the two symmetric off-diagonal ones.
    return _Endpoints(
        first,
        second,
        outcomes,
        np.concatenate((first, second, first, second)),
        np.concatenate((first, second, second, first)),
    )


@dataclass(frozen=True, slots=True)
class _ComponentFit:
    """Fitted scores, with the extras present only when they were asked for."""

    scores: NDArray[np.float64]
    converged: bool
    iterations: int
    standard_errors: NDArray[np.float64] | None
    objective: float | None


def _penalized_hessian(
    endpoints: _Endpoints,
    curvature: NDArray[np.float64],
    size: int,
    l2: float,
) -> NDArray[np.float64]:
    """Accumulate every comparison's curvature into one dense penalized Hessian."""
    weights = np.concatenate((curvature, curvature, -curvature, -curvature))
    flat = np.bincount(
        endpoints.curvature_rows * size + endpoints.curvature_columns,
        weights=weights,
        minlength=size * size,
    )
    hessian = flat.reshape(size, size)
    hessian[np.diag_indices(size)] += l2
    return hessian


def _fit_scores(
    cell_ids: tuple[CellId, ...],
    comparisons: tuple[Comparison, ...],
    l2: float,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
    diagnostics: bool = True,
) -> _ComponentFit:
    """Run penalized Newton steps over one component.

    ``diagnostics`` covers the standard errors and the objective. Both cost a
    further pass over the comparisons and a matrix inversion, and the bootstrap
    reads neither, so it turns them off.
    """
    if l2 <= 0:
        raise ValueError("L2 penalty must be positive")
    size = len(cell_ids)
    endpoints = _endpoints(cell_ids, comparisons)
    scores = np.zeros(size, dtype=np.float64)
    converged = False
    iterations = 0
    for iteration in range(1, max_iterations + 1):
        iterations = iteration
        difference = scores[endpoints.first] - scores[endpoints.second]
        probability = _sigmoid_array(difference)
        residual = probability - endpoints.outcomes
        curvature = probability * (1.0 - probability)

        gradient = l2 * scores
        gradient += np.bincount(endpoints.first, weights=residual, minlength=size)
        gradient -= np.bincount(endpoints.second, weights=residual, minlength=size)
        hessian = _penalized_hessian(endpoints, curvature, size, l2)

        step = np.linalg.solve(hessian, gradient)
        scores -= step
        scores -= scores.mean()
        if float(np.max(np.abs(step))) < tolerance:
            converged = True
            break

    if not diagnostics:
        return _ComponentFit(scores, converged, iterations, None, None)

    difference = scores[endpoints.first] - scores[endpoints.second]
    probability = _sigmoid_array(difference)
    final_hessian = _penalized_hessian(endpoints, probability * (1.0 - probability), size, l2)
    objective = 0.5 * l2 * float(scores @ scores) + float(
        np.sum(np.logaddexp(0.0, difference) - endpoints.outcomes * difference)
    )
    standard_errors = np.sqrt(np.diag(np.linalg.inv(final_hessian)))
    return _ComponentFit(scores, converged, iterations, standard_errors, objective)


def _fit_scores_by_component(
    cell_ids: tuple[CellId, ...],
    comparisons: tuple[Comparison, ...],
    l2: float,
    components: tuple[tuple[CellId, ...], ...],
    *,
    diagnostics: bool = True,
) -> _ComponentFit:
    """Fit every connected component separately and assemble one score vector.

    Scores are identified only up to a shift inside a component, so a joint fit
    would invent an ordering between components. It would also build a dense
    ``len(cell_ids) ** 2`` Hessian for a matrix that is block diagonal, which at
    the pilot's cell count is several gigabytes per solve.

    Every block is small, so the BLAS calls underneath spend far longer
    synchronizing threads than doing the arithmetic. Threads are pinned for the
    duration and restored on the way out.
    """
    index = {cell_id: position for position, cell_id in enumerate(cell_ids)}
    membership = {
        cell_id: number for number, component in enumerate(components) for cell_id in component
    }
    grouped: list[list[Comparison]] = [[] for _ in components]
    for comparison in comparisons:
        grouped[membership[comparison.first]].append(comparison)

    scores = np.zeros(len(cell_ids), dtype=np.float64)
    standard_errors = np.zeros(len(cell_ids), dtype=np.float64) if diagnostics else None
    converged = True
    iterations = 0
    objective = 0.0 if diagnostics else None
    with threadpool_limits(limits=1, user_api="blas"):
        for component, component_comparisons in zip(components, grouped, strict=True):
            block = _fit_scores(
                component, tuple(component_comparisons), l2, diagnostics=diagnostics
            )
            for position, cell_id in enumerate(component):
                scores[index[cell_id]] = block.scores[position]
            if standard_errors is not None and block.standard_errors is not None:
                for position, cell_id in enumerate(component):
                    standard_errors[index[cell_id]] = block.standard_errors[position]
            if objective is not None and block.objective is not None:
                objective += block.objective
            converged = converged and block.converged
            iterations = max(iterations, block.iterations)
    return _ComponentFit(scores, converged, iterations, standard_errors, objective)


def _bootstrap_intervals(
    cell_ids: tuple[CellId, ...],
    comparisons: tuple[Comparison, ...],
    l2: float,
    samples: int,
    seed: int,
    components: tuple[tuple[CellId, ...], ...],
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
        # Only the scores are read back, so the standard errors and objective
        # that a full fit would compute are skipped for every resample.
        estimates[sample_index] = _fit_scores_by_component(
            cell_ids, resampled, l2, components, diagnostics=False
        ).scores
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
    """Fit an L2-penalized within-component ordering with bootstrap intervals."""
    comparisons = build_comparisons(observations)
    cells = {observation.cell_id: _cell(observation) for observation in observations}
    cell_ids = tuple(sorted(cells))
    components = _connected_components(cell_ids, comparisons)
    fitted = _fit_scores_by_component(cell_ids, comparisons, l2, components)
    scores = fitted.scores
    standard_errors = fitted.standard_errors
    if standard_errors is None or fitted.objective is None:  # pragma: no cover
        raise RuntimeError("a diagnostic fit must return standard errors and an objective")
    intervals = _bootstrap_intervals(
        cell_ids, comparisons, l2, bootstrap_samples, seed, components
    )
    completions = Counter(
        observation.cell_id for observation in observations if observation.completed
    )
    counts = Counter(observation.cell_id for observation in observations)
    index = {cell_id: position for position, cell_id in enumerate(cell_ids)}
    ranking = tuple(
        RankedCell(
            rank,
            component_index,
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
        for component_index, component in enumerate(components)
        for rank, cell_id in enumerate(
            sorted(component, key=lambda cell_id: (-scores[index[cell_id]], str(cell_id))),
            start=1,
        )
    )
    return BradleyTerryFit(
        ranking,
        fitted.converged,
        fitted.iterations,
        fitted.objective,
        len(comparisons),
        sum(comparison.outcome == 0.5 for comparison in comparisons),
        components,
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


def _stratum_tables(
    observations: tuple[Observation, ...],
    memberships: dict[CellId, PairMembership],
) -> tuple[TableRow, ...]:
    """Summarize completion for the coordinates that are constant within a trial."""
    rows: list[TableRow] = []
    for stratum in _STRATA:
        groups: dict[str, list[Observation]] = defaultdict(list)
        for observation in observations:
            value = _stratum_value(observation, memberships[observation.cell_id], stratum)
            groups[value].append(observation)
        for value, group in sorted(groups.items()):
            count, successes, rate, low, high = _rate_row(group)
            rows.append(
                {
                    "stratum": stratum,
                    "value": value,
                    "cells": len({row.cell_id for row in group}),
                    "observations": count,
                    "completions": successes,
                    "completion_rate": rate,
                    "wilson_low": low,
                    "wilson_high": high,
                }
            )
    return tuple(rows)


@beartype
def build_pair_exclusivity(
    observations: tuple[Observation, ...],
    memberships: dict[CellId, PairMembership],
) -> tuple[TableRow, ...]:
    """Count how often each pair produced one, both, or neither completion.

    Both-completed means the pair was not exclusive in practice, which is a bank
    defect. Neither-completed means the trial was too hard. The Bradley-Terry
    tie count merges the two, so they are reported apart here.
    """
    by_trial: dict[str, list[Observation]] = defaultdict(list)
    for observation in observations:
        by_trial[str(observation.trial_id)].append(observation)

    tallies: dict[str, Counter[str]] = defaultdict(Counter)
    seen: dict[str, PairMembership] = {}
    for rows in by_trial.values():
        membership = memberships[rows[0].cell_id]
        pair_id = str(membership.pair.pair_id)
        seen[pair_id] = membership
        completed = sum(row.completed for row in rows)
        if completed == 1:
            outcome = "exactly_one"
        elif completed == len(rows):
            outcome = "both_completed"
        else:
            outcome = "neither_completed"
        tallies[pair_id][outcome] += 1
        tallies[pair_id]["trials"] += 1

    table: list[TableRow] = []
    for pair_id in sorted(tallies):
        counts = tallies[pair_id]
        trials = counts["trials"]
        membership = seen[pair_id]
        table.append(
            {
                "pair": pair_id,
                "skill": str(membership.pair.skill),
                "conflict": str(membership.pair.conflict),
                "trials": trials,
                "exactly_one": counts["exactly_one"],
                "both_completed": counts["both_completed"],
                "neither_completed": counts["neither_completed"],
                "exactly_one_rate": counts["exactly_one"] / trials,
                "both_completed_rate": counts["both_completed"] / trials,
                "neither_completed_rate": counts["neither_completed"] / trials,
            }
        )
    return tuple(table)


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


def _components_match_pair_assistant(
    observations: tuple[Observation, ...],
    fit: BradleyTerryFit,
    memberships: dict[CellId, PairMembership],
) -> bool:
    """Return whether every component is exactly one (pair, assistant) block."""
    assistant_by_cell = {
        observation.cell_id: str(observation.assistant) for observation in observations
    }
    blocks = [
        {
            (str(memberships[cell_id].pair.pair_id), assistant_by_cell[cell_id])
            for cell_id in component
        }
        for component in fit.connected_components
    ]
    if not all(len(block) == 1 for block in blocks):
        return False
    return len({next(iter(block)) for block in blocks}) == len(blocks)


def _diagnostics(
    observations: tuple[Observation, ...],
    fit: BradleyTerryFit,
    regularization: tuple[TableRow, ...],
    memberships: dict[CellId, PairMembership],
    exclusivity: tuple[TableRow, ...],
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
        # One component per (pair, assistant) is the expected shape of a
        # bipartite bank, not a defect. This is the check that replaces
        # `comparison_graph_connected` as a pass or fail signal.
        "components_match_pair_assistant": _components_match_pair_assistant(
            observations, fit, memberships
        ),
        "both_completed_trials": sum(_as_int(row["both_completed"]) for row in exclusivity),
        "neither_completed_trials": sum(
            _as_int(row["neither_completed"]) for row in exclusivity
        ),
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
    pairs: tuple[InstructionPair, ...],
    l2: float,
    *,
    bootstrap_samples: int,
    seed: int,
) -> AnalysisBundle:
    """Run ranking, axis, stratum, exclusivity, order, and sensitivity analyses."""
    validate_observations(observations)
    memberships = pair_memberships(observations, pairs)
    fit = fit_bradley_terry(
        observations,
        l2,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    axis_summary, axis_comparisons = _axis_tables(observations, fit)
    stratum_summary = _stratum_tables(observations, memberships)
    exclusivity = build_pair_exclusivity(observations, memberships)
    position, cell_position, axis_position, order_sensitivity = _position_tables(observations)
    regularization = _regularization_table(observations, l2)
    diagnostics = _diagnostics(observations, fit, regularization, memberships, exclusivity)
    return AnalysisBundle(
        fit,
        axis_summary,
        axis_comparisons,
        stratum_summary,
        exclusivity,
        position,
        cell_position,
        axis_position,
        order_sensitivity,
        regularization,
        diagnostics,
    )
