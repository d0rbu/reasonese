"""Sparse feature effects behind the within-trial comparisons.

The per-component Bradley-Terry fit in :mod:`reasonese.analysis` gives every
cell its own score, so it cannot say which of a cell's coordinates carries an
effect. This module refits the same comparisons separately for each evaluation
assistant, with every cell's strength decomposed into a pair-side offset plus
a sparse sum of feature effects::

    logit P(first beats second) = sign * offset[block] + sum_j beta_j * (x_first_j - x_second_j)

The features are treatment contrasts for framing, channel, and author, all
three two-way interactions, and their three-way interaction. An L1 penalty on
``beta`` zeroes the features the comparisons do not support, so the order in
which features enter as the penalty relaxes ranks them by how strongly they
are tied to completion. The offsets keep the L2 penalty of the ranking fit,
which bounds them under separation without shrinking any feature.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np
from beartype import beartype
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from reasonese.analysis import TableRow, build_comparisons
from reasonese.axes import Author, Channel, Framing
from reasonese.instructions import PairMembership, PairSide
from reasonese.observations import CellId, Observation

# The penalty path runs from the value that zeroes every feature down to this
# fraction of it, which is far enough for every supported feature to enter.
PATH_MIN_RATIO = 1e-3
# Logistic curvature p(1 - p) underflows on the tails. glmnet uses this floor.
_WEIGHT_FLOOR = 1e-5
_INNER_TOLERANCE = 1e-10
_OUTER_TOLERANCE = 1e-8
# A gradient this close to the penalty, relative to it, counts as on the boundary.
_KKT_SLACK = 1e-9
_MAX_OUTER_ITERATIONS = 100
_MAX_SWEEPS = 1000
_MIN_STEP = 1e-4

_FRAMING_ORDER = tuple(str(value) for value in Framing)
_CHANNEL_ORDER = tuple(str(value) for value in Channel)
_AUTHOR_ORDER = tuple(str(value) for value in Author)
# The author reference is the first model author present rather than the user,
# because the user writes only three framings. With the user as reference, the
# subagent and reasonese main effects would be observed under no reference
# author, so each would equal the sum of its author interactions and the fit
# would have no unique solution.
_MODEL_AUTHORS = tuple(str(value) for value in Author if value is not Author.USER)
_INTERACTIONS = (
    ("framing", "channel"),
    ("framing", "author"),
    ("channel", "author"),
    ("framing", "channel", "author"),
)
_STATUS_FITTED = "fitted"
_STATUS_CONSTANT = "never differs"
_STATUS_ALIASED = "aliased"


@beartype
@dataclass(frozen=True, slots=True)
class LassoFeature:
    """One candidate feature column and how it was screened before the fit."""

    name: str
    group: str
    status: str
    alias_of: str | None
    alias_sign: int
    differing_comparisons: int
    max_abs_correlation: float | None
    most_correlated_with: str | None


@beartype
@dataclass(frozen=True, slots=True)
class LassoBlock:
    """One pair block whose side offset absorbs its instruction asymmetry."""

    pair_id: str
    comparisons: int


@beartype
@dataclass(frozen=True, slots=True)
class LassoCrossValidation:
    """Held-out loss along the path and the two conventional penalty choices."""

    folds: int
    mean_loss: tuple[float, ...]
    standard_error: tuple[float, ...]
    index_min: int
    index_1se: int


@beartype
@dataclass(frozen=True, slots=True)
class FeatureLasso:
    """One assistant's penalty path over the within-trial comparisons.

    ``coefficients`` and ``offsets`` hold one row per penalty in ``lambdas``;
    ``selected`` is the row reported as the fit, the one-standard-error choice
    when cross-validation ran and otherwise the last, least penalized, row. Both
    paths are empty when no fitted feature is correlated with the outcome once
    the offsets alone are fitted. ``design_rank`` below the number of fitted
    columns means the columns are linearly dependent, so the fitted
    probabilities are unique but the coefficients that produce them are not.
    ``cell_pairs`` counts the distinct unordered cell pairs compared; both
    orderings and every rollout of one pair share a cross-validation fold.
    """

    assistant: str
    comparisons: int
    cell_pairs: int
    references: dict[str, str]
    features: tuple[LassoFeature, ...]
    fitted: tuple[str, ...]
    design_rank: int
    blocks: tuple[LassoBlock, ...]
    null_offsets: tuple[float, ...]
    null_loss: float
    lambda_max: float
    lambdas: tuple[float, ...]
    coefficients: tuple[tuple[float, ...], ...]
    offsets: tuple[tuple[float, ...], ...]
    train_loss: tuple[float, ...]
    converged: tuple[bool, ...]
    iterations: tuple[int, ...]
    cross_validation: LassoCrossValidation | None
    selected: int | None


@beartype
@dataclass(frozen=True, slots=True)
class LassoTables:
    """The four tabular views of one feature lasso."""

    path: tuple[TableRow, ...]
    coefficients: tuple[TableRow, ...]
    features: tuple[TableRow, ...]
    blocks: tuple[TableRow, ...]


# --------------------------------------------------------------------------
# Design
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Design:
    """Comparison outcomes, block offsets, and feature differences as arrays."""

    outcomes: NDArray[np.float64]
    block_index: NDArray[np.intp]
    block_sign: NDArray[np.float64]
    block_count: int
    features: NDArray[np.float64]

    @property
    def size(self) -> int:
        return int(self.outcomes.shape[0])

    def rows(self, selected: NDArray[np.intp]) -> _Design:
        return _Design(
            self.outcomes[selected],
            self.block_index[selected],
            self.block_sign[selected],
            self.block_count,
            np.asfortranarray(self.features[selected]),
        )


@dataclass(frozen=True, slots=True)
class _Axis:
    """One categorical coordinate as treatment dummies against a reference level."""

    reference: str
    levels: tuple[str, ...]
    dummies: NDArray[np.float64]


def _axis(values: list[str], order: tuple[str, ...], preferred: tuple[str, ...]) -> _Axis:
    present = set(values)
    reference = next((level for level in preferred if level in present), None)
    if reference is None:
        reference = next(level for level in order if level in present)
    levels = tuple(level for level in order if level in present and level != reference)
    codes = {level: position for position, level in enumerate(levels)}
    indices = np.fromiter(
        (codes.get(value, -1) for value in values), dtype=np.intp, count=len(values)
    )
    dummies = (indices[:, None] == np.arange(len(levels))[None, :]).astype(np.float64)
    return _Axis(reference, levels, dummies)


@dataclass(frozen=True, slots=True)
class _Candidates:
    """Every candidate feature column, before screening, in documented order."""

    references: dict[str, str]
    names: tuple[str, ...]
    groups: tuple[str, ...]
    matrix: NDArray[np.float64]


def _candidate_columns(observations: tuple[Observation, ...]) -> _Candidates:
    """Build one row of candidate feature values per observation.

    Main effects come first, followed by every two-way interaction among
    framing, channel, and author and then their three-way interaction.
    """
    axes = {
        "framing": _axis(
            [str(row.spec.framing) for row in observations],
            _FRAMING_ORDER,
            (str(Framing.NORMAL),),
        ),
        "channel": _axis(
            [str(row.spec.channel) for row in observations],
            _CHANNEL_ORDER,
            (str(Channel.USER),),
        ),
        "author": _axis(
            [str(row.spec.author) for row in observations], _AUTHOR_ORDER, _MODEL_AUTHORS
        ),
    }
    names: list[str] = []
    groups: list[str] = []
    columns: list[NDArray[np.float64]] = []
    for axis_name in ("framing", "channel", "author"):
        axis = axes[axis_name]
        for position, level in enumerate(axis.levels):
            names.append(f"{axis_name}[{level}]")
            groups.append(axis_name)
            columns.append(axis.dummies[:, position])
    for interaction in _INTERACTIONS:
        interaction_axes = tuple(axes[name] for name in interaction)
        for positions in product(*(range(len(axis.levels)) for axis in interaction_axes)):
            terms = tuple(
                f"{name}[{axis.levels[position]}]"
                for name, axis, position in zip(
                    interaction, interaction_axes, positions, strict=True
                )
            )
            names.append(":".join(terms))
            groups.append(":".join(interaction))
            column = np.ones(len(observations), dtype=np.float64)
            for axis, position in zip(interaction_axes, positions, strict=True):
                column *= axis.dummies[:, position]
            columns.append(column)
    return _Candidates(
        {axis_name: axis.reference for axis_name, axis in axes.items()},
        tuple(names),
        tuple(groups),
        np.column_stack(columns),
    )


def _signature(column: NDArray[np.float64]) -> bytes:
    # Adding zero turns any negative zero into positive zero, so a column and
    # its negation hash consistently.
    return (np.ascontiguousarray(column) + 0.0).tobytes()


def _abs_correlations(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    centred = matrix - matrix.mean(axis=0)
    norms = np.sqrt(np.sum(centred * centred, axis=0))
    scale = np.where(norms > 0.0, norms, np.inf)
    correlations = np.abs((centred.T @ centred) / np.outer(scale, scale))
    np.fill_diagonal(correlations, 0.0)
    return correlations


def _screen(
    candidates: _Candidates, differences: NDArray[np.float64]
) -> tuple[tuple[LassoFeature, ...], tuple[int, ...]]:
    """Drop columns that never differ inside a trial and columns equal to an earlier one."""
    differing = np.count_nonzero(differences, axis=0)
    seen: dict[bytes, str] = {}
    statuses: list[tuple[str, str | None, int]] = []
    fitted: list[int] = []
    for column, name in enumerate(candidates.names):
        values = differences[:, column]
        if differing[column] == 0:
            statuses.append((_STATUS_CONSTANT, None, 0))
            continue
        alias = seen.get(_signature(values))
        if alias is not None:
            statuses.append((_STATUS_ALIASED, alias, 1))
            continue
        alias = seen.get(_signature(-values))
        if alias is not None:
            statuses.append((_STATUS_ALIASED, alias, -1))
            continue
        seen[_signature(values)] = name
        statuses.append((_STATUS_FITTED, None, 0))
        fitted.append(column)

    correlations = (
        _abs_correlations(differences[:, fitted]) if len(fitted) > 1 else np.zeros((0, 0))
    )
    fitted_position = {column: position for position, column in enumerate(fitted)}
    features: list[LassoFeature] = []
    for column, name in enumerate(candidates.names):
        status, alias_of, alias_sign = statuses[column]
        max_abs_correlation: float | None = None
        most_correlated_with: str | None = None
        if status == _STATUS_FITTED and correlations.size:
            position = fitted_position[column]
            partner = int(np.argmax(correlations[position]))
            max_abs_correlation = float(correlations[position, partner])
            most_correlated_with = candidates.names[fitted[partner]]
        features.append(
            LassoFeature(
                name,
                candidates.groups[column],
                status,
                alias_of,
                alias_sign,
                int(differing[column]),
                max_abs_correlation,
                most_correlated_with,
            )
        )
    return tuple(features), tuple(fitted)


@dataclass(frozen=True, slots=True)
class _Assembled:
    assistant: str
    design: _Design
    groups: NDArray[np.intp]
    group_count: int
    references: dict[str, str]
    features: tuple[LassoFeature, ...]
    fitted: tuple[str, ...]
    blocks: tuple[LassoBlock, ...]


def _assemble(
    observations: tuple[Observation, ...], memberships: dict[CellId, PairMembership]
) -> _Assembled:
    assistants = {str(row.assistant) for row in observations}
    if len(assistants) != 1:
        raise ValueError("each feature lasso must contain exactly one evaluation assistant")
    comparisons = build_comparisons(observations)
    row_index = {
        (str(row.trial_id), row.cell_id): position for position, row in enumerate(observations)
    }
    keys = [str(memberships[comparison.first].pair.pair_id) for comparison in comparisons]
    block_keys = sorted(set(keys))
    block_position = {key: position for position, key in enumerate(block_keys)}
    count = len(comparisons)
    first_rows = np.fromiter(
        (row_index[(comparison.trial_id, comparison.first)] for comparison in comparisons),
        dtype=np.intp,
        count=count,
    )
    second_rows = np.fromiter(
        (row_index[(comparison.trial_id, comparison.second)] for comparison in comparisons),
        dtype=np.intp,
        count=count,
    )
    block_index = np.fromiter((block_position[key] for key in keys), dtype=np.intp, count=count)
    block_sign = np.fromiter(
        (
            1.0 if memberships[comparison.first].side is PairSide.FIRST else -1.0
            for comparison in comparisons
        ),
        dtype=np.float64,
        count=count,
    )
    outcomes = np.fromiter(
        (comparison.outcome for comparison in comparisons), dtype=np.float64, count=count
    )
    candidates = _candidate_columns(observations)
    differences = candidates.matrix[first_rows] - candidates.matrix[second_rows]
    features, fitted = _screen(candidates, differences)
    block_sizes = Counter(keys)
    # Comparisons list their cells in one canonical order, so the cell pair is
    # the same for both orderings and every rollout of one study.
    pair_keys = [(comparison.first, comparison.second) for comparison in comparisons]
    pair_position = {key: position for position, key in enumerate(dict.fromkeys(pair_keys))}
    groups = np.fromiter((pair_position[key] for key in pair_keys), dtype=np.intp, count=count)
    return _Assembled(
        next(iter(assistants)),
        _Design(
            outcomes,
            block_index,
            block_sign,
            len(block_keys),
            np.asfortranarray(differences[:, list(fitted)]),
        ),
        groups,
        len(pair_position),
        candidates.references,
        features,
        tuple(candidates.names[column] for column in fitted),
        tuple(LassoBlock(pair_id, block_sizes[pair_id]) for pair_id in block_keys),
    )


# --------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Fit:
    offsets: NDArray[np.float64]
    coefficients: NDArray[np.float64]
    converged: bool
    iterations: int


def _sigmoid_array(values: NDArray[np.float64]) -> NDArray[np.float64]:
    exponential = np.exp(-np.abs(values))
    return np.where(values >= 0.0, 1.0, exponential) / (1.0 + exponential)


def _linear_predictor(
    design: _Design, offsets: NDArray[np.float64], coefficients: NDArray[np.float64]
) -> NDArray[np.float64]:
    eta = design.block_sign * offsets[design.block_index]
    if coefficients.size:
        eta += design.features @ coefficients
    return eta


def _loss(eta: NDArray[np.float64], outcomes: NDArray[np.float64]) -> float:
    return float(np.sum(np.logaddexp(0.0, eta) - outcomes * eta))


def _objective(
    design: _Design,
    eta: NDArray[np.float64],
    offsets: NDArray[np.float64],
    coefficients: NDArray[np.float64],
    l2: float,
    penalty: float,
) -> float:
    l1 = float(np.sum(np.abs(coefficients)))
    value = _loss(eta, design.outcomes) + 0.5 * l2 * float(offsets @ offsets)
    # The offsets-only fit uses an infinite penalty, and inf * 0.0 is nan.
    return value + (penalty * l1 if l1 > 0.0 else 0.0)


def _soft_threshold(value: float, penalty: float) -> float:
    magnitude = abs(value) - penalty
    # A gradient within rounding of the penalty is on the boundary, not past
    # it, so it stays at an explicit zero rather than a 1e-17 coefficient that
    # would count as selected.
    if magnitude <= penalty * _KKT_SLACK:
        return 0.0
    return math.copysign(magnitude, value)


def _max_abs(values: NDArray[np.float64]) -> float:
    return float(np.max(np.abs(values))) if values.size else 0.0


def _quadratic_solve(
    design: _Design,
    weights: NDArray[np.float64],
    working: NDArray[np.float64],
    coefficients: NDArray[np.float64],
    l2: float,
    penalty: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Minimise the weighted least-squares surrogate over a growing active set.

    The surrogate is quadratic, so over the active features it is solved by
    coordinate descent on a small Gram matrix rather than by repeated passes
    over every comparison. Blocks partition the rows, so the offsets have a
    closed form given the coefficients and are refreshed once per sweep. Once
    the active set settles, one pass over every column finds the features whose
    gradient violates the optimality condition; they join the active set and
    the solve repeats until none do. With an infinite penalty nothing can
    enter, which gives the offsets-only fit.
    """
    features = design.features
    block_index = design.block_index
    block_count = design.block_count
    signed_weights = weights * design.block_sign
    block_scale = np.bincount(block_index, weights=weights, minlength=block_count) + l2
    block_linear = np.bincount(block_index, weights=signed_weights * working, minlength=block_count)
    coefficients = coefficients.copy()
    active = [int(column) for column in np.flatnonzero(coefficients)]
    offsets = block_linear / block_scale
    while True:
        columns = np.array(active, dtype=np.intp)
        subset = features[:, columns]
        weighted = subset * weights[:, None]
        gram = subset.T @ weighted
        linear = weighted.T @ working
        cross = np.empty((block_count, len(active)), dtype=np.float64)
        for position in range(len(active)):
            cross[:, position] = np.bincount(
                block_index, weights=signed_weights * subset[:, position], minlength=block_count
            )
        beta = coefficients[columns]
        for _ in range(_MAX_SWEEPS):
            updated = (block_linear - cross @ beta) / block_scale
            largest = _max_abs(updated - offsets)
            offsets = updated
            cross_offsets = cross.T @ offsets
            for position in range(len(active)):
                # Every active column is nonzero on these rows, because a
                # violator has a nonzero gradient and a warm start comes from
                # the same rows, so the curvature is positive.
                curvature = gram[position, position]
                partial = linear[position] - cross_offsets[position]
                partial -= gram[position] @ beta - curvature * beta[position]
                proposed = _soft_threshold(partial, penalty) / curvature
                change = proposed - beta[position]
                if change != 0.0:
                    beta[position] = proposed
                    largest = max(largest, abs(change))
            if largest < _INNER_TOLERANCE:
                break
        coefficients[:] = 0.0
        coefficients[columns] = beta
        residual = working - _linear_predictor(design, offsets, coefficients)
        gradient = features.T @ (weights * residual)
        known = set(active)
        violators = [
            int(column)
            for column in np.flatnonzero(np.abs(gradient) > penalty * (1.0 + _KKT_SLACK))
            if int(column) not in known
        ]
        if not violators:
            return offsets, coefficients
        active.extend(violators)


def _fit_penalty(
    design: _Design,
    offsets: NDArray[np.float64],
    coefficients: NDArray[np.float64],
    l2: float,
    penalty: float,
) -> _Fit:
    """Run proximal Newton steps: solve the surrogate, then backtrack on the objective.

    The fit has converged when the surrogate's solution is the current point,
    which is checked before any line search so that rounding noise in the
    objective can never keep a converged fit iterating. A full step is then
    taken, which keeps the surrogate's exact zeros. An infinite penalty gives
    the offsets-only fit, since no feature can enter.
    """
    offsets = offsets.copy()
    coefficients = coefficients.copy()
    eta = _linear_predictor(design, offsets, coefficients)
    objective = _objective(design, eta, offsets, coefficients, l2, penalty)
    converged = False
    iterations = 0
    for iteration in range(1, _MAX_OUTER_ITERATIONS + 1):
        iterations = iteration
        probability = _sigmoid_array(eta)
        weights = np.maximum(probability * (1.0 - probability), _WEIGHT_FLOOR)
        working = eta - (probability - design.outcomes) / weights
        proposed_offsets, proposed_coefficients = _quadratic_solve(
            design, weights, working, coefficients, l2, penalty
        )
        change = max(
            _max_abs(proposed_offsets - offsets), _max_abs(proposed_coefficients - coefficients)
        )
        if change < _OUTER_TOLERANCE:
            offsets, coefficients = proposed_offsets, proposed_coefficients
            converged = True
            break
        slack = 1e-12 * max(1.0, abs(objective))
        step = 1.0
        while True:
            if step == 1.0:
                candidate_offsets, candidate_coefficients = proposed_offsets, proposed_coefficients
            else:
                candidate_offsets = offsets + step * (proposed_offsets - offsets)
                candidate_coefficients = coefficients + step * (
                    proposed_coefficients - coefficients
                )
            candidate_eta = _linear_predictor(design, candidate_offsets, candidate_coefficients)
            candidate_objective = _objective(
                design, candidate_eta, candidate_offsets, candidate_coefficients, l2, penalty
            )
            if candidate_objective <= objective + slack:
                break
            step *= 0.5
            if step < _MIN_STEP:
                # The surrogate direction descends whenever it is nonzero, so
                # no decrease down here means the point is optimal to rounding.
                return _Fit(offsets, coefficients, False, iterations)
        offsets, coefficients = candidate_offsets, candidate_coefficients
        eta, objective = candidate_eta, candidate_objective
    return _Fit(offsets, coefficients, converged, iterations)


def _null_fit(design: _Design, l2: float) -> _Fit:
    return _fit_penalty(
        design,
        np.zeros(design.block_count, dtype=np.float64),
        np.zeros(design.features.shape[1], dtype=np.float64),
        l2,
        math.inf,
    )


def _lambda_max(design: _Design, null: _Fit) -> float:
    """Return the smallest penalty at which the offsets-only fit is optimal."""
    eta = _linear_predictor(design, null.offsets, null.coefficients)
    gradient = design.features.T @ (_sigmoid_array(eta) - design.outcomes)
    return _max_abs(gradient)


def _fit_path(design: _Design, l2: float, lambdas: Sequence[float], null: _Fit) -> list[_Fit]:
    """Warm-start down the path from the offsets-only fit.

    On the full data the first penalty is ``lambda_max``, where the offsets-only
    fit is already optimal and the solve returns at once. A cross-validation
    fold can have a larger ``lambda_max`` of its own, so its first point is
    solved rather than assumed.
    """
    fits: list[_Fit] = []
    current = null
    for penalty in lambdas:
        current = _fit_penalty(design, current.offsets, current.coefficients, l2, penalty)
        fits.append(current)
    return fits


def _mean_loss(design: _Design, fit: _Fit) -> float:
    eta = _linear_predictor(design, fit.offsets, fit.coefficients)
    return _loss(eta, design.outcomes) / design.size


def _fold_assignment(
    groups: NDArray[np.intp], group_count: int, folds: int, seed: int
) -> NDArray[np.intp]:
    """Assign whole groups to folds, so one study never straddles a fold."""
    generator = np.random.default_rng(seed)
    group_fold = generator.permutation(group_count) % folds
    return group_fold[groups]


def _cross_validate(
    design: _Design,
    groups: NDArray[np.intp],
    group_count: int,
    l2: float,
    lambdas: Sequence[float],
    folds: int,
    seed: int,
) -> LassoCrossValidation:
    """Refit the path without each fold of cell pairs and score the held-out loss.

    The loss is a sum over comparisons, so penalties tuned to the full data
    would bind harder on a smaller fold. Each fold's L1 and L2 penalties are
    scaled by its share of the comparisons, which keeps the per-comparison
    penalties the same and makes the chosen L1 penalty transfer back to the
    full fit.
    """
    assignment = _fold_assignment(groups, group_count, folds, seed)
    losses = np.empty((folds, len(lambdas)), dtype=np.float64)
    for fold in range(folds):
        training = design.rows(np.flatnonzero(assignment != fold))
        held_out = design.rows(np.flatnonzero(assignment == fold))
        share = training.size / design.size
        scaled_l2 = l2 * share
        scaled = [penalty * share for penalty in lambdas]
        fits = _fit_path(training, scaled_l2, scaled, _null_fit(training, scaled_l2))
        losses[fold] = [_mean_loss(held_out, fit) for fit in fits]
    mean = losses.mean(axis=0)
    standard_error = losses.std(axis=0, ddof=1) / math.sqrt(folds)
    index_min = int(np.argmin(mean))
    threshold = mean[index_min] + standard_error[index_min]
    index_1se = int(np.flatnonzero(mean <= threshold)[0])
    return LassoCrossValidation(
        folds,
        tuple(float(value) for value in mean),
        tuple(float(value) for value in standard_error),
        index_min,
        index_1se,
    )


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------


@beartype
def fit_feature_lasso(
    observations: tuple[Observation, ...],
    memberships: dict[CellId, PairMembership],
    l2: float,
    *,
    folds: int,
    path_length: int,
    seed: int,
) -> FeatureLasso:
    """Fit one assistant's path and, with at least two folds, cross-validate it.

    ``l2`` penalizes the block offsets exactly as it penalizes the ranking's
    cell scores. Feature columns are not standardized: every column is a
    difference of indicators, so a feature that rarely differs inside a trial
    needs a larger effect to enter.
    """
    if l2 <= 0:
        raise ValueError("L2 penalty must be positive")
    if folds < 0 or folds == 1:
        raise ValueError("lasso folds must be zero or at least two")
    if path_length < 2:
        raise ValueError("lasso path length must be at least two")
    assembled = _assemble(observations, memberships)
    design = assembled.design
    if folds > assembled.group_count:
        raise ValueError("lasso folds cannot exceed the number of distinct cell pairs")

    with threadpool_limits(limits=1, user_api="blas"):
        null = _null_fit(design, l2)
        lambda_max = _lambda_max(design, null)
        lambdas: tuple[float, ...] = ()
        if lambda_max > 0.0:
            ratios = PATH_MIN_RATIO ** (np.arange(path_length) / (path_length - 1))
            lambdas = tuple(float(value) for value in lambda_max * ratios)
        fits = _fit_path(design, l2, lambdas, null) if lambdas else []
        cross_validation = (
            _cross_validate(
                design, assembled.groups, assembled.group_count, l2, lambdas, folds, seed
            )
            if lambdas and folds
            else None
        )

    selected: int | None = None
    if lambdas:
        selected = cross_validation.index_1se if cross_validation else len(lambdas) - 1
    return FeatureLasso(
        assembled.assistant,
        design.size,
        assembled.group_count,
        assembled.references,
        assembled.features,
        assembled.fitted,
        int(np.linalg.matrix_rank(design.features)) if design.features.shape[1] else 0,
        assembled.blocks,
        tuple(float(value) for value in null.offsets),
        _mean_loss(design, null),
        lambda_max,
        lambdas,
        tuple(tuple(float(value) for value in fit.coefficients) for fit in fits),
        tuple(tuple(float(value) for value in fit.offsets) for fit in fits),
        tuple(_mean_loss(design, fit) for fit in fits),
        tuple(fit.converged for fit in fits),
        tuple(fit.iterations for fit in fits),
        cross_validation,
        selected,
    )


@beartype
def fit_feature_lassos(
    observations: tuple[Observation, ...],
    memberships: dict[CellId, PairMembership],
    l2: float,
    *,
    folds: int,
    path_length: int,
    seed: int,
) -> tuple[FeatureLasso, ...]:
    """Fit an independent feature lasso for every evaluation assistant."""
    assistants = tuple(sorted({str(row.assistant) for row in observations}))
    return tuple(
        fit_feature_lasso(
            tuple(row for row in observations if str(row.assistant) == assistant),
            memberships,
            l2,
            folds=folds,
            path_length=path_length,
            seed=seed,
        )
        for assistant in assistants
    )


@beartype
def entry_index(result: FeatureLasso, feature: str) -> int | None:
    """Return the path index at which a fitted feature first becomes nonzero."""
    column = result.fitted.index(feature)
    for index, row in enumerate(result.coefficients):
        if row[column] != 0.0:
            return index
    return None


def _nonzero(row: tuple[float, ...]) -> int:
    return sum(value != 0.0 for value in row)


@beartype
def selected_features(result: FeatureLasso) -> tuple[str, ...]:
    """Return the features that are nonzero at the selected penalty, in column order."""
    if result.selected is None:
        return ()
    row = result.coefficients[result.selected]
    return tuple(name for name, value in zip(result.fitted, row, strict=True) if value != 0.0)


@beartype
def lasso_tables(result: FeatureLasso) -> LassoTables:
    """Flatten the path, coefficients, screened features, and block offsets."""
    cross_validation = result.cross_validation
    path: list[TableRow] = []
    coefficients: list[TableRow] = []
    for index, penalty in enumerate(result.lambdas):
        path.append(
            {
                "assistant": result.assistant,
                "lambda_index": index,
                "lambda": penalty,
                "lambda_ratio": penalty / result.lambda_max,
                "nonzero": _nonzero(result.coefficients[index]),
                "train_loss": result.train_loss[index],
                "cv_mean_loss": cross_validation.mean_loss[index] if cross_validation else None,
                "cv_se_loss": cross_validation.standard_error[index] if cross_validation else None,
                "converged": result.converged[index],
                "iterations": result.iterations[index],
            }
        )
        for column, name in enumerate(result.fitted):
            coefficients.append(
                {
                    "assistant": result.assistant,
                    "lambda_index": index,
                    "lambda": penalty,
                    "feature": name,
                    "coefficient": result.coefficients[index][column],
                }
            )

    features: list[TableRow] = []
    for feature in result.features:
        entry: int | None = None
        selected_value: float | None = None
        minimum_value: float | None = None
        if feature.status == _STATUS_FITTED and result.selected is not None:
            column = result.fitted.index(feature.name)
            entry = entry_index(result, feature.name)
            selected_value = result.coefficients[result.selected][column]
            if cross_validation is not None:
                minimum_value = result.coefficients[cross_validation.index_min][column]
        features.append(
            {
                "assistant": result.assistant,
                "feature": feature.name,
                "group": feature.group,
                "status": feature.status,
                "alias_of": feature.alias_of,
                "alias_sign": feature.alias_sign,
                "differing_comparisons": feature.differing_comparisons,
                "max_abs_correlation": feature.max_abs_correlation,
                "most_correlated_with": feature.most_correlated_with,
                "entry_index": entry,
                "entry_lambda": None if entry is None else result.lambdas[entry],
                "entry_lambda_ratio": (
                    None if entry is None else result.lambdas[entry] / result.lambda_max
                ),
                "coefficient_selected": selected_value,
                "coefficient_cv_min": minimum_value,
            }
        )

    offsets = result.null_offsets if result.selected is None else result.offsets[result.selected]
    blocks: list[TableRow] = []
    for position, block in enumerate(result.blocks):
        offset = offsets[position]
        blocks.append(
            {
                "assistant": result.assistant,
                "pair": block.pair_id,
                "comparisons": block.comparisons,
                "offset_selected": offset,
                "first_side_win_probability": float(_sigmoid_array(np.array([offset]))[0]),
            }
        )
    return LassoTables(tuple(path), tuple(coefficients), tuple(features), tuple(blocks))


@beartype
def lasso_diagnostics(result: FeatureLasso) -> dict[str, object]:
    """Summarize screening, the path, cross-validation, and the selected fit."""
    statuses = Counter(feature.status for feature in result.features)
    cross_validation = result.cross_validation
    selected: dict[str, object] | None = None
    if result.selected is not None:
        names = selected_features(result)
        selected = {
            "lambda_index": result.selected,
            "lambda": result.lambdas[result.selected],
            "nonzero": len(names),
            "features": list(names),
        }
    validation: dict[str, object] | None = None
    if cross_validation is not None:
        validation = {
            "folds": cross_validation.folds,
            "lambda_min": result.lambdas[cross_validation.index_min],
            "lambda_1se": result.lambdas[cross_validation.index_1se],
            "loss_min": cross_validation.mean_loss[cross_validation.index_min],
            "loss_1se": cross_validation.mean_loss[cross_validation.index_1se],
        }
    return {
        "assistant": result.assistant,
        "comparisons": result.comparisons,
        "cell_pairs": result.cell_pairs,
        "blocks": len(result.blocks),
        "candidate_features": len(result.features),
        "fitted_features": statuses[_STATUS_FITTED],
        "never_differing_features": statuses[_STATUS_CONSTANT],
        "aliased_features": statuses[_STATUS_ALIASED],
        "design_rank": result.design_rank,
        "references": dict(result.references),
        "null_loss": result.null_loss,
        "lambda_max": result.lambda_max,
        "path_length": len(result.lambdas),
        "path_min_ratio": PATH_MIN_RATIO,
        "converged": all(result.converged),
        "cross_validation": validation,
        "selected": selected,
    }
