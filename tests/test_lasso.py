"""Tests for the feature lasso over within-trial comparisons.

Solver tests check the optimality conditions of the penalized objective
directly, so they do not depend on a reference implementation. Feature tests
use fixtures small enough to check every column by hand, and seeded synthetic
studies check that planted effects are recovered end to end.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path
from random import Random

import numpy as np
import pytest
from threadpoolctl import threadpool_info

import reasonese.lasso as lasso
from reasonese.analysis import pair_memberships
from reasonese.analyze import _lasso_lines
from reasonese.analyze import main as analyze
from reasonese.axes import (
    Assistant,
    Author,
    Channel,
    Framing,
    author_framings,
)
from reasonese.instructions import InstructionPair, load_instruction_pairs
from reasonese.judging import TraceFingerprint
from reasonese.lasso import (
    FeatureLasso,
    entry_index,
    fit_feature_lasso,
    fit_feature_lassos,
    lasso_diagnostics,
    lasso_tables,
    selected_features,
)
from reasonese.observations import Observation, cell_id, write_observations
from reasonese.planning import PromptSpec
from reasonese.study import Cell, PositiveInteger, TrialId

BANK = Path("configs/instruction_pairs.yaml")


@cache
def _pairs() -> tuple[InstructionPair, ...]:
    return load_instruction_pairs(BANK)


# --------------------------------------------------------------------------
# Synthetic studies drawn from the lasso's own model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Planted:
    """Effects on the lasso's own contrasts; everything else is zero."""

    reasonese_normal: float = 0.0
    readme: float = 0.0
    nemotron_author: float = 0.0
    reasonese_readme: float = 0.0
    reasonese_readme_nemotron: float = 0.0


def _strength(spec: PromptSpec, planted: _Planted) -> float:
    value = 0.0
    if spec.framing is Framing.REASONESE_NORMAL:
        value += planted.reasonese_normal
    if spec.channel is Channel.README:
        value += planted.readme
    if spec.author is Author.NEMOTRON_3_5_LIGHTNING:
        value += planted.nemotron_author
    if spec.framing is Framing.REASONESE_NORMAL and spec.channel is Channel.README:
        value += planted.reasonese_readme
        if spec.author is Author.NEMOTRON_3_5_LIGHTNING:
            value += planted.reasonese_readme_nemotron
    return value


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _observation(
    trial_id: TrialId,
    spec: PromptSpec,
    assistant: Assistant,
    position: int,
    completed: bool,
) -> Observation:
    return Observation(
        trial_id,
        cell_id(Cell(spec, assistant)),
        spec,
        assistant,
        PositiveInteger.parse(1),
        PositiveInteger.parse(1),
        PositiveInteger.parse(position),
        completed,
        TraceFingerprint.parse(hashlib.sha256(trial_id.encode()).hexdigest()),
        None,
        None,
    )


def _synthetic_observations(
    *,
    trials: int,
    authors: tuple[Author, ...],
    assistants: tuple[Assistant, ...],
    pairs: tuple[InstructionPair, ...],
    framings: tuple[Framing, ...],
    channels: tuple[Channel, ...],
    planted: _Planted,
    offsets: dict[tuple[str, str], float],
    seed: int,
) -> tuple[Observation, ...]:
    """Draw trials whose winner follows the lasso's model with planted effects.

    Every trial completes exactly one instruction, so there are no ties, and
    the pair's first side carries the planted block offset.
    """
    random = Random(seed)
    rows: list[Observation] = []
    for number in range(trials):
        pair = random.choice(pairs)
        assistant = random.choice(assistants)
        while True:
            specs = []
            for instruction in pair.instructions:
                author = random.choice(authors)
                specs.append(
                    PromptSpec(
                        instruction,
                        random.choice(
                            tuple(
                                framing
                                for framing in framings
                                if framing in author_framings(author)
                            )
                        ),
                        random.choice(channels),
                        author,
                    )
                )
            if any(spec.channel is Channel.USER for spec in specs):
                break
        if random.random() < 0.5:
            specs.reverse()
        strengths = []
        for spec in specs:
            strength = _strength(spec, planted)
            if spec.instruction == pair.first:
                strength += offsets[(str(pair.pair_id), str(assistant))]
            strengths.append(strength)
        first_wins = random.random() < _sigmoid(strengths[0] - strengths[1])
        trial_id = TrialId.parse(f"synthetic-{number:06d}")
        rows.append(_observation(trial_id, specs[0], assistant, 1, first_wins))
        rows.append(_observation(trial_id, specs[1], assistant, 2, not first_wins))
    return tuple(rows)


_PLANTED = _Planted(
    reasonese_normal=-1.2,
    readme=-0.8,
    nemotron_author=0.7,
    reasonese_readme=1.5,
    reasonese_readme_nemotron=0.8,
)
_PLANTED_AUTHORS = (Author.GEMMA_4_31B_IT, Author.NEMOTRON_3_5_LIGHTNING)
_PLANTED_ASSISTANTS = (Assistant.GEMMA_4_31B_IT, Assistant.NEMOTRON_3_5_LIGHTNING)
_PLANTED_FRAMINGS = (Framing.NORMAL, Framing.REASONESE_NORMAL)
_PLANTED_CHANNELS = (Channel.USER, Channel.README)


def _planted_offsets(
    pairs: tuple[InstructionPair, ...], assistants: tuple[Assistant, ...]
) -> dict[tuple[str, str], float]:
    values = (0.8, -0.5, 0.3, -1.0)
    return {
        (str(pair.pair_id), str(assistant)): values[position % len(values)]
        for position, (pair, assistant) in enumerate(
            (pair, assistant) for pair in pairs for assistant in assistants
        )
    }


@cache
def _planted_study() -> tuple[Observation, ...]:
    pairs = _pairs()[:2]
    return _synthetic_observations(
        trials=3000,
        authors=_PLANTED_AUTHORS,
        assistants=_PLANTED_ASSISTANTS,
        pairs=pairs,
        framings=_PLANTED_FRAMINGS,
        channels=_PLANTED_CHANNELS,
        planted=_PLANTED,
        offsets=_planted_offsets(pairs, _PLANTED_ASSISTANTS),
        seed=11,
    )


@cache
def _planted_fit() -> FeatureLasso:
    observations = _planted_study()
    assistant = str(_PLANTED_ASSISTANTS[0])
    selected = tuple(row for row in observations if str(row.assistant) == assistant)
    return fit_feature_lasso(
        selected,
        pair_memberships(observations, _pairs()),
        1.0,
        folds=3,
        path_length=30,
        seed=0,
    )


def _random_design(seed: int, rows: int = 400, columns: int = 8, blocks: int = 3) -> lasso._Design:
    """A raw comparison design with three real effects and sparse +-1 columns."""
    generator = np.random.default_rng(seed)
    features = np.zeros((rows, columns), dtype=np.float64)
    for column in range(columns):
        mask = generator.random(rows) < generator.uniform(0.2, 0.6)
        features[mask, column] = generator.choice([-1.0, 1.0], size=int(mask.sum()))
    block_index = generator.integers(0, blocks, size=rows).astype(np.intp)
    block_sign = generator.choice([-1.0, 1.0], size=rows)
    offsets = generator.normal(0.0, 0.5, size=blocks)
    coefficients = np.zeros(columns)
    coefficients[: min(3, columns)] = (1.0, -0.8, 0.5)[:columns]
    eta = block_sign * offsets[block_index] + features @ coefficients
    outcomes = (generator.random(rows) < 1.0 / (1.0 + np.exp(-eta))).astype(np.float64)
    return lasso._Design(outcomes, block_index, block_sign, blocks, np.asfortranarray(features))


def _kkt_violation(design: lasso._Design, fit: lasso._Fit, l2: float, penalty: float) -> float:
    """Return the largest violation of the penalized objective's optimality conditions."""
    eta = lasso._linear_predictor(design, fit.offsets, fit.coefficients)
    residual = lasso._sigmoid_array(eta) - design.outcomes
    offset_gradient = (
        np.bincount(
            design.block_index, weights=design.block_sign * residual, minlength=design.block_count
        )
        + l2 * fit.offsets
    )
    gradient = design.features.T @ residual
    worst = float(np.max(np.abs(offset_gradient)))
    for column, value in enumerate(fit.coefficients):
        if value != 0.0:
            worst = max(worst, abs(float(gradient[column]) + math.copysign(penalty, value)))
        else:
            worst = max(worst, abs(float(gradient[column])) - penalty)
    return worst


def _newton_reference(design: lasso._Design, l2: float) -> tuple[np.ndarray, np.ndarray]:
    """Ridge-offset, unpenalized-coefficient logistic fit by dense Newton steps."""
    blocks = design.block_count
    columns = design.features.shape[1]
    matrix = np.zeros((design.size, blocks + columns), dtype=np.float64)
    matrix[np.arange(design.size), design.block_index] = design.block_sign
    matrix[:, blocks:] = design.features
    ridge = np.concatenate((np.full(blocks, l2), np.zeros(columns)))
    theta = np.zeros(blocks + columns)
    for _ in range(100):
        probability = lasso._sigmoid_array(matrix @ theta)
        gradient = matrix.T @ (probability - design.outcomes) + ridge * theta
        curvature = probability * (1.0 - probability)
        hessian = (matrix * curvature[:, None]).T @ matrix + np.diag(ridge)
        step = np.linalg.solve(hessian, gradient)
        theta -= step
        if float(np.max(np.abs(step))) < 1e-12:
            break
    return theta[:blocks], theta[blocks:]


# --------------------------------------------------------------------------
# Feature columns
# --------------------------------------------------------------------------


def test_candidate_columns_use_the_documented_references_and_indicators() -> None:
    pair = _pairs()[0]
    reference = PromptSpec(pair.first, Framing.NORMAL, Channel.USER, Author.GEMMA_4_31B_IT)
    treated = PromptSpec(
        pair.second,
        Framing.REASONESE_NORMAL,
        Channel.README,
        Author.NEMOTRON_3_5_LIGHTNING,
    )
    observations = (
        _observation(TrialId.parse("trial-1"), reference, Assistant.GEMMA_4_31B_IT, 1, False),
        _observation(TrialId.parse("trial-1"), treated, Assistant.GEMMA_4_31B_IT, 2, True),
    )

    candidates = lasso._candidate_columns(observations)

    assert candidates.references == {
        "framing": "normal",
        "channel": "user message",
        "author": "Gemma 4 31B",
    }
    assert candidates.names == (
        "framing[reasonese-normal]",
        "channel[README.md]",
        "author[Nemotron 3.5 Lightning]",
        "framing[reasonese-normal]:channel[README.md]",
        "framing[reasonese-normal]:author[Nemotron 3.5 Lightning]",
        "channel[README.md]:author[Nemotron 3.5 Lightning]",
        "framing[reasonese-normal]:channel[README.md]:author[Nemotron 3.5 Lightning]",
    )
    values = dict(zip(candidates.names, candidates.matrix.T, strict=True))
    groups = dict(zip(candidates.names, candidates.groups, strict=True))
    # The second row occupies every non-reference level, so every term is active.
    assert values["framing[reasonese-normal]"].tolist() == [0.0, 1.0]
    assert values["channel[README.md]"].tolist() == [0.0, 1.0]
    assert values["author[Nemotron 3.5 Lightning]"].tolist() == [0.0, 1.0]
    assert all(column.tolist() == [0.0, 1.0] for column in values.values())
    triple = "framing[reasonese-normal]:channel[README.md]:author[Nemotron 3.5 Lightning]"
    assert groups[triple] == "framing:channel:author"


def test_default_factorial_has_exactly_the_requested_feature_groups() -> None:
    pair = _pairs()[0]
    assistant = Assistant.GEMMA_4_31B_IT
    authors = (Author.GEMMA_4_31B_IT, Author.NEMOTRON_3_5_LIGHTNING)
    observations = tuple(
        _observation(
            TrialId.parse(f"factorial-{number}"),
            PromptSpec(pair.first, framing, channel, author),
            assistant,
            1,
            False,
        )
        for number, (framing, channel, author) in enumerate(
            (framing, channel, author)
            for framing in Framing
            for channel in Channel
            for author in authors
        )
    )

    candidates = lasso._candidate_columns(observations)

    assert Counter(candidates.groups) == {
        "framing": 7,
        "channel": 2,
        "author": 1,
        "framing:channel": 14,
        "framing:author": 7,
        "channel:author": 2,
        "framing:channel:author": 14,
    }
    assert len(candidates.names) == 47
    assert not any("assistant" in name or "position" in name for name in candidates.names)


def test_references_fall_back_to_the_first_present_level() -> None:
    pair = _pairs()[0]
    trial = TrialId.parse("trial-1")
    observations = (
        _observation(
            trial,
            PromptSpec(pair.first, Framing.CASUAL, Channel.USER, Author.USER),
            Assistant.INKLING,
            1,
            True,
        ),
        _observation(
            trial,
            PromptSpec(pair.second, Framing.PERSUASIVE, Channel.README, Author.USER),
            Assistant.INKLING,
            2,
            False,
        ),
    )

    candidates = lasso._candidate_columns(observations)

    assert candidates.references["framing"] == "casual"
    assert candidates.references["author"] == "user"
    assert "framing[persuasive]" in candidates.names
    assert not any(name.startswith("author[") for name in candidates.names)


def test_screening_drops_constant_columns_and_aliases_duplicates() -> None:
    generator = np.random.default_rng(0)
    base = generator.choice([-1.0, 0.0, 1.0], size=50)
    other = generator.choice([-1.0, 0.0, 1.0], size=50)
    differences = np.column_stack((base, -base, np.zeros(50), base.copy(), other))
    candidates = lasso._Candidates(
        {},
        ("base", "negated", "constant", "copy", "other"),
        ("g", "g", "g", "g", "g"),
        np.zeros((0, 5)),
    )

    features, fitted = lasso._screen(candidates, differences)

    assert fitted == (0, 4)
    by_name = {feature.name: feature for feature in features}
    assert by_name["base"].status == "fitted"
    assert by_name["negated"].status == "aliased"
    assert (by_name["negated"].alias_of, by_name["negated"].alias_sign) == ("base", -1)
    assert by_name["constant"].status == "never differs"
    assert by_name["constant"].differing_comparisons == 0
    assert (by_name["copy"].alias_of, by_name["copy"].alias_sign) == ("base", 1)
    assert by_name["other"].most_correlated_with == "base"
    assert by_name["base"].most_correlated_with == "other"
    assert by_name["base"].max_abs_correlation == pytest.approx(
        abs(float(np.corrcoef(base, other)[0, 1]))
    )
    assert by_name["negated"].max_abs_correlation is None


def test_correlations_treat_a_constant_column_as_uncorrelated() -> None:
    matrix = np.column_stack((np.ones(10), np.arange(10, dtype=np.float64)))
    correlations = lasso._abs_correlations(matrix)
    assert np.all(np.isfinite(correlations))
    assert correlations.tolist() == [[0.0, 0.0], [0.0, 0.0]]


def test_assembly_orients_every_comparison_and_block() -> None:
    observations = _planted_study()
    memberships = pair_memberships(observations, _pairs())
    assistant = str(_PLANTED_ASSISTANTS[0])
    selected = tuple(row for row in observations if str(row.assistant) == assistant)

    assembled = lasso._assemble(selected, memberships)

    design = assembled.design
    assert assembled.assistant == assistant
    assert design.size == len(selected) // 2
    assert design.block_count == 2
    assert sorted(block.pair_id for block in assembled.blocks) == sorted(
        str(pair.pair_id) for pair in _pairs()[:2]
    )
    assert sum(block.comparisons for block in assembled.blocks) == design.size
    assert set(design.block_sign.tolist()) == {-1.0, 1.0}
    assert set(design.outcomes.tolist()) == {0.0, 1.0}
    assert "first_position" not in assembled.fitted
    assert not any("assistant" in feature for feature in assembled.fitted)
    assert design.features.flags.f_contiguous
    assert assembled.group_count == len(set(assembled.groups.tolist())) <= design.size


def test_one_lasso_rejects_multiple_evaluation_assistants() -> None:
    observations = _planted_study()
    with pytest.raises(ValueError, match="exactly one evaluation assistant"):
        lasso._assemble(observations, pair_memberships(observations, _pairs()))


# --------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ratio", [0.5, 0.1, 0.01])
def test_solution_satisfies_the_optimality_conditions(ratio: float) -> None:
    design = _random_design(1)
    null = lasso._null_fit(design, 1.0)
    penalty = ratio * lasso._lambda_max(design, null)

    fit = lasso._fit_penalty(design, null.offsets, null.coefficients, 1.0, penalty)

    assert fit.converged
    assert _kkt_violation(design, fit, 1.0, penalty) < 1e-6
    assert int(np.count_nonzero(fit.coefficients)) >= 1


def test_penalties_at_or_above_lambda_max_keep_every_feature_at_zero() -> None:
    design = _random_design(2)
    null = lasso._null_fit(design, 1.0)
    assert null.converged
    assert _kkt_violation(design, null, 1.0, math.inf) < 1e-8
    lambda_max = lasso._lambda_max(design, null)
    assert lambda_max > 0.0

    above = lasso._fit_penalty(design, null.offsets, null.coefficients, 1.0, 1.05 * lambda_max)
    below = lasso._fit_penalty(design, null.offsets, null.coefficients, 1.0, 0.5 * lambda_max)

    assert above.converged
    assert not np.any(above.coefficients)
    assert above.offsets == pytest.approx(null.offsets, abs=1e-8)
    assert np.any(below.coefficients)


def test_a_vanishing_penalty_matches_an_unpenalized_newton_fit() -> None:
    design = _random_design(3, rows=2000, columns=6)
    null = lasso._null_fit(design, 1.0)
    penalty = 1e-8 * lasso._lambda_max(design, null)

    fit = lasso._fit_penalty(design, null.offsets, null.coefficients, 1.0, penalty)
    offsets, coefficients = _newton_reference(design, 1.0)

    assert fit.converged
    assert fit.offsets == pytest.approx(offsets, abs=1e-5)
    assert fit.coefficients == pytest.approx(coefficients, abs=1e-5)


def test_an_overshooting_surrogate_step_is_backtracked_to_the_same_solution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    design = _random_design(4)
    null = lasso._null_fit(design, 1.0)
    penalty = 0.3 * lasso._lambda_max(design, null)
    clean = lasso._fit_penalty(design, null.offsets, null.coefficients, 1.0, penalty)
    original = lasso._quadratic_solve
    calls: list[int] = []

    def overshoot(
        design_: lasso._Design,
        weights: np.ndarray,
        working: np.ndarray,
        coefficients: np.ndarray,
        l2: float,
        penalty_: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        offsets_, coefficients_ = original(design_, weights, working, coefficients, l2, penalty_)
        calls.append(len(calls))
        if len(calls) == 1:
            # Forty times too far along the descent direction from the start.
            return (
                null.offsets + 40.0 * (offsets_ - null.offsets),
                coefficients + 40.0 * (coefficients_ - coefficients),
            )
        return offsets_, coefficients_

    monkeypatch.setattr(lasso, "_quadratic_solve", overshoot)
    fit = lasso._fit_penalty(design, null.offsets, null.coefficients, 1.0, penalty)

    assert fit.converged
    assert len(calls) == fit.iterations
    assert fit.offsets == pytest.approx(clean.offsets, abs=1e-6)
    assert fit.coefficients == pytest.approx(clean.coefficients, abs=1e-6)


def test_a_step_that_never_descends_stops_at_the_current_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    design = _random_design(5)
    null = lasso._null_fit(design, 1.0)
    penalty = 0.3 * lasso._lambda_max(design, null)
    original = lasso._quadratic_solve

    def astray(
        design_: lasso._Design,
        weights: np.ndarray,
        working: np.ndarray,
        coefficients: np.ndarray,
        l2: float,
        penalty_: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        offsets_, coefficients_ = original(design_, weights, working, coefficients, l2, penalty_)
        return offsets_ + 1e4, coefficients_

    monkeypatch.setattr(lasso, "_quadratic_solve", astray)
    fit = lasso._fit_penalty(design, null.offsets, null.coefficients, 1.0, penalty)

    assert fit.converged is False
    assert fit.iterations == 1
    assert fit.offsets == pytest.approx(null.offsets)
    assert not np.any(fit.coefficients)


def test_the_warm_started_path_is_monotone_in_penalty_and_loss() -> None:
    design = _random_design(6, rows=800)
    null = lasso._null_fit(design, 1.0)
    lambda_max = lasso._lambda_max(design, null)
    lambdas = [lambda_max * ratio for ratio in (1.0, 0.5, 0.2, 0.1, 0.05, 0.01)]

    fits = lasso._fit_path(design, 1.0, lambdas, null)

    assert not np.any(fits[0].coefficients)
    assert fits[0].offsets == pytest.approx(null.offsets, abs=1e-8)
    assert all(fit.converged for fit in fits)
    losses = [lasso._mean_loss(design, fit) for fit in fits]
    assert losses == sorted(losses, reverse=True)
    for penalty, fit in zip(lambdas[1:], fits[1:], strict=True):
        assert _kkt_violation(design, fit, 1.0, penalty) < 1e-6


def test_fit_pins_blas_threads_and_restores_them(monkeypatch: pytest.MonkeyPatch) -> None:
    ambient = [info["num_threads"] for info in threadpool_info() if info["user_api"] == "blas"]
    if not ambient or max(ambient) == 1:
        pytest.skip("BLAS already runs single-threaded here, so pinning cannot be observed")
    before = [(info["user_api"], info["num_threads"]) for info in threadpool_info()]
    seen: list[list[int]] = []
    original = lasso._fit_penalty

    def spy(
        design: lasso._Design,
        offsets: np.ndarray,
        coefficients: np.ndarray,
        l2: float,
        penalty: float,
    ) -> lasso._Fit:
        seen.append(
            [info["num_threads"] for info in threadpool_info() if info["user_api"] == "blas"]
        )
        return original(design, offsets, coefficients, l2, penalty)

    monkeypatch.setattr(lasso, "_fit_penalty", spy)
    observations = _planted_study()
    assistant = str(_PLANTED_ASSISTANTS[0])
    selected = tuple(row for row in observations if str(row.assistant) == assistant)
    fit_feature_lasso(
        selected,
        pair_memberships(observations, _pairs()),
        1.0,
        folds=0,
        path_length=3,
        seed=0,
    )
    monkeypatch.undo()

    assert seen, "the fit never reached the solver"
    for limits in seen:
        assert limits and all(count == 1 for count in limits), (
            f"BLAS threads were not pinned during the fit: {limits}"
        )
    after = [(info["user_api"], info["num_threads"]) for info in threadpool_info()]
    assert after == before, "thread limits leaked out of the fit"


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


def test_planted_effects_are_selected_with_their_signs() -> None:
    result = _planted_fit()

    assert all(result.converged)
    assert result.design_rank == len(result.fitted)
    assert result.cross_validation is not None
    assert result.selected is not None
    assert result.selected == result.cross_validation.index_1se
    chosen = selected_features(result)
    coefficients = dict(zip(result.fitted, result.coefficients[result.selected], strict=True))
    assert {
        "framing[reasonese-normal]",
        "channel[README.md]",
        "author[Nemotron 3.5 Lightning]",
        "framing[reasonese-normal]:channel[README.md]",
        "framing[reasonese-normal]:channel[README.md]:author[Nemotron 3.5 Lightning]",
    } <= set(chosen)
    assert coefficients["framing[reasonese-normal]"] < 0.0
    assert coefficients["channel[README.md]"] < 0.0
    assert coefficients["author[Nemotron 3.5 Lightning]"] > 0.0
    assert coefficients["framing[reasonese-normal]:channel[README.md]"] > 0.0
    triple = "framing[reasonese-normal]:channel[README.md]:author[Nemotron 3.5 Lightning]"
    assert coefficients[triple] > 0.0
    assert len(chosen) <= 10

    planted = _planted_offsets(_pairs()[:2], _PLANTED_ASSISTANTS)
    for row in lasso_tables(result).blocks:
        assert row["offset_selected"] == pytest.approx(
            planted[(str(row["pair"]), result.assistant)], abs=0.5
        )


def test_each_evaluation_assistant_gets_an_independent_fit() -> None:
    observations = _planted_study()
    memberships = pair_memberships(observations, _pairs())

    results = fit_feature_lassos(observations, memberships, 1.0, folds=0, path_length=5, seed=0)

    assert {result.assistant for result in results} == {
        str(assistant) for assistant in _PLANTED_ASSISTANTS
    }
    assert sum(result.comparisons for result in results) == len(observations) // 2
    for result in results:
        selected = tuple(row for row in observations if str(row.assistant) == result.assistant)
        assert result == fit_feature_lasso(
            selected, memberships, 1.0, folds=0, path_length=5, seed=0
        )


def test_cross_validation_is_seeded_and_orders_its_two_choices() -> None:
    result = _planted_fit()
    observations = _planted_study()
    selected = tuple(row for row in observations if str(row.assistant) == result.assistant)
    again = fit_feature_lasso(
        selected, pair_memberships(observations, _pairs()), 1.0, folds=3, path_length=30, seed=0
    )
    assert again == result

    validation = result.cross_validation
    assert validation is not None
    assert validation.folds == 3
    assert len(validation.mean_loss) == len(result.lambdas) == 30
    assert all(error >= 0.0 for error in validation.standard_error)
    assert validation.index_1se <= validation.index_min
    assert validation.mean_loss[validation.index_min] < validation.mean_loss[0]
    assert validation.mean_loss[validation.index_1se] <= (
        validation.mean_loss[validation.index_min] + validation.standard_error[validation.index_min]
    )


def test_tables_and_diagnostics_agree_with_the_path() -> None:
    result = _planted_fit()
    tables = lasso_tables(result)
    diagnostics = lasso_diagnostics(result)

    assert len(tables.path) == len(result.lambdas)
    assert tables.path[0]["nonzero"] == 0
    assert tables.path[0]["lambda_ratio"] == pytest.approx(1.0)
    assert tables.path[-1]["lambda_ratio"] == pytest.approx(lasso.PATH_MIN_RATIO)
    for row in tables.path:
        index = int(str(row["lambda_index"]))
        assert row["nonzero"] == sum(value != 0.0 for value in result.coefficients[index])
    assert len(tables.coefficients) == len(result.lambdas) * len(result.fitted)

    fitted_rows = [row for row in tables.features if row["status"] == "fitted"]
    for row in fitted_rows:
        name = str(row["feature"])
        column = result.fitted.index(name)
        entry = entry_index(result, name)
        assert row["entry_index"] == entry
        if entry is None:
            assert all(values[column] == 0.0 for values in result.coefficients)
            assert row["entry_lambda"] is None
        else:
            assert result.coefficients[entry][column] != 0.0
            assert all(values[column] == 0.0 for values in result.coefficients[:entry])
            assert row["entry_lambda"] == pytest.approx(result.lambdas[entry])
    selected = {
        str(row["feature"]) for row in fitted_rows if row["coefficient_selected"] not in (None, 0.0)
    }
    assert selected == set(selected_features(result))

    assert len(tables.blocks) == 2
    for rows in (tables.path, tables.coefficients, tables.features, tables.blocks):
        assert {row["assistant"] for row in rows} == {result.assistant}
    for row in tables.blocks:
        assert row["first_side_win_probability"] == pytest.approx(
            _sigmoid(float(str(row["offset_selected"])))
        )

    assert diagnostics["fitted_features"] == len(result.fitted)
    assert diagnostics["candidate_features"] == len(result.features)
    assert diagnostics["design_rank"] == result.design_rank
    assert diagnostics["cell_pairs"] == result.cell_pairs
    assert 1 <= result.cell_pairs <= result.comparisons
    assert result.selected is not None
    assert diagnostics["selected"] == {
        "lambda_index": result.selected,
        "lambda": result.lambdas[result.selected],
        "nonzero": len(selected_features(result)),
        "features": list(selected_features(result)),
    }
    json.dumps(diagnostics)


def test_entry_index_rejects_an_unknown_feature() -> None:
    with pytest.raises(ValueError):
        entry_index(_planted_fit(), "unknown")


def test_all_tied_outcomes_give_no_penalty_path() -> None:
    source = _planted_study()
    assistant = str(_PLANTED_ASSISTANTS[0])
    observations = tuple(
        replace(row, completed=True) for row in source if str(row.assistant) == assistant
    )[:400]
    memberships = pair_memberships(source, _pairs())

    result = fit_feature_lasso(observations, memberships, 1.0, folds=5, path_length=10, seed=0)

    assert result.lambda_max == 0.0
    assert result.lambdas == ()
    assert result.selected is None
    assert result.cross_validation is None
    assert all(offset == pytest.approx(0.0) for offset in result.null_offsets)
    assert selected_features(result) == ()
    tables = lasso_tables(result)
    assert tables.path == ()
    assert tables.coefficients == ()
    assert all(row["entry_index"] is None for row in tables.features)
    assert all(row["first_side_win_probability"] == pytest.approx(0.5) for row in tables.blocks)
    diagnostics = lasso_diagnostics(result)
    assert diagnostics["selected"] is None
    assert diagnostics["cross_validation"] is None
    assert diagnostics["converged"] is True


def test_invalid_settings_are_rejected() -> None:
    source = _planted_study()
    assistant = str(_PLANTED_ASSISTANTS[0])
    observations = tuple(row for row in source if str(row.assistant) == assistant)[:8]
    memberships = pair_memberships(source, _pairs())
    with pytest.raises(ValueError, match="L2 penalty"):
        fit_feature_lasso(observations, memberships, 0.0, folds=0, path_length=5, seed=0)
    with pytest.raises(ValueError, match="folds must be zero or at least two"):
        fit_feature_lasso(observations, memberships, 1.0, folds=1, path_length=5, seed=0)
    with pytest.raises(ValueError, match="path length"):
        fit_feature_lasso(observations, memberships, 1.0, folds=0, path_length=1, seed=0)
    with pytest.raises(ValueError, match="cannot exceed the number of distinct cell pairs"):
        fit_feature_lasso(observations, memberships, 1.0, folds=5, path_length=5, seed=0)


def test_analysis_cli_can_skip_cross_validation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    observations = _planted_study()
    path = tmp_path / "observations.jsonl"
    write_observations(path, observations)
    output = tmp_path / "analysis"

    assert (
        analyze(
            [
                "--observations",
                str(path),
                "--pairs",
                str(BANK),
                "--output",
                str(output),
                "--bootstrap-samples",
                "0",
                "--lasso-folds",
                "0",
                "--lasso-path-length",
                "8",
            ]
        )
        == 0
    )

    summary = json.loads(capsys.readouterr().out)
    assert all(count >= 4 for count in summary["lasso_selected_features"].values())
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "## Feature lasso" in report
    assert "No cross-validation was run" in report
    assert "framing-by-channel-by-author interaction" in report
    assert "Position is excluded" in report
    for name in (
        "lasso_path.csv",
        "lasso_coefficients.csv",
        "lasso_features.csv",
        "lasso_blocks.csv",
    ):
        assert (output / name).is_file()
    diagnostics = json.loads((output / "diagnostics.json").read_text(encoding="utf-8"))
    assistant_diagnostics = diagnostics["feature_lasso"]["assistants"]
    assert set(assistant_diagnostics) == {str(assistant) for assistant in _PLANTED_ASSISTANTS}
    assert all(item["cross_validation"] is None for item in assistant_diagnostics.values())
    assert all(item["selected"]["lambda_index"] == 7 for item in assistant_diagnostics.values())


def test_a_design_without_feature_columns_has_no_penalty_to_relax() -> None:
    design = _random_design(7, columns=0)
    null = lasso._null_fit(design, 1.0)
    assert null.converged
    assert null.coefficients.size == 0
    assert lasso._lambda_max(design, null) == 0.0
    assert _kkt_violation(design, null, 1.0, math.inf) < 1e-8


def test_the_iteration_cap_reports_a_fit_that_did_not_converge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    design = _random_design(8)
    null = lasso._null_fit(design, 1.0)
    penalty = 0.3 * lasso._lambda_max(design, null)
    monkeypatch.setattr(lasso, "_MAX_OUTER_ITERATIONS", 1)

    fit = lasso._fit_penalty(design, null.offsets, null.coefficients, 1.0, penalty)

    assert fit.converged is False
    assert fit.iterations == 1
    assert np.any(fit.coefficients)


def test_report_lines_flag_a_path_point_that_did_not_converge() -> None:
    result = _planted_fit()
    unconverged = replace(result, converged=(False,) * len(result.lambdas))
    assert not any("did not converge" in line for line in _lasso_lines(result))
    assert any("did not converge" in line for line in _lasso_lines(unconverged))


def test_analysis_cli_reports_an_empty_path_when_every_trial_ties(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _planted_study()
    assistant = str(_PLANTED_ASSISTANTS[0])
    observations = tuple(
        replace(row, completed=True) for row in source if str(row.assistant) == assistant
    )[:400]
    path = tmp_path / "observations.jsonl"
    write_observations(path, observations)
    output = tmp_path / "analysis"

    assert (
        analyze(
            [
                "--observations",
                str(path),
                "--pairs",
                str(BANK),
                "--output",
                str(output),
                "--bootstrap-samples",
                "0",
            ]
        )
        == 0
    )

    summary = json.loads(capsys.readouterr().out)
    assert summary["lasso_selected_features"] == {assistant: None}
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "No penalty path was fitted" in report
    assert not (output / "lasso_path.csv").exists()
    assert not (output / "lasso_coefficients.csv").exists()
    assert (output / "lasso_features.csv").is_file()
    assert (output / "lasso_blocks.csv").is_file()


def _mirrored_pairings(count: int) -> tuple[Observation, ...]:
    """``count`` distinct cell pairs, each run in both orders, as the real design does."""
    pair = _pairs()[0]
    assistant = Assistant.GEMMA_4_31B_IT
    second = PromptSpec(pair.second, Framing.NORMAL, Channel.USER, Author.USER)
    rows: list[Observation] = []
    for index, framing in enumerate(list(Framing)[:count]):
        first = PromptSpec(pair.first, framing, Channel.USER, Author.GEMMA_4_31B_IT)
        for order, specs in enumerate(((first, second), (second, first))):
            trial = TrialId.parse(f"pairing-{index}-order-{order}")
            rows.append(_observation(trial, specs[0], assistant, 1, index % 2 == 0))
            rows.append(_observation(trial, specs[1], assistant, 2, index % 2 == 1))
    return tuple(rows)


def test_folds_keep_both_orderings_of_a_cell_pair_together() -> None:
    observations = _mirrored_pairings(6)
    assembled = lasso._assemble(observations, pair_memberships(observations, _pairs()))

    assert assembled.design.size == 12
    assert assembled.group_count == 6
    groups = assembled.groups.tolist()
    # Comparisons are built in trial order: the two orderings of one pairing are adjacent.
    assert groups[0::2] == groups[1::2]
    assert len(set(groups)) == 6

    assignment = lasso._fold_assignment(assembled.groups, assembled.group_count, 3, seed=4)
    assert assignment.tolist()[0::2] == assignment.tolist()[1::2]
    assert set(assignment.tolist()) == {0, 1, 2}
    again = lasso._fold_assignment(assembled.groups, assembled.group_count, 3, seed=4)
    assert again.tolist() == assignment.tolist()


def test_cross_validation_scales_both_penalties_to_each_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    design = _random_design(9, rows=300)
    null = lasso._null_fit(design, 1.0)
    lambdas = [lasso._lambda_max(design, null) * ratio for ratio in (1.0, 0.3, 0.1)]
    seen: list[tuple[int, float, list[float]]] = []
    original = lasso._fit_path

    def record(
        training: lasso._Design, l2: float, penalties: list[float], start: lasso._Fit
    ) -> list[lasso._Fit]:
        seen.append((training.size, l2, list(penalties)))
        return original(training, l2, penalties, start)

    monkeypatch.setattr(lasso, "_fit_path", record)
    groups = np.arange(design.size, dtype=np.intp)
    lasso._cross_validate(design, groups, design.size, 1.0, lambdas, 3, 0)

    assert len(seen) == 3
    assert sum(size for size, _, _ in seen) == 2 * design.size
    for size, scaled_l2, penalties in seen:
        assert scaled_l2 == pytest.approx(size / design.size)
        assert penalties == pytest.approx([penalty * size / design.size for penalty in lambdas])
