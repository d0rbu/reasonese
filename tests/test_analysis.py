"""Tests for observations and the within-pair Bradley-Terry analysis.

Fixtures are built from the real instruction-pair bank, so every trial holds the
two instructions of one mutually exclusive pair exactly as collection produces
them. Conditions are deliberately few so that expected counts stay checkable by
hand; the sampler's full-scale behaviour is covered in `test_sampling.py`.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from functools import cache
from pathlib import Path

import pytest

import reasonese.analysis as analysis
import reasonese.analyze as analyze_module
from reasonese.analysis import (
    analyze_observations,
    build_comparisons,
    build_pair_exclusivity,
    fit_bradley_terry,
    pair_memberships,
    validate_observations,
)
from reasonese.analyze import main as analyze
from reasonese.analyze import write_analysis
from reasonese.axes import Assistant, Author, Channel, Framing, Instruction
from reasonese.instructions import (
    InstructionPair,
    PairMembership,
    instruction_index,
    load_instruction_pairs,
)
from reasonese.judging import TraceFingerprint
from reasonese.observations import (
    Observation,
    cell_id,
    load_observations,
    observation_from_dict,
    observation_to_dict,
    write_observations,
)
from reasonese.planning import PromptSpec
from reasonese.study import Cell, PositiveInteger, TrialId, build_trials, make_study

BANK = Path("configs/instruction_pairs.yaml")

# One order-sensitive condition per side plus a system-prompt condition, which
# is the smallest set that still exercises every channel pairing rule.
_FIRST_CONDITIONS = (
    (Framing.NORMAL, Channel.USER, Author.INKLING),
    (Framing.REASONESE_NORMAL, Channel.USER, Author.INKLING_SMALL),
    (Framing.CASUAL, Channel.SYSTEM, Author.QWEN3_8_FLASH),
)
_SECOND_CONDITIONS = (
    (Framing.NORMAL, Channel.USER, Author.INKLING),
    (Framing.PERSUASIVE, Channel.README, Author.QWEN3_8_2_4T),
)


@cache
def _pairs() -> tuple[InstructionPair, ...]:
    return load_instruction_pairs(BANK)


def _index() -> dict[str, PairMembership]:
    return {
        str(instruction): membership
        for instruction, membership in instruction_index(_pairs()).items()
    }


def _count(row: dict[str, object], key: str) -> int:
    """Read an integer table or diagnostic entry, asserting it really is one."""
    value = row[key]
    assert isinstance(value, int), f"{key} should be an integer, got {type(value).__name__}"
    return value


def _spec(
    text: str,
    framing: Framing,
    channel: Channel,
    author: Author,
) -> PromptSpec:
    return PromptSpec(Instruction.parse(text), framing, channel, author)


def _completed(spec: PromptSpec, position: int, pair: InstructionPair) -> bool:
    """Complete the first side always, except one condition that needs position 2."""
    if spec.instruction == pair.first:
        if spec.framing is Framing.REASONESE_NORMAL:
            return position == 2
        return True
    return spec.framing is Framing.PERSUASIVE


def _pair_observations(
    pair: InstructionPair,
    assistant: Assistant,
    rollouts: int = 2,
) -> tuple[Observation, ...]:
    first_specs = tuple(PromptSpec(pair.first, *condition) for condition in _FIRST_CONDITIONS)
    second_specs = tuple(
        PromptSpec(pair.second, *condition) for condition in _SECOND_CONDITIONS
    )
    rows: list[Observation] = []
    for first in first_specs:
        for second in second_specs:
            if Channel.USER not in (first.channel, second.channel):
                continue
            study = make_study((first, second), assistant, rollouts)
            for trial in build_trials(study):
                fingerprint = TraceFingerprint.parse(
                    hashlib.sha256(str(trial.trial_id).encode()).hexdigest()
                )
                for position, spec in enumerate(trial.matchup.inputs, start=1):
                    rows.append(
                        Observation(
                            trial.trial_id,
                            cell_id(Cell(spec, assistant)),
                            spec,
                            assistant,
                            trial.permutation,
                            trial.rollout,
                            PositiveInteger.parse(position),
                            _completed(spec, position, pair),
                            fingerprint,
                            f"assistant-{trial.trial_id}",
                            f"judge-{trial.trial_id}-{position}",
                        )
                    )
    return tuple(rows)


def _synthetic_observations() -> tuple[Observation, ...]:
    return _pair_observations(_pairs()[0], Assistant.QWEN3_8_2_4T)


def _two_pair_observations() -> tuple[Observation, ...]:
    return _pair_observations(_pairs()[0], Assistant.QWEN3_8_2_4T) + _pair_observations(
        _pairs()[1], Assistant.QWEN3_8_2_4T
    )


def _two_assistant_observations() -> tuple[Observation, ...]:
    return _pair_observations(_pairs()[0], Assistant.QWEN3_8_2_4T) + _pair_observations(
        _pairs()[0], Assistant.INKLING
    )


def _split_component_observations() -> tuple[Observation, ...]:
    """Keep two disjoint studies from one pair so its block splits in two."""
    rows = _synthetic_observations()
    keep = {
        (Framing.NORMAL, Framing.NORMAL),
        (Framing.REASONESE_NORMAL, Framing.PERSUASIVE),
    }
    by_trial: dict[str, list[Observation]] = {}
    for row in rows:
        by_trial.setdefault(str(row.trial_id), []).append(row)
    selected: list[Observation] = []
    for trial_rows in by_trial.values():
        framings = {row.spec.framing for row in trial_rows}
        if any(framings == set(option) for option in keep):
            selected.extend(trial_rows)
    return tuple(selected)


def _all_equal_observations(completed: bool) -> tuple[Observation, ...]:
    return tuple(
        replace(observation, completed=completed)
        for observation in _synthetic_observations()
    )


# --------------------------------------------------------------------------
# Observation serialization and validation
# --------------------------------------------------------------------------


def test_observation_jsonl_round_trip(tmp_path: Path) -> None:
    observations = _synthetic_observations()
    path = tmp_path / "observations.jsonl"
    write_observations(path, observations)

    assert load_observations(path) == observations
    assert observation_from_dict(observation_to_dict(observations[0])) == observations[0]


def test_fixture_has_the_shape_the_later_assertions_assume() -> None:
    observations = _synthetic_observations()
    # Three first-side conditions times two second-side, less the one pairing
    # with no user-message channel, times two orderings times two rollouts.
    assert len(observations) == 2 * (3 * 2 - 1) * 2 * 2
    assert len({str(row.trial_id) for row in observations}) == 20
    assert len({row.cell_id for row in observations}) == 5


def test_validation_accepts_complete_trials_and_rejects_duplicate_cells() -> None:
    observations = _synthetic_observations()
    assert validate_observations(observations) is None
    with pytest.raises(ValueError, match="only one observation per cell"):
        validate_observations((*observations, observations[0]))


@pytest.mark.parametrize(
    ("replacement", "error"),
    [
        ({"position": PositiveInteger.parse(2)}, "positions must be exactly"),
        ({"permutation": PositiveInteger.parse(2)}, "inconsistent permutation"),
        ({"rollout": PositiveInteger.parse(2)}, "inconsistent rollout"),
        ({"trace_fingerprint": TraceFingerprint.parse("f" * 64)}, "trace fingerprints"),
    ],
)
def test_validation_rejects_inconsistent_trial_metadata(
    replacement: dict[str, object], error: str
) -> None:
    first_trial = _synthetic_observations()[:2]
    changed = replace(first_trial[0], **replacement)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=error):
        validate_observations((changed, *first_trial[1:]))


def test_validation_rejects_single_cell_trials_and_cell_id_coordinate_collisions() -> None:
    row = _synthetic_observations()[0]
    with pytest.raises(ValueError, match="exactly two cells"):
        validate_observations((row,))

    different_spec = _spec("Different.", Framing.NORMAL, Channel.USER, Author.USER)
    collision = replace(row, spec=different_spec, trial_id=TrialId.parse("other-trial"))
    with pytest.raises(ValueError, match="multiple coordinate tuples"):
        validate_observations((row, collision))


def test_validation_rejects_multiple_assistants_within_one_trial() -> None:
    rows = _synthetic_observations()[:2]
    changed = replace(
        rows[1],
        assistant=Assistant.INKLING_SMALL,
        cell_id=cell_id(Cell(rows[1].spec, Assistant.INKLING_SMALL)),
    )
    with pytest.raises(ValueError, match="multiple assistant"):
        validate_observations((rows[0], changed))


# --------------------------------------------------------------------------
# Pair membership
# --------------------------------------------------------------------------


def test_pair_memberships_label_every_cell_with_its_pair_and_side() -> None:
    observations = _synthetic_observations()
    memberships = pair_memberships(observations, _pairs())
    pair = _pairs()[0]

    assert len(memberships) == 5
    for observation in observations:
        membership = memberships[observation.cell_id]
        assert membership.pair.pair_id == pair.pair_id
        expected = "first" if observation.spec.instruction == pair.first else "second"
        assert str(membership.side) == expected


def test_pair_memberships_reject_an_instruction_outside_the_bank() -> None:
    rows = _synthetic_observations()[:2]
    stray = _spec("Not in the bank.", Framing.NORMAL, Channel.USER, Author.USER)
    changed = replace(rows[0], spec=stray, cell_id=cell_id(Cell(stray, rows[0].assistant)))
    with pytest.raises(ValueError, match="absent from the pair bank"):
        pair_memberships((changed, rows[1]), _pairs())


def test_pair_memberships_reject_a_trial_that_mixes_two_pairs() -> None:
    rows = _synthetic_observations()[:2]
    other = _spec(
        str(_pairs()[1].second), Framing.NORMAL, Channel.USER, Author.INKLING
    )
    changed = replace(rows[1], spec=other, cell_id=cell_id(Cell(other, rows[1].assistant)))
    with pytest.raises(ValueError, match="different pairs"):
        pair_memberships((rows[0], changed), _pairs())


def test_pair_memberships_reject_a_trial_using_one_side_twice() -> None:
    rows = _synthetic_observations()[:2]
    same_side = _spec(
        str(_pairs()[0].first), Framing.SUBAGENT, Channel.USER, Author.INKLING
    )
    duplicate = replace(
        rows[1], spec=same_side, cell_id=cell_id(Cell(same_side, rows[1].assistant))
    )
    memberships_source = (rows[0], duplicate)
    first_side = {
        str(pair_memberships((row,), _pairs())[row.cell_id].side)
        for row in memberships_source
    }
    assert first_side == {"first"}
    with pytest.raises(ValueError, match="repeats one side"):
        pair_memberships(memberships_source, _pairs())


def test_instruction_index_requires_a_non_empty_bank() -> None:
    with pytest.raises(ValueError, match="at least one instruction pair"):
        instruction_index(())


# --------------------------------------------------------------------------
# Comparisons and the Bradley-Terry fit
# --------------------------------------------------------------------------


def test_pairwise_conversion_uses_half_wins_for_equal_verdicts() -> None:
    comparisons = build_comparisons(_synthetic_observations())

    assert len(comparisons) == 20
    assert {comparison.outcome for comparison in comparisons} == {0.0, 0.5, 1.0}
    assert all(comparison.first < comparison.second for comparison in comparisons)


def test_bradley_terry_orders_within_one_component() -> None:
    fit = fit_bradley_terry(
        _synthetic_observations(),
        1.0,
        bootstrap_samples=20,
        seed=7,
    )
    pair = _pairs()[0]

    assert fit.converged is True
    assert len(fit.connected_components) == 1
    assert [item.component_index for item in fit.ranking] == [0] * 5
    assert [item.rank for item in fit.ranking] == [1, 2, 3, 4, 5]
    assert sum(item.score for item in fit.ranking) == pytest.approx(0.0)
    assert all(item.standard_error > 0 for item in fit.ranking)
    assert all(item.bootstrap_low is not None for item in fit.ranking)

    # The never-completed second-side condition must finish last.
    last = fit.ranking[-1]
    assert last.cell.spec.instruction == pair.second
    assert last.cell.spec.framing is Framing.NORMAL
    assert last.completions == 0


def test_every_component_is_centred_on_zero_independently() -> None:
    fit = fit_bradley_terry(_two_pair_observations(), 1.0)
    assert len(fit.connected_components) == 2

    by_component: dict[int, list[float]] = {}
    for item in fit.ranking:
        by_component.setdefault(item.component_index, []).append(item.score)
    assert len(by_component) == 2
    for scores in by_component.values():
        assert sum(scores) == pytest.approx(0.0)
        assert len(scores) == 5


def test_ranks_restart_inside_every_component() -> None:
    fit = fit_bradley_terry(_two_assistant_observations(), 1.0)
    assert len(fit.connected_components) == 2
    ranks: dict[int, list[int]] = {}
    for item in fit.ranking:
        ranks.setdefault(item.component_index, []).append(item.rank)
    assert ranks == {0: [1, 2, 3, 4, 5], 1: [1, 2, 3, 4, 5]}


def test_all_true_and_all_false_trials_become_tied_rankings() -> None:
    for completed in (True, False):
        fit = fit_bradley_terry(_all_equal_observations(completed), 1.0)
        assert all(item.score == pytest.approx(0.0) for item in fit.ranking)
        assert fit.tie_count == fit.comparison_count


def test_bradley_terry_rejects_nonpositive_penalty_and_negative_bootstrap() -> None:
    observations = _synthetic_observations()
    with pytest.raises(ValueError, match="L2 penalty"):
        fit_bradley_terry(observations, 0.0)
    with pytest.raises(ValueError, match="bootstrap samples"):
        fit_bradley_terry(observations, 1.0, bootstrap_samples=-1)


# --------------------------------------------------------------------------
# Exclusivity
# --------------------------------------------------------------------------


def test_pair_exclusivity_separates_both_completed_from_neither_completed() -> None:
    observations = _synthetic_observations()
    memberships = pair_memberships(observations, _pairs())
    table = build_pair_exclusivity(observations, memberships)
    pair = _pairs()[0]

    assert len(table) == 1
    row = table[0]
    assert row["pair"] == str(pair.pair_id)
    assert row["skill"] == str(pair.skill)
    assert row["conflict"] == str(pair.conflict)
    assert row["trials"] == 20
    assert row["exactly_one"] == 12
    assert row["both_completed"] == 6
    assert row["neither_completed"] == 2
    assert _count(row, "exactly_one") + _count(row, "both_completed") + _count(
        row, "neither_completed"
    ) == 20
    assert row["both_completed_rate"] == pytest.approx(6 / 20)
    assert row["neither_completed_rate"] == pytest.approx(2 / 20)

    # Bradley-Terry merges both outcomes into a tie, which is exactly why the
    # split is reported separately.
    fit = fit_bradley_terry(observations, 1.0)
    assert fit.tie_count == 8


def test_pair_exclusivity_reports_one_row_per_pair() -> None:
    observations = _two_pair_observations()
    memberships = pair_memberships(observations, _pairs())
    table = build_pair_exclusivity(observations, memberships)

    assert [row["pair"] for row in table] == sorted(
        str(pair.pair_id) for pair in _pairs()[:2]
    )
    assert sum(_count(row, "trials") for row in table) == 40


# --------------------------------------------------------------------------
# The analysis bundle
# --------------------------------------------------------------------------


def test_analysis_reports_axes_strata_position_effects_and_diagnostics() -> None:
    observations = _synthetic_observations()
    bundle = analyze_observations(
        observations,
        _pairs(),
        1.0,
        bootstrap_samples=10,
        seed=3,
    )

    # Instruction and assistant are gone from the Bradley-Terry axes because
    # neither varies inside a trial.
    assert {row["axis"] for row in bundle.axis_summary} == {
        "framing",
        "channel",
        "author",
    }
    assert {row["stratum"] for row in bundle.stratum_summary} == {
        "assistant",
        "skill",
        "conflict",
        "pair",
    }
    assert all("mean_bt_score" not in row for row in bundle.stratum_summary)

    reasonese_positions = [
        row
        for row in bundle.axis_position_effects
        if row["axis"] == "framing" and row["value"] == "reasonese-normal"
    ]
    reasonese_rates = {row["position"]: row["completion_rate"] for row in reasonese_positions}
    assert reasonese_rates == {1: 0.0, 2: 1.0}

    assert bundle.diagnostics["position_balanced"] is True
    assert bundle.diagnostics["components_match_pair_assistant"] is True
    assert bundle.diagnostics["both_completed_trials"] == 6
    assert bundle.diagnostics["neither_completed_trials"] == 2
    assert len(bundle.regularization_sensitivity) == 15
    assert bundle.axis_comparisons
    assert bundle.pair_exclusivity


def test_analysis_reports_one_component_per_pair_and_assistant() -> None:
    for observations in (_two_pair_observations(), _two_assistant_observations()):
        bundle = analyze_observations(
            observations,
            _pairs(),
            1.0,
            bootstrap_samples=0,
            seed=0,
        )
        assert bundle.diagnostics["comparison_graph_connected"] is False
        assert bundle.diagnostics["components_match_pair_assistant"] is True
        assert len(bundle.fit.connected_components) == 2


def test_analysis_flags_a_pair_block_that_split_into_two_components() -> None:
    bundle = analyze_observations(
        _split_component_observations(),
        _pairs(),
        1.0,
        bootstrap_samples=0,
        seed=0,
    )
    assert len(bundle.fit.connected_components) == 2
    # Two components inside one (pair, assistant) block is a real defect, unlike
    # the expected one component per block.
    assert bundle.diagnostics["components_match_pair_assistant"] is False


def test_analysis_flags_position_imbalance_after_a_complete_trial_is_removed() -> None:
    bundle = analyze_observations(
        _synthetic_observations()[2:],
        _pairs(),
        1.0,
        bootstrap_samples=0,
        seed=0,
    )
    assert bundle.diagnostics["position_balanced"] is False


def test_write_analysis_emits_complete_artifact_set(tmp_path: Path) -> None:
    bundle = analyze_observations(
        _synthetic_observations(),
        _pairs(),
        1.0,
        bootstrap_samples=5,
        seed=0,
    )
    output = tmp_path / "analysis"
    write_analysis(output, bundle, 1.0, _index())

    expected = {
        "ranking.csv",
        "axis_summary.csv",
        "axis_comparisons.csv",
        "stratum_summary.csv",
        "pair_exclusivity.csv",
        "position_summary.csv",
        "cell_position_effects.csv",
        "axis_position_effects.csv",
        "order_sensitivity.csv",
        "regularization_sensitivity.csv",
        "diagnostics.json",
        "report.md",
    }
    assert {path.name for path in output.iterdir()} == expected

    with (output / "ranking.csv").open() as handle:
        ranking = list(csv.DictReader(handle))
    assert len(ranking) == 5
    pair = _pairs()[0]
    assert {row["pair"] for row in ranking} == {str(pair.pair_id)}
    assert {row["side"] for row in ranking} == {"first", "second"}
    assert {row["component"] for row in ranking} == {"0"}
    assert [row["rank"] for row in ranking] == ["1", "2", "3", "4", "5"]

    with (output / "pair_exclusivity.csv").open() as handle:
        exclusivity = list(csv.DictReader(handle))
    assert exclusivity[0]["both_completed"] == "6"

    report = (output / "report.md").read_text()
    assert "Within-component cell ordering" in report
    assert "Pair exclusivity" in report
    assert "## Strata" in report
    assert str(pair.pair_id) in report
    assert "reasonese-normal" in report


def test_analysis_cli_combines_inputs_and_prints_diagnostics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    observations = _synthetic_observations()
    midpoint = len(observations) // 2
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_observations(first, observations[:midpoint])
    write_observations(second, observations[midpoint:])
    output = tmp_path / "analysis"

    assert (
        analyze(
            [
                "--observations",
                str(first),
                str(second),
                "--pairs",
                str(BANK),
                "--output",
                str(output),
                "--bootstrap-samples",
                "5",
                "--seed",
                "2",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "both_completed_trials": 6,
        "cells": 5,
        "components": 1,
        "components_match_pair_assistant": True,
        "neither_completed_trials": 2,
        "observations": 40,
        "output": str(output),
        "position_balanced": True,
        "trials": 20,
    }


def test_axis_and_stratum_lookups_reject_unknown_names() -> None:
    observations = _synthetic_observations()
    membership = pair_memberships(observations, _pairs())[observations[0].cell_id]
    with pytest.raises(ValueError, match="unknown axis"):
        analysis._axis_value(observations[0], "instruction")
    with pytest.raises(ValueError, match="unknown stratum"):
        analysis._stratum_value(observations[0], membership, "framing")


def test_validation_requires_at_least_one_observation() -> None:
    with pytest.raises(ValueError, match="at least one observation"):
        validate_observations(())


def test_table_value_coercion_rejects_non_numeric_entries() -> None:
    assert analysis._as_float(2) == 2.0
    assert analysis._as_int(2) == 2
    with pytest.raises(TypeError, match="numeric table value"):
        analysis._as_float("2")
    with pytest.raises(TypeError, match="integer table value"):
        analysis._as_int(2.0)


def test_component_check_fails_when_one_component_spans_two_pairs() -> None:
    observations = _two_pair_observations()
    memberships = pair_memberships(observations, _pairs())
    fit = fit_bradley_terry(observations, 1.0)
    assert analysis._components_match_pair_assistant(observations, fit, memberships) is True

    merged = replace(
        fit,
        connected_components=(
            tuple(sorted(cell for component in fit.connected_components for cell in component)),
        ),
    )
    assert analysis._components_match_pair_assistant(observations, merged, memberships) is False

    duplicated = replace(
        fit,
        connected_components=(fit.connected_components[0], fit.connected_components[0]),
    )
    assert (
        analysis._components_match_pair_assistant(observations, duplicated, memberships) is False
    )


def test_report_helpers_handle_empty_tables_and_bad_values(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    analyze_module._write_csv(path, ())
    assert not path.exists()

    assert analyze_module._format_float(None) == "NA"
    assert analyze_module._format_float(0.5) == "0.5000"
    with pytest.raises(TypeError, match="numeric report value"):
        analyze_module._format_float("0.5")


def test_analysis_cli_reports_invalid_data(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(SystemExit, match="2"):
        analyze(
            [
                "--observations",
                str(path),
                "--pairs",
                str(BANK),
                "--output",
                str(tmp_path / "out"),
            ]
        )


def test_analysis_cli_reports_a_missing_pair_bank(tmp_path: Path) -> None:
    path = tmp_path / "observations.jsonl"
    write_observations(path, _synthetic_observations())
    with pytest.raises(SystemExit, match="2"):
        analyze(
            [
                "--observations",
                str(path),
                "--pairs",
                str(tmp_path / "missing.yaml"),
                "--output",
                str(tmp_path / "out"),
            ]
        )
