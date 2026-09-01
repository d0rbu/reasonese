from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from reasonese.analysis import (
    analyze_observations,
    build_comparisons,
    fit_bradley_terry,
    validate_observations,
)
from reasonese.analyze import main as analyze
from reasonese.analyze import write_analysis
from reasonese.axes import Assistant, Author, Channel, Framing, Instruction
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


def _spec(
    text: str,
    framing: Framing,
    channel: Channel,
    author: Author,
) -> PromptSpec:
    return PromptSpec(Instruction.parse(text), framing, channel, author)


def _synthetic_observations() -> tuple[Observation, ...]:
    strong = _spec("Always complete A.", Framing.NORMAL, Channel.SYSTEM, Author.INKLING)
    order_sensitive = _spec(
        "Complete B when it is second.",
        Framing.REASONESE_NORMAL,
        Channel.USER,
        Author.INKLING_SMALL,
    )
    weak = _spec("Never complete C.", Framing.CASUAL, Channel.USER, Author.QWEN3_8_FLASH)
    observations: list[Observation] = []
    for inputs in ((strong, order_sensitive), (strong, weak), (order_sensitive, weak)):
        study = make_study(inputs, Assistant.QWEN3_8_2_4T, 2)
        for trial in build_trials(study):
            fingerprint = TraceFingerprint.parse(
                hashlib.sha256(str(trial.trial_id).encode()).hexdigest()
            )
            for position, spec in enumerate(trial.matchup.inputs, start=1):
                completed = spec == strong or (spec == order_sensitive and position == 2)
                observations.append(
                    Observation(
                        trial.trial_id,
                        cell_id(Cell(spec, study.assistant)),
                        spec,
                        study.assistant,
                        trial.permutation,
                        trial.rollout,
                        PositiveInteger.parse(position),
                        completed,
                        fingerprint,
                        f"assistant-{trial.trial_id}",
                        f"judge-{trial.trial_id}-{position}",
                    )
                )
    return tuple(observations)


def _all_equal_observations(completed: bool) -> tuple[Observation, ...]:
    observations = _synthetic_observations()
    return tuple(replace(observation, completed=completed) for observation in observations)


def test_observation_jsonl_round_trip(tmp_path: Path) -> None:
    observations = _synthetic_observations()
    path = tmp_path / "observations.jsonl"
    write_observations(path, observations)

    assert load_observations(path) == observations
    assert observation_from_dict(observation_to_dict(observations[0])) == observations[0]


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda row: [], "must be a mapping"),
        (lambda row: {"trial_id": row["trial_id"]}, "invalid fields"),
        (lambda row: {**row, "completed": 1}, "must be a boolean"),
        (lambda row: {**row, "position": True}, "must be integers"),
        (lambda row: {**row, "assistant_response_id": 1}, "strings or null"),
        (lambda row: {**row, "cell_id": "0" * 16}, "does not match"),
    ],
)
def test_observation_parser_rejects_bad_rows(
    mutate: Callable[[dict[str, object]], object], error: str
) -> None:
    row = observation_to_dict(_synthetic_observations()[0])
    with pytest.raises(ValueError, match=error):
        observation_from_dict(mutate(row))


def test_observation_loader_reports_empty_and_line_errors(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n")
    with pytest.raises(ValueError, match="contains no observations"):
        load_observations(empty)

    bad = tmp_path / "bad.jsonl"
    bad.write_text("\nnot-json\n")
    with pytest.raises(ValueError, match=r"bad.jsonl:2"):
        load_observations(bad)


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


def test_pairwise_conversion_uses_half_wins_for_equal_verdicts() -> None:
    observations = _synthetic_observations()
    comparisons = build_comparisons(observations)

    assert len(comparisons) == 12
    assert {comparison.outcome for comparison in comparisons} == {0.0, 0.5, 1.0}
    assert all(comparison.first < comparison.second for comparison in comparisons)


def test_bradley_terry_recovers_total_order_and_bootstrap_intervals() -> None:
    fit = fit_bradley_terry(
        _synthetic_observations(),
        1.0,
        bootstrap_samples=20,
        seed=7,
    )

    assert fit.converged is True
    assert [str(item.cell.spec.instruction) for item in fit.ranking] == [
        "Always complete A.",
        "Complete B when it is second.",
        "Never complete C.",
    ]
    assert sum(item.score for item in fit.ranking) == pytest.approx(0.0)
    assert all(item.standard_error > 0 for item in fit.ranking)
    assert all(item.bootstrap_low is not None for item in fit.ranking)
    assert len(fit.connected_components) == 1


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


def _disconnected_observations() -> tuple[Observation, ...]:
    all_rows = _synthetic_observations()
    first_trial = all_rows[:2]
    second_source = all_rows[2:4]
    second_trial = tuple(
        replace(
            row,
            trial_id=TrialId.parse("disconnected-second"),
            spec=_spec(
                f"Disconnected {index}.",
                Framing.PERSUASIVE,
                Channel.USER,
                Author.USER,
            ),
            cell_id=cell_id(
                Cell(
                    _spec(
                        f"Disconnected {index}.",
                        Framing.PERSUASIVE,
                        Channel.USER,
                        Author.USER,
                    ),
                    row.assistant,
                )
            ),
            position=PositiveInteger.parse(index),
        )
        for index, row in enumerate(second_source, start=1)
    )
    return (*first_trial, *second_trial)


def test_disconnected_comparison_graph_is_reported() -> None:
    fit = fit_bradley_terry(_disconnected_observations(), 1.0)
    assert len(fit.connected_components) == 2


def test_analysis_reports_axes_position_effects_balance_and_sensitivity() -> None:
    observations = _synthetic_observations()
    bundle = analyze_observations(
        observations,
        1.0,
        bootstrap_samples=10,
        seed=3,
    )
    reasonese_positions = [
        row
        for row in bundle.axis_position_effects
        if row["axis"] == "framing" and row["value"] == "reasonese-normal"
    ]
    reasonese_rates = {row["position"]: row["completion_rate"] for row in reasonese_positions}
    sensitive_cell = next(
        row
        for row in bundle.order_sensitivity
        if row["kind"] == "cell" and row["instruction"] == "Complete B when it is second."
    )

    assert reasonese_rates == {1: 0.0, 2: 1.0}
    assert sensitive_cell["position_rate_range"] == 1.0
    assert bundle.diagnostics["position_balanced"] is True
    assert bundle.diagnostics["comparison_graph_connected"] is True
    assert len(bundle.regularization_sensitivity) == 9
    assert len(bundle.axis_summary) >= 5
    assert bundle.axis_comparisons


def test_analysis_flags_position_imbalance_after_a_complete_trial_is_removed() -> None:
    observations = _synthetic_observations()[2:]
    bundle = analyze_observations(
        observations,
        1.0,
        bootstrap_samples=0,
        seed=0,
    )
    assert bundle.diagnostics["position_balanced"] is False


def test_write_analysis_emits_complete_artifact_set(tmp_path: Path) -> None:
    bundle = analyze_observations(
        _synthetic_observations(),
        1.0,
        bootstrap_samples=5,
        seed=0,
    )
    output = tmp_path / "analysis"
    write_analysis(output, bundle, 1.0)

    expected = {
        "ranking.csv",
        "axis_summary.csv",
        "axis_comparisons.csv",
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
    assert len(ranking) == 3
    assert ranking[0]["instruction"] == "Always complete A."
    report = (output / "report.md").read_text()
    assert "Total cell ordering" in report
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
        "cells": 3,
        "comparison_graph_connected": True,
        "observations": 24,
        "output": str(output),
        "position_balanced": True,
        "trials": 12,
    }


def test_analysis_cli_reports_invalid_data(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(SystemExit, match="2"):
        analyze(["--observations", str(path), "--output", str(tmp_path / "out")])
