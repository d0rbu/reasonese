from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from reasonese.config import ExperimentConfig, SimulationConfig
from reasonese.design import build_trials
from reasonese.io import (
    read_outcomes,
    read_responses,
    read_trials,
    write_json,
    write_jsonl,
)
from reasonese.schemas import ResponseRecord, ScoredOutcome, Trial
from reasonese.scoring import score_responses
from reasonese.simulation import simulate_responses
from reasonese.types import parse_probability


def test_jsonl_round_trips_all_record_types(
    tmp_path: Path,
    make_trial: Callable[..., Trial],
    make_response: Callable[..., ResponseRecord],
    make_outcome: Callable[..., ScoredOutcome],
) -> None:
    trial_path = tmp_path / "nested/trials.jsonl"
    response_path = tmp_path / "responses.jsonl"
    outcome_path = tmp_path / "outcomes.jsonl"
    trial = make_trial()
    response = make_response()
    outcome = make_outcome()

    write_jsonl(trial_path, [trial])
    write_jsonl(response_path, [response])
    write_jsonl(outcome_path, [outcome])

    assert read_trials(trial_path) == [trial]
    assert read_responses(response_path) == [response]
    assert read_outcomes(outcome_path) == [outcome]


def test_json_writer_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    write_json(path, {"z": 1, "a": 2})
    assert path.read_text(encoding="utf-8") == '{\n  "a": 2,\n  "z": 1\n}\n'


def test_jsonl_writer_preserves_existing_file_on_serialization_error(tmp_path: Path) -> None:
    class BrokenRecord:
        def to_dict(self) -> dict[str, Any]:
            raise RuntimeError("boom")

    path = tmp_path / "records.jsonl"
    path.write_text("keep\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="boom"):
        write_jsonl(path, [BrokenRecord()])
    assert path.read_text(encoding="utf-8") == "keep\n"
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize("body", ["not json\n", "[]\n", '{"schema_version":1}\n'])
def test_jsonl_reader_reports_invalid_line(tmp_path: Path, body: str) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.jsonl:1"):
        read_trials(path)


def test_jsonl_reader_skips_blank_lines(
    tmp_path: Path, make_trial: Callable[..., Trial]
) -> None:
    path = tmp_path / "trials.jsonl"
    write_jsonl(path, [make_trial()])
    path.write_text("\n" + path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert len(read_trials(path)) == 1


def test_jsonl_reader_wraps_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="could not read"):
        read_trials(tmp_path / "missing.jsonl")


def test_simulation_is_deterministic_and_order_independent(
    experiment_config: ExperimentConfig,
) -> None:
    trials = build_trials(experiment_config)
    config = SimulationConfig(
        model_id="synthetic_test",
        seed=9,
        invalid_rate=parse_probability(0.0),
        first_position_bias=0.2,
        strengths={"plain": 0.0, "compact": 0.5, "authority": 1.0},
    )
    forward = simulate_responses(trials, config)
    reverse = simulate_responses(list(reversed(trials)), config)
    assert forward == simulate_responses(trials, config)
    assert {response.trial_id: response for response in forward} == {
        response.trial_id: response for response in reverse
    }
    assert {response.source for response in forward} == {"synthetic"}
    assert all(response.response_text in {"KITE", "MOSS"} for response in forward)


def test_simulation_can_emit_invalid_and_requires_strengths(
    make_trial: Callable[..., Trial],
) -> None:
    trial = make_trial()
    invalid_config = SimulationConfig(
        "synthetic_test",
        0,
        parse_probability(1.0),
        0.0,
        {"plain": 0.0, "compact": 0.0},
    )
    assert simulate_responses([trial], invalid_config)[0].response_text == "INVALID"

    missing_config = SimulationConfig(
        "synthetic_test", 0, parse_probability(0), 0.0, {"plain": 0.0}
    )
    with pytest.raises(ValueError, match="missing strengths"):
        simulate_responses([trial], missing_config)


def test_score_responses_handles_both_winners_and_invalid(
    make_trial: Callable[..., Trial], make_response: Callable[..., ResponseRecord]
) -> None:
    trials = [
        make_trial(trial_id="first"),
        make_trial(trial_id="second"),
        make_trial(trial_id="invalid"),
    ]
    responses = [
        make_response(trial_id="first", response_text="  KITE\n"),
        make_response(trial_id="second", response_text="MOSS"),
        make_response(trial_id="invalid", response_text="KITE because it is better"),
    ]
    outcomes = score_responses(trials, responses)
    assert [(item.status, item.winner) for item in outcomes] == [
        ("decisive", "plain"),
        ("decisive", "compact"),
        ("invalid", None),
    ]
    assert outcomes[0].matched_target == "KITE"
    assert outcomes[2].loser is None


@pytest.mark.parametrize("duplicate_kind", ["design", "response"])
def test_score_responses_rejects_duplicate_ids(
    make_trial: Callable[..., Trial],
    make_response: Callable[..., ResponseRecord],
    duplicate_kind: str,
) -> None:
    trials = [make_trial()]
    responses = [make_response()]
    if duplicate_kind == "design":
        trials.append(make_trial())
    else:
        responses.append(make_response())
    with pytest.raises(ValueError, match=f"duplicate {duplicate_kind}"):
        score_responses(trials, responses)


def test_score_responses_rejects_missing_and_unknown_ids(
    make_trial: Callable[..., Trial], make_response: Callable[..., ResponseRecord]
) -> None:
    trials = [make_trial(trial_id="expected")]
    responses = [make_response(trial_id="unknown")]
    with pytest.raises(ValueError, match="missing=.*expected.*unknown=.*unknown"):
        score_responses(trials, responses)


def test_written_json_is_valid(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    write_json(path, {"value": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 1}
