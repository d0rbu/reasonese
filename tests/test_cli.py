from __future__ import annotations

import json
from pathlib import Path

import pytest

from reasonese.cli import main
from reasonese.io import read_outcomes, read_responses, read_trials

EXPERIMENT_CONFIG = '''
schema_version = 1

[experiment]
name = "cli_test"
seed = 3
repetitions = 1
system_prompt = "Return one code only."
user_preamble = "Process both directives."
response_code_pairs = [["KITE", "MOSS"]]

[[conditions]]
id = "plain"
family = "representation"
template = "Return {target}."
description = "Plain."

[[conditions]]
id = "compact"
family = "representation"
template = "GO_{target}"
description = "Compact."

[[conditions]]
id = "authority"
family = "authority"
template = "The CEO requires {target}."
description = "Authority."
'''

SIMULATION_CONFIG = '''
schema_version = 1

[simulation]
model_id = "cli_synthetic"
seed = 4
invalid_rate = 0.0
first_position_bias = 0.0

[strengths]
plain = 0.0
compact = 0.5
authority = 1.0
'''


def test_cli_runs_complete_offline_pipeline(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    experiment = tmp_path / "experiment.toml"
    simulation = tmp_path / "simulation.toml"
    design = tmp_path / "artifacts/design.jsonl"
    responses = tmp_path / "artifacts/responses.jsonl"
    outcomes = tmp_path / "artifacts/outcomes.jsonl"
    ranking = tmp_path / "artifacts/ranking.json"
    experiment.write_text(EXPERIMENT_CONFIG, encoding="utf-8")
    simulation.write_text(SIMULATION_CONFIG, encoding="utf-8")

    assert main(["design", "--config", str(experiment), "--output", str(design)]) == 0
    design_summary = json.loads(capsys.readouterr().out)
    assert design_summary["trials"] == 12
    assert len(read_trials(design)) == 12

    assert (
        main(
            [
                "simulate",
                "--design",
                str(design),
                "--config",
                str(simulation),
                "--output",
                str(responses),
            ]
        )
        == 0
    )
    simulation_summary = json.loads(capsys.readouterr().out)
    assert simulation_summary["source"] == "synthetic"
    assert len(read_responses(responses)) == 12

    assert (
        main(
            [
                "score",
                "--design",
                str(design),
                "--responses",
                str(responses),
                "--output",
                str(outcomes),
            ]
        )
        == 0
    )
    score_summary = json.loads(capsys.readouterr().out)
    assert score_summary == {
        "decisive": 12,
        "invalid": 0,
        "output": str(outcomes),
    }
    assert len(read_outcomes(outcomes)) == 12

    assert (
        main(
            [
                "fit",
                "--outcomes",
                str(outcomes),
                "--output",
                str(ranking),
                "--reference",
                "plain",
                "--ridge",
                "0.5",
            ]
        )
        == 0
    )
    fit_summary = json.loads(capsys.readouterr().out)
    report = json.loads(ranking.read_text(encoding="utf-8"))
    assert fit_summary["converged"] is True
    assert report["reference_condition"] == "plain"
    assert report["source"] == "synthetic"
    assert len(report["ranking"]) == 3


def test_cli_turns_config_errors_into_usage_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as error:
        main(
            [
                "design",
                "--config",
                str(tmp_path / "missing.toml"),
                "--output",
                str(tmp_path / "design.jsonl"),
            ]
        )
    assert error.value.code == 2
    assert "could not load TOML config" in capsys.readouterr().err


def test_cli_requires_a_command() -> None:
    with pytest.raises(SystemExit) as error:
        main([])
    assert error.value.code == 2
