from __future__ import annotations

import json
from pathlib import Path

import pytest

from reasonese.cli import main
from reasonese.io import read_prompt_specs


def test_axes_command_prints_the_manifest(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["axes"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert len(output["framing"]) == 6
    assert len(output["channel"]) == 3
    assert len(output["author"]) == 5


def test_plan_command_writes_full_design(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "prompt_specs.jsonl"
    assert (
        main(
            [
                "plan",
                "--instructions",
                "configs/example_instructions.toml",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "authors": 5,
        "channels": 3,
        "framings": 6,
        "instructions": 2,
        "output": str(output),
        "specs": 180,
        "specs_per_instruction": 90,
    }
    assert len(read_prompt_specs(output)) == 180


def test_plan_command_reports_input_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "plan",
                "--instructions",
                str(tmp_path / "missing.toml"),
                "--output",
                str(tmp_path / "output.jsonl"),
            ]
        )
    assert "missing.toml" in capsys.readouterr().err


def test_cli_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit, match="2"):
        main([])
