from __future__ import annotations

import json
from pathlib import Path

import pytest

from reasonese.cli import main


def test_axes_command_prints_direct_values(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["axes"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["channel"] == ["system prompt", "user message", "README.md"]
    assert output["author"][1] == "Qwen3.8 Flash"


def test_plan_command_writes_ninety_specs_per_instruction(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "specs.jsonl"
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
    assert summary["specs_per_instruction"] == 90
    assert summary["specs"] == 180
    assert len(output.read_text().splitlines()) == 180


def test_cli_reports_missing_input(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "plan",
                "--instructions",
                str(tmp_path / "missing.toml"),
                "--output",
                str(tmp_path / "specs.jsonl"),
            ]
        )


def test_cli_requires_a_command() -> None:
    with pytest.raises(SystemExit, match="2"):
        main([])
