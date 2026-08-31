from __future__ import annotations

import json
from pathlib import Path

import pytest

from reasonese.plan import main as plan
from reasonese.show_axes import main as show_axes


def test_show_axes_prints_direct_values(capsys: pytest.CaptureFixture[str]) -> None:
    assert show_axes() is None
    output = json.loads(capsys.readouterr().out)
    assert output["channel"] == ["system prompt", "user message", "README.md"]
    assert output["author"][1] == "Qwen3.8 Flash"


def test_plan_writes_ninety_specs_per_instruction(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "specs.jsonl"
    assert (
        plan(
            [
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


def test_plan_reports_missing_input(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="2"):
        plan(
            [
                "--instructions",
                str(tmp_path / "missing.toml"),
                "--output",
                str(tmp_path / "specs.jsonl"),
            ]
        )


def test_plan_requires_its_arguments() -> None:
    with pytest.raises(SystemExit, match="2"):
        plan([])
