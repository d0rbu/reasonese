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


def test_plan_writes_ninety_specs_for_each_side_of_every_pair(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "specs.jsonl"
    assert (
        plan(
            [
                "--pairs",
                "configs/instruction_pairs.yaml",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    summary = json.loads(capsys.readouterr().out)
    assert summary["specs_per_instruction"] == 90
    assert summary["instruction_pairs"] == 24
    assert summary["instructions"] == 48
    assert summary["authors"] == 5
    assert summary["specs"] == 48 * 90
    assert len(output.read_text().splitlines()) == 48 * 90


def test_plan_filters_by_author(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "specs.jsonl"
    assert (
        plan(
            [
                "--pairs",
                "configs/instruction_pairs.yaml",
                "--output",
                str(output),
                "--author",
                "Inkling",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["authors"] == 1
    assert summary["specs"] == 48 * 18
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert {row["author"] for row in rows} == {"Inkling"}


def test_plan_rejects_repeated_authors(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="2"):
        plan(
            [
                "--pairs",
                "configs/instruction_pairs.yaml",
                "--output",
                str(tmp_path / "specs.jsonl"),
                "--author",
                "Inkling",
                "--author",
                "Inkling",
            ]
        )


def test_plan_reports_missing_input(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="2"):
        plan(
            [
                "--pairs",
                str(tmp_path / "missing.yaml"),
                "--output",
                str(tmp_path / "specs.jsonl"),
            ]
        )


def test_plan_requires_its_arguments() -> None:
    with pytest.raises(SystemExit, match="2"):
        plan([])
