"""Write planned datapoints."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import yaml
from beartype import beartype

from reasonese.planning import PromptSpec
from reasonese.study import Study, study_to_dict


@beartype
def write_prompt_specs(path: Path, specs: tuple[PromptSpec, ...]) -> None:
    """Write datapoints as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for spec in specs:
            handle.write(json.dumps(asdict(spec), sort_keys=True))
            handle.write("\n")


@beartype
def write_study_suite(path: Path, studies: tuple[Study, ...]) -> None:
    """Write distinct studies to one readable YAML artifact."""
    if not studies:
        raise ValueError("at least one study is required")
    if len(studies) != len(set(studies)):
        raise ValueError("studies must be distinct")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {"studies": [study_to_dict(study) for study in studies]},
            handle,
            sort_keys=False,
            allow_unicode=True,
            width=100,
        )
