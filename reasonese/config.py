"""Load matchups and studies from YAML.

Base instructions are no longer loaded from a free-form list. They arrive in
mutually exclusive pairs through `reasonese.instructions.load_instruction_pairs`,
because arbitrary instructions cannot be paired into a study.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from beartype import beartype

from reasonese.matchup import Matchup, matchup_from_dict
from reasonese.study import Study, study_from_dict


@beartype
def load_matchup(path: Path) -> Matchup:
    """Load one matchup from YAML."""
    with path.open(encoding="utf-8") as handle:
        return matchup_from_dict(yaml.safe_load(handle))


@beartype
def load_study(path: Path) -> Study:
    """Load and validate one permutation-balanced study from YAML."""
    with path.open(encoding="utf-8") as handle:
        return study_from_dict(yaml.safe_load(handle))


@beartype
def load_study_suite(path: Path) -> tuple[Study, ...]:
    """Load a non-empty sequence of distinct studies from one YAML file."""
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or set(raw) != {"studies"}:
        raise ValueError("study suite must contain exactly one 'studies' field")
    raw_studies = raw["studies"]
    if not isinstance(raw_studies, list) or not raw_studies:
        raise ValueError("study suite must contain at least one study")
    studies = tuple(study_from_dict(item) for item in raw_studies)
    if len(studies) != len(set(studies)):
        raise ValueError("study suite entries must be distinct")
    return studies
