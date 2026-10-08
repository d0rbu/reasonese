"""Bind a collection output root to one complete scenario selection."""

from __future__ import annotations

import json
import os
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile

from beartype import beartype

from reasonese.scenarios import ScenarioLibrary, require_scenario_selection
from reasonese.study_cache import SqliteStudyCache

_FILENAME = "scenario_selection.json"


def _selection(scenarios: ScenarioLibrary | None) -> dict[str, object]:
    if scenarios is None:
        return {"scenarios": None}
    return {
        "scenarios": {
            str(pair_id): {
                "instructions": sorted(
                    str(instruction)
                    for instruction, membership in scenarios.memberships.items()
                    if membership.pair.pair_id == pair_id
                ),
                "source": scenario.source,
                "adaptation": scenario.adaptation,
                "messages": [asdict(message) for message in scenario.messages],
            }
            for pair_id, scenario in scenarios.scenarios.items()
        }
    }


def _require_selection(path: Path, selection: dict[str, object]) -> None:
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid scenario selection in {path}") from error
    if saved != selection:
        raise ValueError(
            f"{path.parent} has a different scenario selection or an edited scenario; "
            "use a fresh output directory"
        )


def _validate_legacy_root(
    root: Path, scenarios: ScenarioLibrary | None, selection: dict[str, object]
) -> None:
    # Include studies outside the current invocation, in either cache layout.
    for path in root.rglob(_FILENAME):
        _require_selection(path, selection)
    saved_trials: set[str] = set()
    for path in root.rglob("collection.sqlite3"):
        with closing(SqliteStudyCache(path).iter_traces_readonly()) as traces:
            for trial_id, trace in traces:
                require_scenario_selection(scenarios, trace.setup)
                saved_trials.add(str(trial_id))
    for path in root.rglob("observations.jsonl"):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                observation = json.loads(line)
                if (
                    not isinstance(observation, dict)
                    or not isinstance(observation.get("trial_id"), str)
                    or observation["trial_id"] not in saved_trials
                ):
                    raise ValueError(
                        f"cannot verify scenario selection for existing observations in {path}; "
                        "use a fresh output directory"
                    )


@beartype
def bind_scenario_selection(root: Path, scenarios: ScenarioLibrary | None) -> None:
    """Check the whole output root before new studies, cache writes, or provider work."""
    root = root.resolve()
    selection = _selection(scenarios)
    # A standalone study inside a previously bound suite still belongs to that suite.
    for parent in root.parents:
        ancestor = parent / _FILENAME
        if ancestor.exists():
            _require_selection(ancestor, selection)
    path = root / _FILENAME
    if path.exists():
        _require_selection(path, selection)
        return
    _validate_legacy_root(root, scenarios, selection)
    root.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", dir=root, encoding="utf-8") as handle:
        json.dump(selection, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        try:
            # Publish a complete record atomically, without replacing another collector's.
            os.link(handle.name, path)
        except FileExistsError:
            _require_selection(path, selection)
