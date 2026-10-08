"""A collection root keeps one scenario selection across different studies."""

from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier

import pytest

from reasonese.axes import Framing
from reasonese.cache import YamlMessageCache
from reasonese.collect_data import CollectionTask, collect_studies, collect_study
from reasonese.message_qa_cache import YamlMessageQaCache
from reasonese.openrouter import OpenRouterClient, RoutePreference
from reasonese.routing import CollectionRouting
from reasonese.scenario_selection import bind_scenario_selection
from reasonese.scenarios import ScenarioLibrary, load_scenarios
from reasonese.study import Study, make_study
from reasonese.study_cache import SqliteStudyCache
from tests.test_scenarios import (
    PAIR_ID,
    SCENARIOS,
    FakeTransport,
    _collection_posts,
    _library,
    _manual_library,
    _minimal,
    _pair,
    _pairs,
    _parse,
    _study,
)


def _other_study() -> Study:
    study = _study()
    return make_study(
        tuple(replace(spec, framing=Framing.CASUAL) for spec in study.inputs),
        study.assistant,
        1,
    )


def _edited_library() -> ScenarioLibrary:
    return ScenarioLibrary({_pair().pair_id: _parse(_minimal())}, _library().memberships)


@pytest.mark.parametrize("shared_cache", [False, True], ids=["separate-databases", "suite"])
@pytest.mark.parametrize("legacy", [False, True], ids=["bound-root", "legacy-root"])
@pytest.mark.parametrize(
    ("initial", "changed"),
    [(_library(), None), (None, _library()), (_library(), _edited_library())],
    ids=["scenario-to-bare", "bare-to-scenario", "edited-scenario"],
)
def test_new_study_cannot_change_an_existing_roots_scenario_selection(
    tmp_path: Path,
    shared_cache: bool,
    legacy: bool,
    initial: ScenarioLibrary | None,
    changed: ScenarioLibrary | None,
) -> None:
    root = tmp_path / "collection"
    manual = _manual_library(tmp_path)
    cache = SqliteStudyCache(root / "collection.sqlite3") if shared_cache else None
    messages = YamlMessageCache(root / "generated_messages.yaml")
    qa = YamlMessageQaCache(root / "message_qa.yaml")
    collect_studies(
        (CollectionTask(_study(), root / "first"),),
        OpenRouterClient(FakeTransport(_collection_posts())),
        manual, messages, qa,
        prefer_batch=True,
        routing=CollectionRouting(RoutePreference.BATCH, True),
        shared_cache=cache,
        scenarios=initial,
    )
    if legacy:
        (root / "scenario_selection.json").unlink(missing_ok=True)
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    spare = FakeTransport(_collection_posts())

    with pytest.raises(ValueError, match="different scenario selection"):
        collect_studies(
            (CollectionTask(_other_study(), root / "second"),),
            OpenRouterClient(spare),
            manual, messages, qa,
            prefer_batch=True,
            routing=CollectionRouting(RoutePreference.BATCH, True),
            shared_cache=cache,
            scenarios=changed,
        )

    assert spare.post_calls == []
    assert not (root / "second").exists()
    assert before == {
        path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()
    }


@pytest.mark.parametrize("shared_cache", [False, True], ids=["separate-databases", "suite"])
@pytest.mark.parametrize("scenarios", [None, _library()], ids=["bare", "scenario"])
def test_same_selection_can_extend_a_legacy_root_and_resume(
    tmp_path: Path, shared_cache: bool, scenarios: ScenarioLibrary | None
) -> None:
    root = tmp_path / "collection"
    manual = _manual_library(tmp_path)
    cache = SqliteStudyCache(root / "collection.sqlite3") if shared_cache else None
    messages = YamlMessageCache(root / "generated_messages.yaml")
    qa = YamlMessageQaCache(root / "message_qa.yaml")
    routing = CollectionRouting(RoutePreference.BATCH, True)
    collect_studies(
        (CollectionTask(_study(), root / "first"),),
        OpenRouterClient(FakeTransport(_collection_posts())), manual, messages, qa,
        prefer_batch=True, routing=routing, shared_cache=cache, scenarios=scenarios,
    )
    path = root / "scenario_selection.json"
    selection = path.read_bytes()
    path.unlink()
    observations = (root / "first" / "observations.jsonl").read_bytes()
    tasks = (CollectionTask(_other_study(), root / "second"),)

    cold = collect_studies(
        tasks, OpenRouterClient(FakeTransport(_collection_posts())), manual, messages, qa,
        prefer_batch=True, routing=routing, shared_cache=cache, scenarios=scenarios,
    )
    warm = collect_studies(
        tasks, None, manual, messages, qa,
        prefer_batch=True, routing=routing, shared_cache=cache, scenarios=scenarios,
    )

    assert cold[0].trace_cache_hits == 0
    assert warm[0].trace_cache_hits == warm[0].judgment_cache_hits == 2
    assert cold[0].observations == warm[0].observations
    assert path.read_bytes() == selection
    assert (root / "first" / "observations.jsonl").read_bytes() == observations


def test_selection_uses_template_contents_and_binds_even_unused_pairs(tmp_path: Path) -> None:
    root = tmp_path / "collection"
    bind_scenario_selection(root, _library())
    path = root / "scenario_selection.json"
    original = path.read_bytes()

    copied = tmp_path / "copied-scenarios"
    copied.mkdir()
    (copied / f"{PAIR_ID}.yaml").write_bytes((SCENARIOS / f"{PAIR_ID}.yaml").read_bytes())
    bind_scenario_selection(root, load_scenarios(copied, _pairs()))
    assert path.read_bytes() == original

    added_pair = _pair("prime-1234-bare-vs-table")
    extended = ScenarioLibrary(
        {
            **_library().scenarios,
            added_pair.pair_id: replace(_parse(_minimal()), pair_id=added_pair.pair_id),
        },
        _library().memberships,
    )
    with pytest.raises(ValueError, match="different scenario selection"):
        bind_scenario_selection(root, extended)
    assert path.read_bytes() == original


def test_standalone_study_respects_an_ancestor_collection_selection(tmp_path: Path) -> None:
    root = tmp_path / "collection"
    bind_scenario_selection(root, _library())
    transport = FakeTransport(_collection_posts())

    with pytest.raises(ValueError, match="different scenario selection"):
        collect_study(
            _study(), root / "new-study", OpenRouterClient(transport), _manual_library(tmp_path),
            prefer_batch=True, routing=CollectionRouting(RoutePreference.BATCH, True),
        )
    assert transport.post_calls == []
    assert not (root / "new-study").exists()
    bind_scenario_selection(root / "compatible", _library())


def test_new_parent_root_cannot_absorb_a_different_nested_selection(tmp_path: Path) -> None:
    root = tmp_path / "collection"
    bind_scenario_selection(root / "first", _library())

    with pytest.raises(ValueError, match="different scenario selection"):
        bind_scenario_selection(root, None)
    assert not (root / "scenario_selection.json").exists()


def test_invalid_selection_record_is_not_replaced(tmp_path: Path) -> None:
    path = tmp_path / "scenario_selection.json"
    path.write_text("{unfinished", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid scenario selection"):
        bind_scenario_selection(tmp_path, None)
    assert path.read_text(encoding="utf-8") == "{unfinished"


@pytest.mark.parametrize("row", ['{"trial_id": "missing"}', '[]', '{"trial_id": []}'])
def test_legacy_observations_without_trace_evidence_cannot_be_adopted(
    tmp_path: Path, row: str
) -> None:
    path = tmp_path / "observations.jsonl"
    path.write_text("\n" + row + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cannot verify scenario selection"):
        bind_scenario_selection(tmp_path, None)
    assert not (tmp_path / "scenario_selection.json").exists()


def test_legacy_empty_output_can_be_adopted_without_changing_its_database(tmp_path: Path) -> None:
    (tmp_path / "observations.jsonl").write_text("\n", encoding="utf-8")
    database = tmp_path / "collection.sqlite3"
    connection = sqlite3.connect(database)
    connection.close()
    original = database.read_bytes()

    bind_scenario_selection(tmp_path, None)

    assert json.loads((tmp_path / "scenario_selection.json").read_text()) == {"scenarios": None}
    assert database.read_bytes() == original
    assert tuple(SqliteStudyCache(tmp_path / "missing.sqlite3").iter_traces_readonly()) == ()
    assert not (tmp_path / "missing.sqlite3").exists()


def test_malformed_legacy_trace_database_is_not_adopted(tmp_path: Path) -> None:
    connection = sqlite3.connect(tmp_path / "collection.sqlite3")
    try:
        connection.execute("CREATE TABLE traces (wrong_column TEXT)")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(sqlite3.OperationalError, match="no such column"):
        bind_scenario_selection(tmp_path, None)
    assert not (tmp_path / "scenario_selection.json").exists()


@pytest.mark.parametrize("same_selection", [False, True])
def test_concurrent_collectors_cannot_replace_a_root_selection(
    tmp_path: Path, same_selection: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    import reasonese.scenario_selection as selection_module

    ready = Barrier(2)
    validate = selection_module._validate_legacy_root

    def synchronize(root: Path, scenarios: ScenarioLibrary | None, selection: dict[str, object]) -> None:
        validate(root, scenarios, selection)
        ready.wait(timeout=10)

    monkeypatch.setattr(selection_module, "_validate_legacy_root", synchronize)
    selected = (None, None if same_selection else _library())
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(bind_scenario_selection, tmp_path, value) for value in selected]
        errors = [future.exception() for future in futures]

    assert sum(error is None for error in errors) == (2 if same_selection else 1)
    assert all(error is None or isinstance(error, ValueError) for error in errors)
    recorded = json.loads((tmp_path / "scenario_selection.json").read_text())
    assert recorded["scenarios"] is None or PAIR_ID in recorded["scenarios"]
    assert [path.name for path in tmp_path.iterdir()] == ["scenario_selection.json"]
