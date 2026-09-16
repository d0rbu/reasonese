"""Replay-score probes over exact saved assistant-facing contexts."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
import textwrap
import types
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

import reasonese.probe_posthoc as posthoc_module
from reasonese.axes import Assistant
from reasonese.conversation import (
    ConversationSetup,
    ConversationTrace,
    GeneratedMessage,
    GeneratedText,
    construct_conversation,
)
from reasonese.probe_posthoc import main as posthoc_main
from reasonese.probe_posthoc import score_saved_collections
from reasonese.probe_qa import (
    ProbeExpectation,
    ProbeQaRequest,
    ProbeQaVerdict,
    probe_expectation,
    probe_qa_report,
    probe_qa_requests,
    score_probe_qa_diagnostics,
)
from reasonese.study import Study, Trial, build_trials, study_to_dict
from reasonese.study_cache import SqliteStudyCache
from tests.test_authoring_exclusions import _studies


class DeterministicScorer:
    """Small local fixture whose scores depend on the exact rendered context."""

    def __init__(self) -> None:
        self.preflighted = 0
        self.calls: list[tuple[ProbeQaRequest, ...]] = []

    def preflight(self, assistants: tuple[Assistant, ...]) -> None:
        self.preflighted += len(assistants)

    def check(self, requests: tuple[ProbeQaRequest, ...]) -> tuple[ProbeQaVerdict, ...]:
        self.calls.append(requests)
        verdicts = []
        for request in requests:
            fingerprint_payload = json.dumps(
                {
                    "messages": request.setup.openrouter_messages(),
                    "permutation": request.permutation,
                    "position": request.position,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            context_fingerprint = hashlib.sha256(fingerprint_payload.encode()).hexdigest()
            expectation = probe_expectation(request.spec)
            reasoning_probability = 0.6
            role_probabilities = (
                ("system", 0.1),
                ("user", 0.1),
                ("tool", 0.1),
                ("reasoning", reasoning_probability),
                ("assistant", 0.1),
            )
            complies = (
                None
                if expectation is ProbeExpectation.DESCRIPTIVE
                else expectation is ProbeExpectation.REASONING
            )
            issue = None if complies is not False else "deterministic reference mismatch"
            verdicts.append(
                ProbeQaVerdict(
                    request,
                    context_fingerprint,
                    role_probabilities,
                    reasoning_probability,
                    expectation,
                    complies,
                    issue,
                )
            )
        return tuple(verdicts)


def _saved_setup(trial: Trial, text_revision: str = "saved") -> ConversationSetup:
    messages = tuple(
        GeneratedMessage(
            spec,
            GeneratedText.parse(f"{text_revision} materialized text for {spec.instruction}"),
            None,
        )
        for spec in trial.matchup.inputs
    )
    return construct_conversation(trial.matchup, messages)


def _save_collection(
    root: Path,
    study: Study,
    *,
    missing_permutation: int | None = None,
    split_first_permutation: bool = False,
    text_revision: str = "saved",
) -> tuple[Path, tuple[Trial, ...]]:
    root.mkdir(parents=True)
    (root / "study.yaml").write_text(
        yaml.safe_dump(study_to_dict(study), sort_keys=False), encoding="utf-8"
    )
    trials = build_trials(study)
    traces = []
    for trial in trials:
        permutation = int(trial.permutation)
        if permutation == missing_permutation:
            continue
        revision = text_revision
        if split_first_permutation and permutation == 1 and int(trial.rollout) == 2:
            revision = "changed rollout context"
        traces.append(
            (
                trial.trial_id,
                ConversationTrace(_saved_setup(trial, revision), {"id": str(trial.trial_id)}),
            )
        )
    SqliteStudyCache(root / "collection.sqlite3").put_traces(tuple(traces))
    return root, trials


def test_posthoc_scores_exact_saved_contexts_and_matches_inline_measurements(
    tmp_path: Path,
) -> None:
    study = _studies()[0]
    collection, trials = _save_collection(tmp_path / "collection", study)
    source_database = collection / "collection.sqlite3"
    source_before = hashlib.sha256(source_database.read_bytes()).hexdigest()
    scorer = DeterministicScorer()

    summary = score_saved_collections(
        collection,
        tmp_path / "posthoc",
        scorer,
        {"probe_sha256": "deterministic-test-fixture"},
    )

    saved_setups = (_saved_setup(trials[0]), _saved_setup(trials[2]))
    inline_scorer = DeterministicScorer()
    inline_requests = probe_qa_requests(study, (saved_setups[0], saved_setups[1]))
    inline_verdicts, inline_issues = score_probe_qa_diagnostics(inline_scorer, inline_requests)
    inline_report = probe_qa_report((study,), inline_verdicts, inline_issues)
    posthoc_report = json.loads((tmp_path / "posthoc/probe_qa_report.json").read_text())

    assert summary["scores"] == 4
    assert posthoc_report["scores"] == inline_report["scores"]
    assert posthoc_report["counts"] == inline_report["counts"]
    assert posthoc_report["source_scope"] == "saved_delivered_contexts_only"
    assert "not_assessed" in posthoc_report["eligibility"]
    assert posthoc_report["source_manifest"].endswith("probe_posthoc_manifest.json")
    assert scorer.preflighted == 1
    assert len(scorer.calls) == 1
    assert scorer.calls[0][0].setup.content_for_input(0).startswith("saved materialized")
    assert hashlib.sha256(source_database.read_bytes()).hexdigest() == source_before

    manifest = json.loads((tmp_path / "posthoc/probe_posthoc_manifest.json").read_text())
    saved_trace_rows = manifest["identity"]["studies"][0]["trials_with_saved_traces"]
    assert len(saved_trace_rows) == len(trials)
    assert all(row["trace_fingerprint"] for row in saved_trace_rows)
    assert {row["permutation"] for row in saved_trace_rows} == {1, 2}
    assert all(row["trial_id"] for row in saved_trace_rows)


def test_posthoc_reports_missing_order_without_reconstructing_from_current_text(
    tmp_path: Path,
) -> None:
    study = _studies()[0]
    collection, _ = _save_collection(
        tmp_path / "collection",
        study,
        missing_permutation=2,
        text_revision="saved original",
    )
    # A changed authoring cache cannot fill a missing delivered setup.
    (collection / "generated_messages.yaml").write_text(
        "changed current materialization that must not be used\n", encoding="utf-8"
    )

    summary = score_saved_collections(
        collection,
        tmp_path / "posthoc",
        DeterministicScorer(),
        {"probe_sha256": "fixture"},
    )
    report = json.loads((tmp_path / "posthoc/probe_qa_report.json").read_text())

    assert summary["scores"] == 2
    assert report["counts"]["missing_requests"] == 2
    assert report["counts"]["coverage"] == 0.5
    assert {row["permutation"] for row in report["missing"]} == {2}
    assert all(row["reason"].startswith("no saved delivered context") for row in report["missing"])


def test_posthoc_rejects_mixed_contexts_for_one_order_without_choosing_one(
    tmp_path: Path,
) -> None:
    study = _studies()[0]
    collection, _ = _save_collection(
        tmp_path / "collection", study, split_first_permutation=True
    )

    score_saved_collections(
        collection,
        tmp_path / "posthoc",
        DeterministicScorer(),
        {"probe_sha256": "fixture"},
    )
    report = json.loads((tmp_path / "posthoc/probe_qa_report.json").read_text())

    assert report["counts"]["scores"] == 2
    assert report["counts"]["error_requests"] == 2
    assert report["counts"]["missing_requests"] == 0
    assert {row["permutation"] for row in report["errors"]} == {1}
    assert all("multiple delivered or scheduled contexts" in row["reason"] for row in report["errors"])


def test_posthoc_resume_requires_same_saved_source_and_probe_identity(tmp_path: Path) -> None:
    study = _studies()[0]
    collection, trials = _save_collection(tmp_path / "collection", study)
    output = tmp_path / "posthoc"
    probe_identity: dict[str, object] = {"probe_sha256": "fixture-v1"}
    score_saved_collections(collection, output, DeterministicScorer(), probe_identity)

    # Same identity is resumable; changed instrumentation requires a new receipt directory.
    score_saved_collections(collection, output, DeterministicScorer(), probe_identity)
    with pytest.raises(ValueError, match="identity changed"):
        score_saved_collections(
            collection,
            output,
            DeterministicScorer(),
            {"probe_sha256": "fixture-v2"},
        )

    source_cache = SqliteStudyCache(collection / "collection.sqlite3")
    saved_traces = source_cache.load_traces(trials)
    changed_trace_rows = tuple(
        (
            trial.trial_id,
            replace(saved_traces[trial.trial_id], setup=_saved_setup(trial, "updated source")),
        )
        for trial in trials
        if int(trial.permutation) == 1
    )
    source_cache.put_traces(changed_trace_rows)
    with pytest.raises(ValueError, match="identity changed"):
        score_saved_collections(collection, output, DeterministicScorer(), probe_identity)

    updated_output = tmp_path / "updated-posthoc"
    score_saved_collections(
        collection, updated_output, DeterministicScorer(), probe_identity
    )
    original_report = json.loads((output / "probe_qa_report.json").read_text())
    updated_report = json.loads((updated_output / "probe_qa_report.json").read_text())
    original_fingerprints = {
        row["position"]: row["context_fingerprint"]
        for row in original_report["scores"]
        if row["permutation"] == 1
    }
    updated_fingerprints = {
        row["position"]: row["context_fingerprint"]
        for row in updated_report["scores"]
        if row["permutation"] == 1
    }
    assert original_fingerprints.keys() == updated_fingerprints.keys()
    assert all(original_fingerprints[index] != updated_fingerprints[index] for index in (1, 2))


def test_posthoc_help_does_not_import_local_probe_scorer() -> None:
    script = textwrap.dedent(
        """
        import contextlib
        import io
        import sys
        from reasonese import probe_posthoc
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                probe_posthoc.main(['--help'])
        except SystemExit as error:
            assert error.code == 0
        assert 'reasonese.local_probe_qa' not in sys.modules
        assert 'torch' not in sys.modules
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("database_kind", ["absent", "unrelated-table"])
def test_posthoc_missing_source_database_or_trace_table_stays_readonly(
    tmp_path: Path, database_kind: str
) -> None:
    study = _studies()[0]
    collection = tmp_path / "collection"
    collection.mkdir()
    (collection / "study.yaml").write_text(
        yaml.safe_dump(study_to_dict(study)), encoding="utf-8"
    )
    database = collection / "collection.sqlite3"
    if database_kind == "unrelated-table":
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE unrelated (value TEXT)")
        original_bytes = database.read_bytes()
    else:
        original_bytes = None

    summary = score_saved_collections(
        collection,
        tmp_path / f"posthoc-{database_kind}",
        DeterministicScorer(),
        {"probe_sha256": "fixture"},
    )
    report = json.loads(
        (tmp_path / f"posthoc-{database_kind}/probe_qa_report.json").read_text()
    )

    assert summary["scores"] == 0
    assert report["counts"]["missing_requests"] == 4
    if original_bytes is None:
        assert not database.exists()
    else:
        assert database.read_bytes() == original_bytes


def test_posthoc_rejects_malformed_study_and_manifest(tmp_path: Path) -> None:
    malformed_collection = tmp_path / "malformed-collection"
    malformed_collection.mkdir()
    (malformed_collection / "study.yaml").write_text("not: a study\n", encoding="utf-8")
    with pytest.raises(ValueError):
        score_saved_collections(
            malformed_collection,
            tmp_path / "malformed-output",
            DeterministicScorer(),
            {"probe_sha256": "fixture"},
        )

    study = _studies()[0]
    collection, _ = _save_collection(tmp_path / "collection", study)
    output = tmp_path / "posthoc"
    score_saved_collections(
        collection, output, DeterministicScorer(), {"probe_sha256": "fixture"}
    )
    valid_manifest = json.loads(
        (output / "probe_posthoc_manifest.json").read_text(encoding="utf-8")
    )
    (output / "probe_posthoc_manifest.json").write_text("invalid json", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest is invalid"):
        score_saved_collections(
            collection, output, DeterministicScorer(), {"probe_sha256": "fixture"}
        )
    valid_manifest["format_version"] = True
    (output / "probe_posthoc_manifest.json").write_text(
        json.dumps(valid_manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unsupported format"):
        score_saved_collections(
            collection, output, DeterministicScorer(), {"probe_sha256": "fixture"}
        )


def test_posthoc_refuses_nonempty_output_without_a_matching_identity_receipt(
    tmp_path: Path,
) -> None:
    study = _studies()[0]
    collection, _ = _save_collection(tmp_path / "collection", study)
    output = tmp_path / "occupied-output"
    output.mkdir()
    unrelated_cache = output / "probe_qa_cache.json"
    unrelated_cache.write_text("foreign scores", encoding="utf-8")
    scorer = DeterministicScorer()

    with pytest.raises(ValueError, match="no identity manifest"):
        score_saved_collections(
            collection, output, scorer, {"probe_sha256": "fixture"}
        )

    assert unrelated_cache.read_text(encoding="utf-8") == "foreign scores"
    assert scorer.preflighted == 0
    assert scorer.calls == []


def test_posthoc_requires_explicit_probe_identity(tmp_path: Path) -> None:
    study = _studies()[0]
    collection, _ = _save_collection(tmp_path / "collection", study)
    output = tmp_path / "posthoc"
    scorer = DeterministicScorer()

    with pytest.raises(ValueError, match="probe_identity must identify"):
        score_saved_collections(collection, output, scorer, {})

    assert not output.exists()
    assert scorer.preflighted == 0
    assert scorer.calls == []


def test_posthoc_writes_receipt_before_scoring_for_interrupted_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = _studies()[0]
    collection, _ = _save_collection(tmp_path / "collection", study)
    output = tmp_path / "posthoc"
    cache_path = output / "probe_qa_cache.json"

    class CacheWritingScorer(DeterministicScorer):
        def check(self, requests: tuple[ProbeQaRequest, ...]) -> tuple[ProbeQaVerdict, ...]:
            if cache_path.exists():
                assert cache_path.read_text(encoding="utf-8") == "persisted local scores"
            else:
                cache_path.write_text("persisted local scores", encoding="utf-8")
            return super().check(requests)

    original_report = posthoc_module.probe_qa_report

    def interrupt_report(*args, **kwargs):
        raise RuntimeError("simulated interruption after score cache write")

    monkeypatch.setattr(posthoc_module, "probe_qa_report", interrupt_report)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        score_saved_collections(
            collection,
            output,
            CacheWritingScorer(),
            {"probe_sha256": "fixture"},
        )

    receipt_path = output / "probe_posthoc_manifest.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    cache_before_resume = cache_path.read_bytes()
    assert receipt["identity"]["probe"]["probe_sha256"] == "fixture"
    assert cache_before_resume == b"persisted local scores"

    monkeypatch.setattr(posthoc_module, "probe_qa_report", original_report)
    score_saved_collections(
        collection,
        output,
        CacheWritingScorer(),
        {"probe_sha256": "fixture"},
    )

    assert receipt_path.is_file()
    assert cache_path.read_bytes() == cache_before_resume


def test_posthoc_cli_uses_separate_output_and_reuses_its_scorer_cache_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    study = _studies()[0]
    collection, _ = _save_collection(tmp_path / "collection", study)
    role_probe_config = tmp_path / "role-probes.json"
    role_probe_config.write_text("fixture config", encoding="utf-8")
    cache_paths: list[Path] = []

    class FakeLocalScorer(DeterministicScorer):
        def __init__(self, bundle_path: Path, cache_path: Path, *, execution_device: str) -> None:
            del bundle_path
            assert execution_device == "cpu"
            cache_paths.append(cache_path)
            super().__init__()

    fake_module = types.ModuleType("reasonese.local_probe_qa")
    fake_module.__dict__.update(
        {
            "LocalProbeQaScorer": FakeLocalScorer,
            "probe_bundle_identity": lambda path: {
            "config_path": str(path.resolve()),
            "config_sha256": "deterministic-test-config",
            },
        }
    )
    monkeypatch.setitem(sys.modules, "reasonese.local_probe_qa", fake_module)
    args = [
        "--collection",
        str(collection),
        "--role-probes",
        str(role_probe_config),
        "--output",
        str(tmp_path / "posthoc"),
        "--probe-execution-device",
        "cpu",
    ]

    assert posthoc_main(args) == 0
    first_summary = json.loads(capsys.readouterr().out)
    assert posthoc_main(args) == 0
    second_summary = json.loads(capsys.readouterr().out)

    assert first_summary == second_summary
    assert len(cache_paths) == 2
    assert cache_paths[0] == cache_paths[1] == tmp_path / "posthoc/probe_qa_cache.json"


def test_posthoc_cli_rejects_nested_output_before_constructing_scorer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = _studies()[0]
    collection, _ = _save_collection(tmp_path / "collection", study)
    role_probe_config = tmp_path / "role-probes.json"
    role_probe_config.write_text("fixture config", encoding="utf-8")
    constructed = False

    def scorer_factory(*args, **kwargs):
        nonlocal constructed
        constructed = True
        return DeterministicScorer()

    fake_module = types.ModuleType("reasonese.local_probe_qa")
    fake_module.__dict__.update(
        {
            "LocalProbeQaScorer": scorer_factory,
            "probe_bundle_identity": lambda path: {"config": str(path)},
        }
    )
    monkeypatch.setitem(sys.modules, "reasonese.local_probe_qa", fake_module)

    with pytest.raises(SystemExit) as error:
        posthoc_main(
            [
                "--collection",
                str(collection),
                "--role-probes",
                str(role_probe_config),
                "--output",
                str(collection / "nested-output"),
            ]
        )

    assert error.value.code == 2
    assert not constructed
