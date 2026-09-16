"""Replay local activation-probe diagnostics from saved collection traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from beartype import beartype

from reasonese.config import load_study
from reasonese.conversation import ConversationSetup, ConversationTrace
from reasonese.judging import fingerprint_traces
from reasonese.probe_qa import (
    ProbeQaDiagnosticIssue,
    ProbeQaIssueKind,
    ProbeQaRequest,
    ProbeQaScorer,
    probe_qa_report,
    probe_qa_requests_for_setups,
    score_probe_qa_diagnostics,
)
from reasonese.study import Study, Trial, TrialId, build_trials, study_fingerprint
from reasonese.study_cache import SqliteStudyCache


@beartype
def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@beartype
@dataclass(frozen=True, slots=True)
class _SavedStudy:
    directory: Path
    database: Path
    study: Study
    trials: tuple[Trial, ...]


@beartype
def _collection_studies(collection: Path) -> tuple[_SavedStudy, ...]:
    """Find standalone or suite studies and their existing trace databases."""
    if not collection.is_dir():
        raise ValueError(f"collection directory does not exist: {collection}")
    if (collection / "study.yaml").is_file():
        directories = (collection,)
    else:
        directories = tuple(
            sorted(
                child
                for child in collection.iterdir()
                if child.is_dir() and (child / "study.yaml").is_file()
            )
        )
    if not directories:
        raise ValueError(f"collection contains no saved study.yaml files: {collection}")
    shared_database = collection / "collection.sqlite3"
    studies = tuple(
        _SavedStudy(
            directory,
            shared_database if shared_database.is_file() else directory / "collection.sqlite3",
            load_study(directory / "study.yaml"),
            (),
        )
        for directory in directories
    )
    fingerprints = tuple(study_fingerprint(saved.study) for saved in studies)
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("saved collection contains duplicate study fingerprints")
    return tuple(
        _SavedStudy(saved.directory, saved.database, saved.study, build_trials(saved.study))
        for saved in studies
    )


@beartype
def score_saved_collections(
    collection: Path,
    output: Path,
    scorer: ProbeQaScorer,
    probe_identity: Mapping[str, object],
) -> dict[str, object]:
    """Replay-score saved contexts and write a separate identity-bound report."""
    collection_path = collection.resolve()
    output_path = output.resolve()
    if not probe_identity:
        raise ValueError("posthoc probe_identity must identify the scoring artifacts")
    if output_path == collection_path or collection_path in output_path.parents:
        raise ValueError("posthoc probe output must be outside the collection directory")

    saved_studies = _collection_studies(collection_path)
    by_database: dict[Path, list[_SavedStudy]] = defaultdict(list)
    for saved in saved_studies:
        by_database[saved.database].append(saved)

    traces_by_study: dict[str, dict[TrialId, ConversationTrace]] = {}
    for database, database_studies in by_database.items():
        trials = tuple(trial for saved in database_studies for trial in saved.trials)
        traces = SqliteStudyCache(database).load_traces_readonly(trials)
        for saved in database_studies:
            traces_by_study[study_fingerprint(saved.study)] = {
                trial.trial_id: traces[trial.trial_id]
                for trial in saved.trials
                if trial.trial_id in traces
            }

    requests: list[ProbeQaRequest] = []
    issues: list[ProbeQaDiagnosticIssue] = []
    source_rows: list[dict[str, object]] = []
    source_trials_by_study: dict[str, list[dict[str, object]]] = {}
    traces_for_fingerprint: list[ConversationTrace] = []
    trace_coordinates: list[tuple[str, int, Trial]] = []
    setups_by_study: dict[str, dict[int, list[ConversationSetup]]] = {}
    for saved in saved_studies:
        study = saved.study
        study_id = study_fingerprint(study)
        traces = traces_by_study[study_id]
        setups_by_study[study_id] = {1: [], 2: []}
        source_trials: list[dict[str, object]] = []
        source_trials_by_study[study_id] = source_trials
        for trial in saved.trials:
            trace = traces.get(trial.trial_id)
            if trace is None:
                continue
            if trace.setup.matchup != trial.matchup:
                raise ValueError("saved trace setup does not match its study trial")
            setups_by_study[study_id][int(trial.permutation)].append(trace.setup)
            traces_for_fingerprint.append(trace)
            trace_coordinates.append((study_id, int(trial.permutation), trial))
            source_trials.append(
                {
                    "trial_id": str(trial.trial_id),
                    "permutation": int(trial.permutation),
                    "rollout": int(trial.rollout),
                    "trace_fingerprint": None,
                    "terminal_status": trace.terminal_status,
                    "route_provenance": (
                        trace.provenance.to_dict() if trace.provenance is not None else None
                    ),
                }
            )
        source_rows.append(
            {
                "study_id": study_id,
                "study_file": str((saved.directory / "study.yaml").resolve()),
                "study_file_sha256": _file_sha256(saved.directory / "study.yaml"),
                "database": str(saved.database.resolve()),
                "trials_with_saved_traces": source_trials,
            }
        )

    fingerprinted = fingerprint_traces(tuple(traces_for_fingerprint))
    trace_fingerprint_by_trial: dict[TrialId, str] = {}
    for (_, _, trial), item in zip(trace_coordinates, fingerprinted, strict=True):
        trace_fingerprint_by_trial[trial.trial_id] = str(item.fingerprint)
    for study_source_trials in source_trials_by_study.values():
        for raw_trial in study_source_trials:
            trial_id = TrialId.parse(str(raw_trial["trial_id"]))
            raw_trial["trace_fingerprint"] = trace_fingerprint_by_trial[trial_id]

    for saved in saved_studies:
        study = saved.study
        study_id = study_fingerprint(study)
        permutation_setups = setups_by_study[study_id]
        available: list[tuple[int, ConversationSetup]] = []
        for permutation in (1, 2):
            unique = tuple(dict.fromkeys(permutation_setups[permutation]))
            if len(unique) > 1:
                reason = "saved rollouts contain multiple delivered contexts for this permutation"
                issues.extend(
                    ProbeQaDiagnosticIssue(
                        study_id,
                        permutation,
                        position,
                        ProbeQaIssueKind.ERROR,
                        reason,
                    )
                    for position in (1, 2)
                )
            elif not unique:
                issues.extend(
                    ProbeQaDiagnosticIssue(
                        study_id,
                        permutation,
                        position,
                        ProbeQaIssueKind.MISSING,
                        "no saved delivered context exists for this permutation",
                    )
                    for position in (1, 2)
                )
            else:
                available.append((permutation, unique[0]))
        requests.extend(probe_qa_requests_for_setups(study, tuple(available)))

    source_identity: dict[str, object] = {
        "collection": str(collection_path),
        "studies": source_rows,
        "probe": dict(probe_identity),
    }
    manifest_path = output_path / "probe_posthoc_manifest.json"
    if output_path.exists() and not output_path.is_dir():
        raise ValueError("posthoc output path exists and is not a directory")
    existing_manifest = manifest_path.is_file()
    if output_path.is_dir() and any(output_path.iterdir()) and not existing_manifest:
        raise ValueError(
            "nonempty posthoc output has no identity manifest; choose a fresh output directory"
        )
    if existing_manifest:
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("existing posthoc manifest is invalid") from error
        if (
            not isinstance(existing, dict)
            or set(existing)
            != {"format_version", "source_scope", "identity", "probe_report", "probe_cache"}
            or type(existing.get("format_version")) is not int
            or existing.get("format_version") != 1
            or existing.get("source_scope") != "saved_delivered_contexts_only"
            or existing.get("probe_report") != str(output_path / "probe_qa_report.json")
            or existing.get("probe_cache") != str(output_path / "probe_qa_cache.json")
        ):
            raise ValueError("existing posthoc manifest has an unsupported format")
        if existing.get("identity") != source_identity:
            raise ValueError(
                "posthoc output identity changed; choose a fresh output directory for this source"
            )
    else:
        output_path.mkdir(parents=True, exist_ok=True)

    manifest = {
        "format_version": 1,
        "source_scope": "saved_delivered_contexts_only",
        "identity": source_identity,
        "probe_report": str(output_path / "probe_qa_report.json"),
        "probe_cache": str(output_path / "probe_qa_cache.json"),
    }
    if not existing_manifest:
        temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
        temporary_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_manifest.replace(manifest_path)

    verdicts, scoring_issues = score_probe_qa_diagnostics(scorer, tuple(requests))
    issues.extend(scoring_issues)
    report = probe_qa_report(
        tuple(saved.study for saved in saved_studies), verdicts, tuple(issues)
    )
    report["source_scope"] = "saved_delivered_contexts_only"
    report["eligibility"] = "not_assessed; join study and trial IDs to collection observations"
    report["source_manifest"] = str(manifest_path)
    limitations = getattr(scorer, "limitations", ())
    if limitations:
        report["limitations"] = list(limitations)
    report_path = output_path / "probe_qa_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "collection": str(collection_path),
        "output": str(output_path),
        "studies": len(saved_studies),
        "scores": len(verdicts),
        "missing_requests": report["counts"]["missing_requests"],
        "error_requests": report["counts"]["error_requests"],
        "report": str(report_path),
    }


@beartype
def main(argv: Sequence[str] | None = None) -> int:
    """Score saved delivered input spans without providers, authoring, or judgments."""
    parser = argparse.ArgumentParser(prog="reasonese-score-probes")
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--role-probes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-execution-device", default="cuda:0")
    args = parser.parse_args(argv)
    try:
        if (
            args.output.resolve() == args.collection.resolve()
            or args.collection.resolve() in args.output.resolve().parents
        ):
            raise ValueError("posthoc probe output must be outside the collection directory")

        # Keep optional framework imports behind argument parsing so --help stays offline.
        from reasonese.local_probe_qa import LocalProbeQaScorer, probe_bundle_identity

        probe_identity = probe_bundle_identity(args.role_probes)
        scorer = LocalProbeQaScorer(
            args.role_probes,
            args.output / "probe_qa_cache.json",
            execution_device=args.probe_execution_device,
        )
        summary = score_saved_collections(
            args.collection, args.output, scorer, probe_identity
        )
    except ImportError:
        parser.error("local role-probe scoring requires the 'probes' extra")
    except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0
