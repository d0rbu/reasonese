"""Collect multiple studies through shared provider batches and caches."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from beartype import beartype

from reasonese.cache import YamlMessageCache
from reasonese.collect_data import CollectionTask, collect_studies
from reasonese.config import load_study, load_study_suite
from reasonese.local_probe_qa import LocalProbeQaScorer
from reasonese.manual_messages import ManualMessageLibrary
from reasonese.message_qa_cache import YamlMessageQaCache
from reasonese.observations import write_observations
from reasonese.openrouter import OpenRouterClient, RequestsTransport
from reasonese.routing import add_route_arguments, routing_from_arguments
from reasonese.study import Study, build_trials, study_fingerprint
from reasonese.study_cache import SqliteStudyCache


@beartype
def collection_tasks(study_paths: tuple[Path, ...], output_dir: Path) -> tuple[CollectionTask, ...]:
    """Load study files into distinct, human-readable output directories."""
    if not study_paths:
        raise ValueError("at least one --study path is required")
    names = tuple(path.stem for path in study_paths)
    if len(set(names)) != len(names):
        raise ValueError("study filenames must have distinct stems")
    return tuple(
        CollectionTask(load_study(path), output_dir / name)
        for path, name in zip(study_paths, names, strict=True)
    )


@beartype
def suite_collection_tasks(
    studies: tuple[Study, ...], output_dir: Path
) -> tuple[CollectionTask, ...]:
    """Map suite studies to stable fingerprint-named output directories."""
    if not studies:
        raise ValueError("at least one study is required")
    if len(studies) != len(set(studies)):
        raise ValueError("study suite entries must be distinct")
    names = tuple(study_fingerprint(study) for study in studies)
    if len(names) != len(set(names)):
        raise ValueError("study fingerprints must be distinct")
    return tuple(
        CollectionTask(study, output_dir / name) for study, name in zip(studies, names, strict=True)
    )


@beartype
def main(argv: Sequence[str] | None = None) -> int:
    """Collect multiple studies with shared authoring, QA, assistant, and judge batches."""
    parser = argparse.ArgumentParser(prog="reasonese-collect-studies")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--study", action="append", type=Path)
    source.add_argument("--suite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--user-messages", type=Path, default=Path("prompts/user"))
    parser.add_argument("--no-batch", action="store_true")
    parser.add_argument("--role-probes", type=Path)
    parser.add_argument("--probe-execution-device", default="cuda:0")
    add_route_arguments(parser)
    args = parser.parse_args(argv)

    try:
        routing = routing_from_arguments(args)
        if args.suite is None:
            study_paths = tuple(args.study)
            tasks = collection_tasks(study_paths, args.output)
        else:
            studies = load_study_suite(args.suite)
            tasks = suite_collection_tasks(studies, args.output)
            study_paths = tuple(args.suite for _ in studies)
        routing.announce(
            tuple(spec.author for task in tasks for spec in task.study.inputs),
            tuple(task.study.assistant for task in tasks),
            prefer_batch=not args.no_batch,
        )
        api_key = os.environ.get("OPENROUTER_API_KEY")
        client = OpenRouterClient(RequestsTransport(api_key)) if api_key is not None else None
        results = collect_studies(
            tasks,
            client,
            ManualMessageLibrary(args.user_messages),
            YamlMessageCache(args.output / "generated_messages.yaml"),
            YamlMessageQaCache(args.output / "message_qa.yaml"),
            prefer_batch=not args.no_batch,
            probe_scorer=(
                LocalProbeQaScorer(
                    args.role_probes,
                    args.output / "probe_qa_cache.json",
                    execution_device=args.probe_execution_device,
                )
                if args.role_probes is not None
                else None
            ),
            routing=routing,
            shared_cache=(
                SqliteStudyCache(args.output / "collection.sqlite3")
                if args.suite is not None
                else None
            ),
        )
        if args.suite is not None:
            write_observations(
                args.output / "observations.jsonl",
                tuple(observation for result in results for observation in result.observations),
            )
    except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as error:
        parser.error(str(error))

    study_summaries = [
        {
            "cells": len(task.study.inputs),
            "excluded_comparisons": int(
                bool(
                    result.excluded_inputs
                    or any(row.complies is False for row in result.probe_qa_verdicts)
                )
            ),
            "excluded_trials": (
                len(build_trials(task.study))
                if result.excluded_inputs
                or any(row.complies is False for row in result.probe_qa_verdicts)
                else 0
            ),
            "judgment_cache_hits": int(result.judgment_cache_hits),
            "observations": len(result.observations),
            "output": str(task.output_dir),
            "study": str(path),
            "trace_cache_hits": int(result.trace_cache_hits),
            "trials": len(result.trials),
        }
        for path, task, result in zip(study_paths, tasks, results, strict=True)
    ]
    summary = {
        "routes": routing.summary(),
        "authoring_report": str(args.output / "authoring_report.json"),
        "excluded_comparisons": sum(item["excluded_comparisons"] for item in study_summaries),
        "excluded_trials": sum(item["excluded_trials"] for item in study_summaries),
        "observations": sum(item["observations"] for item in study_summaries),
        "probe_qa_report": (
            str(args.output / "probe_qa_report.json") if args.role_probes is not None else None
        ),
        "output": str(args.output),
        "studies": study_summaries,
        "trials": sum(item["trials"] for item in study_summaries),
    }
    if args.suite is not None:
        summary["study_count"] = len(study_summaries)
    print(json.dumps(summary, sort_keys=True))
    return 0
