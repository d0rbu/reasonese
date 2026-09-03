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
from reasonese.manual_messages import ManualMessageLibrary
from reasonese.message_qa_cache import YamlMessageQaCache
from reasonese.observations import write_observations
from reasonese.openrouter import OpenRouterClient, RequestsTransport
from reasonese.study import Study, study_fingerprint


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
        CollectionTask(study, output_dir / name)
        for study, name in zip(studies, names, strict=True)
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
    args = parser.parse_args(argv)

    try:
        if args.suite is None:
            study_paths = tuple(args.study)
            tasks = collection_tasks(study_paths, args.output)
        else:
            studies = load_study_suite(args.suite)
            tasks = suite_collection_tasks(studies, args.output)
            study_paths = tuple(args.suite for _ in studies)
        api_key = os.environ.get("OPENROUTER_API_KEY")
        client = OpenRouterClient(RequestsTransport(api_key)) if api_key is not None else None
        results = collect_studies(
            tasks,
            client,
            ManualMessageLibrary(args.user_messages),
            YamlMessageCache(args.output / "generated_messages.yaml"),
            YamlMessageQaCache(args.output / "message_qa.yaml"),
            prefer_batch=not args.no_batch,
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
        "observations": sum(item["observations"] for item in study_summaries),
        "output": str(args.output),
        "studies": study_summaries,
        "trials": sum(item["trials"] for item in study_summaries),
    }
    if args.suite is not None:
        summary["study_count"] = len(study_summaries)
    print(json.dumps(summary, sort_keys=True))
    return 0
