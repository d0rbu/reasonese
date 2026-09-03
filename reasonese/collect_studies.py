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
from reasonese.config import load_study
from reasonese.manual_messages import ManualMessageLibrary
from reasonese.message_qa_cache import YamlMessageQaCache
from reasonese.openrouter import OpenRouterClient, RequestsTransport


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
def main(argv: Sequence[str] | None = None) -> int:
    """Collect multiple studies with shared authoring, QA, assistant, and judge batches."""
    parser = argparse.ArgumentParser(prog="reasonese-collect-studies")
    parser.add_argument("--study", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--user-messages", type=Path, default=Path("prompts/user"))
    parser.add_argument("--no-batch", action="store_true")
    args = parser.parse_args(argv)

    try:
        tasks = collection_tasks(tuple(args.study), args.output)
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
        for path, task, result in zip(args.study, tasks, results, strict=True)
    ]
    print(
        json.dumps(
            {
                "observations": sum(item["observations"] for item in study_summaries),
                "output": str(args.output),
                "studies": study_summaries,
                "trials": sum(item["trials"] for item in study_summaries),
            },
            sort_keys=True,
        )
    )
    return 0
