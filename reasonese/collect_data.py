"""Collect permutation-balanced traces, judgments, and observation rows."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml
from beartype import beartype
from phantom.interval import Natural

from reasonese.axes import Assistant
from reasonese.cache import YamlMessageCache, YamlTraceCache
from reasonese.check_messages import audit_messages, require_compliant_messages
from reasonese.config import load_study
from reasonese.conversation import (
    ConversationSetup,
    ConversationTrace,
    GeneratedMessage,
    construct_conversation,
)
from reasonese.judging import Judgment, judge_traces, trace_fingerprint
from reasonese.judgment_cache import YamlJudgmentCache
from reasonese.manual_messages import ManualMessageLibrary
from reasonese.message_qa_cache import YamlMessageQaCache
from reasonese.observations import Observation, observations_from_trial, write_observations
from reasonese.openrouter import OpenRouterClient, RequestsTransport, model_route
from reasonese.planning import PromptSpec
from reasonese.runner import materialize_specs, run_assistants
from reasonese.study import Study, Trial, build_trials, study_to_dict


@beartype
@dataclass(frozen=True, slots=True)
class CollectionTask:
    """One study and its independent trace, judgment, and observation directory."""

    study: Study
    output_dir: Path


@dataclass(slots=True)
class _CollectionState:
    """Mutable orchestration state for one collection task."""

    task: CollectionTask
    trials: tuple[Trial, ...]
    traces: dict[str, ConversationTrace]
    missing_trials: list[Trial]
    trace_hits: int
    judgments: dict[str, Judgment]
    judgment_hits: int


@beartype
@dataclass(frozen=True, slots=True)
class CollectionResult:
    """Completed trials and cache-use metadata for one study collection."""

    observations: tuple[Observation, ...]
    trials: tuple[Trial, ...]
    trace_cache_hits: Natural
    judgment_cache_hits: Natural


def _prepare_task(task: CollectionTask, manual_messages: ManualMessageLibrary) -> _CollectionState:
    trials = build_trials(task.study)
    task.output_dir.mkdir(parents=True, exist_ok=True)
    with (task.output_dir / "study.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(study_to_dict(task.study), handle, sort_keys=False, allow_unicode=True)

    traces: dict[str, ConversationTrace] = {}
    missing_trials: list[Trial] = []
    trace_hits = 0
    for trial in trials:
        cache = YamlTraceCache(task.output_dir / "trials" / str(trial.trial_id) / "trace.yaml")
        cached = cache.get(trial.matchup)
        if cached is None or not manual_messages.matches(cached.setup):
            missing_trials.append(trial)
        else:
            traces[str(trial.trial_id)] = cached
            trace_hits += 1
    return _CollectionState(task, trials, traces, missing_trials, trace_hits, {}, 0)


def _messages_from_cached_trace(state: _CollectionState) -> tuple[GeneratedMessage, ...]:
    first_trace = state.traces[str(state.trials[0].trial_id)]
    content_by_spec = {
        spec: first_trace.setup.content_for_input(index)
        for index, spec in enumerate(first_trace.setup.matchup.inputs)
    }
    return tuple(
        GeneratedMessage(spec, content_by_spec[spec], None) for spec in state.task.study.inputs
    )


@beartype
def collect_studies(
    tasks: tuple[CollectionTask, ...],
    client: OpenRouterClient | None,
    manual_messages: ManualMessageLibrary,
    message_cache: YamlMessageCache,
    qa_cache: YamlMessageQaCache,
    *,
    prefer_batch: bool,
) -> tuple[CollectionResult, ...]:
    """Collect studies together, batching independent provider work across task boundaries."""
    if not tasks:
        raise ValueError("at least one collection task is required")
    output_dirs = tuple(task.output_dir for task in tasks)
    if len(set(output_dirs)) != len(output_dirs):
        raise ValueError("collection task output directories must be distinct")
    studies = tuple(task.study for task in tasks)
    if len(set(studies)) != len(studies):
        raise ValueError("collection task studies must be distinct")

    states = tuple(_prepare_task(task, manual_messages) for task in tasks)
    specs_to_materialize = tuple(
        dict.fromkeys(
            spec for state in states if state.missing_trials for spec in state.task.study.inputs
        )
    )
    materialized_by_spec: dict[PromptSpec, GeneratedMessage] = {}
    if specs_to_materialize:
        if client is None:
            raise ValueError("OPENROUTER_API_KEY is required for uncached conversation trials")
        materialized_by_spec = {
            message.spec: message
            for message in materialize_specs(
                specs_to_materialize,
                client,
                message_cache,
                manual_messages,
                prefer_batch=prefer_batch,
            )
        }

    generated_by_state: list[tuple[GeneratedMessage, ...]] = []
    for state in states:
        if state.missing_trials:
            generated_by_state.append(
                tuple(materialized_by_spec[spec] for spec in state.task.study.inputs)
            )
        else:
            generated_by_state.append(_messages_from_cached_trace(state))

    require_compliant_messages(
        audit_messages(
            tuple(message for messages in generated_by_state for message in messages),
            qa_cache,
            client,
        )
    )

    assistant_work: dict[
        Assistant,
        list[tuple[_CollectionState, Trial, ConversationSetup]],
    ] = {}
    for state, generated in zip(states, generated_by_state, strict=True):
        by_spec = {message.spec: message for message in generated}
        for trial in state.missing_trials:
            setup = construct_conversation(
                trial.matchup,
                tuple(by_spec[spec] for spec in trial.matchup.inputs),
            )
            assistant_work.setdefault(state.task.study.assistant, []).append((state, trial, setup))

    if assistant_work:
        if client is None:  # pragma: no cover - guarded by materialization above
            raise RuntimeError("assistant work requires an OpenRouter client")
        for assistant, work in assistant_work.items():
            new_traces = run_assistants(
                tuple(setup for _, _, setup in work),
                model_route(assistant),
                client,
                prefer_batch=prefer_batch,
            )
            for (state, trial, _), trace in zip(work, new_traces, strict=True):
                YamlTraceCache(
                    state.task.output_dir / "trials" / str(trial.trial_id) / "trace.yaml"
                ).put(trace)
                state.traces[str(trial.trial_id)] = trace

    judgment_caches: list[YamlJudgmentCache] = []
    missing_judgments: list[tuple[int, Trial, ConversationTrace]] = []
    for state_index, state in enumerate(states):
        judgment_cache = YamlJudgmentCache(state.task.output_dir / "judgments.yaml")
        judgment_caches.append(judgment_cache)
        cached_judgments = {
            (judgment.matchup, judgment.trace_fingerprint): judgment
            for judgment in judgment_cache.load()
        }
        for trial in state.trials:
            trace = state.traces[str(trial.trial_id)]
            cached = cached_judgments.get((trial.matchup, trace_fingerprint(trace)))
            if cached is None:
                missing_judgments.append((state_index, trial, trace))
            else:
                state.judgments[str(trial.trial_id)] = cached
                state.judgment_hits += 1

    if missing_judgments:
        if client is None:
            raise ValueError("OPENROUTER_API_KEY is required for uncached judgments")
        new_judgments = judge_traces(
            tuple(trace for _, _, trace in missing_judgments),
            client,
        )
        judgments_by_state: dict[int, list[Judgment]] = {}
        for (state_index, trial, _), judgment in zip(missing_judgments, new_judgments, strict=True):
            states[state_index].judgments[str(trial.trial_id)] = judgment
            judgments_by_state.setdefault(state_index, []).append(judgment)
        for state_index, judgments in judgments_by_state.items():
            judgment_caches[state_index].put_many(tuple(judgments))

    results: list[CollectionResult] = []
    for state in states:
        traces = tuple(state.traces[str(trial.trial_id)] for trial in state.trials)
        observations = tuple(
            observation
            for trial, trace in zip(state.trials, traces, strict=True)
            for observation in observations_from_trial(
                trial,
                trace,
                state.judgments[str(trial.trial_id)],
            )
        )
        write_observations(state.task.output_dir / "observations.jsonl", observations)
        results.append(
            CollectionResult(
                observations,
                state.trials,
                Natural.parse(state.trace_hits),
                Natural.parse(state.judgment_hits),
            )
        )
    return tuple(results)


@beartype
def collect_study(
    study: Study,
    output_dir: Path,
    client: OpenRouterClient | None,
    manual_messages: ManualMessageLibrary,
    *,
    prefer_batch: bool,
) -> CollectionResult:
    """Collect or resume every permutation and rollout in one study."""
    return collect_studies(
        (CollectionTask(study, output_dir),),
        client,
        manual_messages,
        YamlMessageCache(output_dir / "generated_messages.yaml"),
        YamlMessageQaCache(output_dir / "message_qa.yaml"),
        prefer_batch=prefer_batch,
    )[0]


@beartype
def main(argv: Sequence[str] | None = None) -> int:
    """Collect one permutation-balanced study through OpenRouter."""
    parser = argparse.ArgumentParser(prog="reasonese-collect-data")
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--user-messages", type=Path, default=Path("prompts/user"))
    parser.add_argument("--no-batch", action="store_true")
    args = parser.parse_args(argv)

    try:
        study = load_study(args.study)
        api_key = os.environ.get("OPENROUTER_API_KEY")
        client = OpenRouterClient(RequestsTransport(api_key)) if api_key is not None else None
        result = collect_study(
            study,
            args.output,
            client,
            ManualMessageLibrary(args.user_messages),
            prefer_batch=not args.no_batch,
        )
    except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as error:
        parser.error(str(error))

    print(
        json.dumps(
            {
                "cells": len(study.inputs),
                "judgment_cache_hits": int(result.judgment_cache_hits),
                "observations": len(result.observations),
                "output": str(args.output),
                "trace_cache_hits": int(result.trace_cache_hits),
                "trials": len(result.trials),
            },
            sort_keys=True,
        )
    )
    return 0
