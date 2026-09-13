"""Collect permutation-balanced traces, judgments, and observation rows."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import yaml
from beartype import beartype
from phantom.interval import Natural

from reasonese.authoring_report import authoring_report
from reasonese.axes import Assistant
from reasonese.cache import YamlMessageCache
from reasonese.check_messages import audit_messages
from reasonese.config import load_study
from reasonese.conversation import (
    ConversationSetup,
    ConversationTrace,
    GeneratedMessage,
    construct_conversation,
)
from reasonese.judging import (
    FingerprintedTrace,
    Judgment,
    fingerprint_traces,
    judge_fingerprinted_traces,
)
from reasonese.manual_messages import ManualMessageLibrary, ManualMessageSnapshot
from reasonese.message_qa import MessageQaVerdict
from reasonese.message_qa_cache import YamlMessageQaCache
from reasonese.observations import Observation, observations_from_trials, write_observations
from reasonese.openrouter import OpenRouterClient, RequestsTransport, select_route
from reasonese.planning import PromptSpec
from reasonese.routing import CollectionRouting, add_route_arguments, routing_from_arguments
from reasonese.runner import (
    AssistantRunGroup,
    materialize_specs,
    record_cached_authors,
    run_assistant_groups,
)
from reasonese.study import Study, Trial, TrialId, build_trials, study_to_dict
from reasonese.study_cache import SqliteStudyCache


@beartype
@dataclass(frozen=True, slots=True)
class CollectionTask:
    """One study and its observation directory."""

    study: Study
    output_dir: Path


@dataclass(slots=True)
class _CollectionState:
    """Mutable orchestration state for one collection task."""

    task: CollectionTask
    cache: SqliteStudyCache
    trials: tuple[Trial, ...]
    traces: dict[str, FingerprintedTrace]
    missing_trials: list[Trial]
    trace_hits: int
    judgments: dict[str, Judgment]
    judgment_hits: int
    excluded_inputs: tuple[MessageQaVerdict, ...] = ()


@beartype
@dataclass(frozen=True, slots=True)
class CollectionResult:
    """Completed trials and cache-use metadata for one study collection."""

    observations: tuple[Observation, ...]
    trials: tuple[Trial, ...]
    trace_cache_hits: Natural
    judgment_cache_hits: Natural
    excluded_inputs: tuple[MessageQaVerdict, ...] = ()


def _prepare_task(
    task: CollectionTask,
    manual_messages: ManualMessageSnapshot,
    cache: SqliteStudyCache,
    trials: tuple[Trial, ...],
    cached_traces: Mapping[TrialId, ConversationTrace],
) -> _CollectionState:
    task.output_dir.mkdir(parents=True, exist_ok=True)
    with (task.output_dir / "study.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(study_to_dict(task.study), handle, sort_keys=False, allow_unicode=True)

    traces: dict[str, FingerprintedTrace] = {}
    missing_trials: list[Trial] = []
    cached_to_fingerprint: list[tuple[Trial, ConversationTrace]] = []
    manual_matches: dict[ConversationSetup, bool] = {}
    trace_hits = 0
    for trial in trials:
        cached = cached_traces.get(trial.trial_id)
        if cached is not None and cached.setup not in manual_matches:
            manual_matches[cached.setup] = manual_messages.matches(cached.setup)
        if (
            cached is None
            or cached.setup.matchup != trial.matchup
            or not manual_matches[cached.setup]
        ):
            missing_trials.append(trial)
        else:
            cached_to_fingerprint.append((trial, cached))
    for (trial, _), fingerprinted in zip(
        cached_to_fingerprint,
        fingerprint_traces(tuple(trace for _, trace in cached_to_fingerprint)),
        strict=True,
    ):
        traces[str(trial.trial_id)] = fingerprinted
        trace_hits += 1
    return _CollectionState(task, cache, trials, traces, missing_trials, trace_hits, {}, 0)


def _messages_from_cached_trace(state: _CollectionState) -> tuple[GeneratedMessage, ...]:
    first_trace = state.traces[str(state.trials[0].trial_id)].trace
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
    routing: CollectionRouting | None = None,
    shared_cache: SqliteStudyCache | None = None,
) -> tuple[CollectionResult, ...]:
    """Collect studies together, batching independent provider work across task boundaries."""
    routing = routing or CollectionRouting()
    if not tasks:
        raise ValueError("at least one collection task is required")
    output_dirs = tuple(task.output_dir for task in tasks)
    if len(set(output_dirs)) != len(output_dirs):
        raise ValueError("collection task output directories must be distinct")
    studies = tuple(task.study for task in tasks)
    if len(set(studies)) != len(studies):
        raise ValueError("collection task studies must be distinct")

    trials_by_task = tuple(build_trials(task.study) for task in tasks)
    all_trials = tuple(trial for trials in trials_by_task for trial in trials)
    trial_ids = tuple(trial.trial_id for trial in all_trials)
    if len(set(trial_ids)) != len(trial_ids):
        raise ValueError("collection task trial identifiers must be distinct")

    manual_snapshot = manual_messages.snapshot(
        tuple(spec for task in tasks for spec in task.study.inputs)
    )
    if shared_cache is None:
        caches = tuple(SqliteStudyCache(task.output_dir / "collection.sqlite3") for task in tasks)
        cached_traces_by_task = tuple(
            cache.load_traces(trials) for cache, trials in zip(caches, trials_by_task, strict=True)
        )
    else:
        cached_traces = shared_cache.load_traces(all_trials)
        caches = tuple(shared_cache for _ in tasks)
        cached_traces_by_task = tuple(cached_traces for _ in tasks)
    states = tuple(
        _prepare_task(task, manual_snapshot, cache, trials, cached_traces_for_task)
        for task, cache, trials, cached_traces_for_task in zip(
            tasks,
            caches,
            trials_by_task,
            cached_traces_by_task,
            strict=True,
        )
    )
    specs_to_materialize = tuple(
        dict.fromkeys(
            spec for state in states if state.missing_trials for spec in state.task.study.inputs
        )
    )
    for state in states:
        for trace in state.traces.values():
            routing.record(
                "assistant",
                state.task.study.assistant,
                "cache",
                trace.trace.provenance,
                trace.trace.response,
            )
    record_cached_authors(
        tuple(trace.trace for state in states for trace in state.traces.values()),
        message_cache,
        routing,
    )
    materialized_by_spec: dict[PromptSpec, GeneratedMessage] = {}
    if specs_to_materialize:
        materialized_by_spec = {
            message.spec: message
            for message in materialize_specs(
                specs_to_materialize,
                client,
                message_cache,
                manual_snapshot,
                prefer_batch=prefer_batch,
                routing=routing,
                require_collection_permission=True,
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

    audit = audit_messages(
        tuple(message for messages in generated_by_state for message in messages),
        qa_cache,
        client,
        routing=routing,
    )
    rejected = {verdict.spec: verdict for verdict in audit.verdicts if not verdict.complies}
    report = authoring_report(studies, audit.verdicts)
    report_path = message_cache.path.parent / "authoring_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for verdict in rejected.values():
        logging.getLogger(__name__).warning(
            "Author message QA failed: author=%s framing=%s channel=%s instruction=%r issues=%s",
            verdict.spec.author, verdict.spec.framing, verdict.spec.channel,
            verdict.spec.instruction, list(verdict.issues),
        )
    for state in states:
        state.excluded_inputs = tuple(
            rejected[spec] for spec in state.task.study.inputs if spec in rejected
        )
        if state.excluded_inputs:
            # Keep cached evidence, but never publish stale outcomes for excluded comparisons.
            write_observations(state.task.output_dir / "observations.jsonl", ())
            state.trials = ()
            state.missing_trials = []
            state.trace_hits = 0
            state.judgment_hits = 0
    if rejected:
        # A previous aggregate is no longer authoritative; the suite CLI rebuilds it on success.
        (message_cache.path.parent / "observations.jsonl").unlink(missing_ok=True)
    print(f"Authoring QA: {json.dumps(report['counts'], sort_keys=True)}; report={report_path}", file=sys.stderr)

    assistant_work: dict[
        Assistant,
        list[tuple[_CollectionState, Trial, ConversationSetup]],
    ] = {}
    for state, generated in zip(states, generated_by_state, strict=True):
        by_spec = {message.spec: message for message in generated}
        setups_by_matchup = {
            matchup: construct_conversation(
                matchup,
                tuple(by_spec[spec] for spec in matchup.inputs),
            )
            for matchup in dict.fromkeys(trial.matchup for trial in state.missing_trials)
        }
        for trial in state.missing_trials:
            assistant_work.setdefault(state.task.study.assistant, []).append(
                (state, trial, setups_by_matchup[trial.matchup])
            )

    if assistant_work:
        routing.require_paid("uncached assistant work (chargeable web search)")
        if client is None:
            raise ValueError("OPENROUTER_API_KEY is required for uncached conversation trials")
        ordered_work = tuple(assistant_work.items())
        completed_traces: dict[tuple[int, int], ConversationTrace] = {}
        completion_lock = Lock()

        def collect_trace(group_index: int, setup_index: int, trace: ConversationTrace) -> None:
            with completion_lock:
                completed_traces[(group_index, setup_index)] = trace

        try:
            run_assistant_groups(
                tuple(
                    AssistantRunGroup(
                        select_route(assistant, routing.preference),
                        tuple(setup for _, _, setup in work),
                    )
                    for assistant, work in ordered_work
                ),
                client,
                on_trace=collect_trace,
            )
        finally:
            # The scheduler joins its workers before returning or raising. Preserve
            # completed trials in one transaction per cache even when a peer fails.
            cache_writes: dict[SqliteStudyCache, list[tuple[TrialId, ConversationTrace]]] = {}
            completed_keys = sorted(completed_traces)
            fingerprinted_traces = fingerprint_traces(
                tuple(completed_traces[key] for key in completed_keys)
            )
            for (group_index, setup_index), trace in zip(
                completed_keys, fingerprinted_traces, strict=True
            ):
                state, trial, _ = ordered_work[group_index][1][setup_index]
                state.traces[str(trial.trial_id)] = trace
                routing.record(
                    "assistant",
                    trial.matchup.assistant,
                    "new",
                    trace.trace.provenance,
                    trace.trace.response,
                )
                cache_writes.setdefault(state.cache, []).append((trial.trial_id, trace.trace))
            for cache, records in cache_writes.items():
                cache.put_traces(tuple(records))

    missing_judgments: list[tuple[int, Trial, FingerprintedTrace]] = []
    if shared_cache is None:
        cached_judgments_by_state = tuple(
            state.cache.load_judgments(state.trials) for state in states
        )
    else:
        cached_judgments = shared_cache.load_judgments(all_trials)
        cached_judgments_by_state = tuple(cached_judgments for _ in states)
    for state_index, (state, cached_judgments) in enumerate(
        zip(states, cached_judgments_by_state, strict=True)
    ):
        for trial in state.trials:
            trace = state.traces[str(trial.trial_id)]
            cached = cached_judgments.get(trial.trial_id)
            if (
                cached is None
                or cached.matchup != trial.matchup
                or cached.trace_fingerprint != trace.fingerprint
            ):
                missing_judgments.append((state_index, trial, trace))
            else:
                state.judgments[str(trial.trial_id)] = cached
                state.judgment_hits += 1

    if missing_judgments:
        print(
            f"{len(missing_judgments)} uncached judgments (including any changed trace fingerprints)",
            file=sys.stderr,
        )
        routing.require_paid(f"{len(missing_judgments)} uncached judgments")
        if client is None:
            raise ValueError("OPENROUTER_API_KEY is required for uncached judgments")
        new_judgments = judge_fingerprinted_traces(
            tuple(trace for _, _, trace in missing_judgments),
            client,
        )
        judgments_by_state: dict[int, list[tuple[TrialId, Judgment]]] = {}
        for (state_index, trial, _), judgment in zip(missing_judgments, new_judgments, strict=True):
            states[state_index].judgments[str(trial.trial_id)] = judgment
            judgments_by_state.setdefault(state_index, []).append((trial.trial_id, judgment))
        if shared_cache is None:
            for state_index, judgments in judgments_by_state.items():
                states[state_index].cache.put_judgments(tuple(judgments))
        else:
            shared_cache.put_judgments(
                tuple(
                    item
                    for state_index in sorted(judgments_by_state)
                    for item in judgments_by_state[state_index]
                )
            )

    results: list[CollectionResult] = []
    for state in states:
        traces = tuple(state.traces[str(trial.trial_id)] for trial in state.trials)
        observations = observations_from_trials(
            state.trials,
            traces,
            tuple(state.judgments[str(trial.trial_id)] for trial in state.trials),
        )
        write_observations(state.task.output_dir / "observations.jsonl", observations)
        results.append(
            CollectionResult(
                observations,
                state.trials,
                Natural.parse(state.trace_hits),
                Natural.parse(state.judgment_hits),
                state.excluded_inputs,
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
    routing: CollectionRouting | None = None,
) -> CollectionResult:
    """Collect or resume every permutation and rollout in one study."""
    return collect_studies(
        (CollectionTask(study, output_dir),),
        client,
        manual_messages,
        YamlMessageCache(output_dir / "generated_messages.yaml"),
        YamlMessageQaCache(output_dir / "message_qa.yaml"),
        prefer_batch=prefer_batch,
        routing=routing,
    )[0]


@beartype
def main(argv: Sequence[str] | None = None) -> int:
    """Collect one permutation-balanced study through OpenRouter."""
    parser = argparse.ArgumentParser(prog="reasonese-collect-data")
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--user-messages", type=Path, default=Path("prompts/user"))
    parser.add_argument("--no-batch", action="store_true")
    add_route_arguments(parser)
    args = parser.parse_args(argv)

    try:
        routing = routing_from_arguments(args)
        study = load_study(args.study)
        routing.announce(
            tuple(spec.author for spec in study.inputs),
            (study.assistant,),
            prefer_batch=not args.no_batch,
        )
        api_key = os.environ.get("OPENROUTER_API_KEY")
        client = OpenRouterClient(RequestsTransport(api_key)) if api_key is not None else None
        result = collect_study(
            study,
            args.output,
            client,
            ManualMessageLibrary(args.user_messages),
            prefer_batch=not args.no_batch,
            routing=routing,
        )
    except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as error:
        parser.error(str(error))

    print(
        json.dumps(
            {
                "routes": routing.summary(),
                "cells": len(study.inputs),
                "excluded_comparisons": int(bool(result.excluded_inputs)),
                "excluded_trials": len(build_trials(study)) if result.excluded_inputs else 0,
                "authoring_report": str(args.output / "authoring_report.json"),
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
