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

from reasonese.cache import YamlMessageCache, YamlTraceCache
from reasonese.config import load_study
from reasonese.conversation import ConversationTrace, construct_conversation
from reasonese.judging import Judgment, judge_traces, trace_fingerprint
from reasonese.judgment_cache import YamlJudgmentCache
from reasonese.observations import Observation, observations_from_trial, write_observations
from reasonese.openrouter import OpenRouterClient, RequestsTransport, model_route
from reasonese.runner import materialize_messages, run_assistants
from reasonese.study import Study, Trial, build_trials, study_to_dict


@beartype
@dataclass(frozen=True, slots=True)
class CollectionResult:
    """Completed trials and cache-use metadata for one study collection."""

    observations: tuple[Observation, ...]
    trials: tuple[Trial, ...]
    trace_cache_hits: Natural
    judgment_cache_hits: Natural


@beartype
def collect_study(
    study: Study,
    output_dir: Path,
    client: OpenRouterClient | None,
    *,
    prefer_batch: bool,
) -> CollectionResult:
    """Collect or resume every permutation and rollout in one study."""
    trials = build_trials(study)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "study.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(study_to_dict(study), handle, sort_keys=False, allow_unicode=True)

    trace_by_id: dict[str, ConversationTrace] = {}
    missing_trials: list[Trial] = []
    trace_hits = 0
    for trial in trials:
        cache = YamlTraceCache(output_dir / "trials" / str(trial.trial_id) / "trace.yaml")
        cached = cache.get(trial.matchup)
        if cached is None:
            missing_trials.append(trial)
        else:
            trace_by_id[str(trial.trial_id)] = cached
            trace_hits += 1

    if missing_trials:
        if client is None:
            raise ValueError("OPENROUTER_API_KEY is required for uncached conversation trials")
        base_matchup = trials[0].matchup
        generated = materialize_messages(
            base_matchup,
            client,
            YamlMessageCache(output_dir / "generated_messages.yaml"),
            prefer_batch=prefer_batch,
        )
        by_spec = {message.spec: message for message in generated}
        setups = tuple(
            construct_conversation(
                trial.matchup,
                tuple(by_spec[spec] for spec in trial.matchup.inputs),
            )
            for trial in missing_trials
        )
        new_traces = run_assistants(
            setups,
            model_route(study.assistant),
            client,
            prefer_batch=prefer_batch,
        )
        for trial, trace in zip(missing_trials, new_traces, strict=True):
            YamlTraceCache(output_dir / "trials" / str(trial.trial_id) / "trace.yaml").put(trace)
            trace_by_id[str(trial.trial_id)] = trace

    traces = tuple(trace_by_id[str(trial.trial_id)] for trial in trials)
    judgment_cache = YamlJudgmentCache(output_dir / "judgments.yaml")
    cached_judgments = {
        (judgment.matchup, judgment.trace_fingerprint): judgment
        for judgment in judgment_cache.load()
    }
    judgment_by_trial: dict[str, Judgment] = {}
    missing_judgment_trials: list[Trial] = []
    missing_judgment_traces: list[ConversationTrace] = []
    judgment_hits = 0
    for trial, trace in zip(trials, traces, strict=True):
        key = (trial.matchup, trace_fingerprint(trace))
        cached = cached_judgments.get(key)
        if cached is None:
            missing_judgment_trials.append(trial)
            missing_judgment_traces.append(trace)
        else:
            judgment_by_trial[str(trial.trial_id)] = cached
            judgment_hits += 1

    if missing_judgment_traces:
        if client is None:
            raise ValueError("OPENROUTER_API_KEY is required for uncached judgments")
        new_judgments = judge_traces(tuple(missing_judgment_traces), client)
        judgment_cache.put_many(new_judgments)
        for trial, judgment in zip(missing_judgment_trials, new_judgments, strict=True):
            judgment_by_trial[str(trial.trial_id)] = judgment

    observations = tuple(
        observation
        for trial, trace in zip(trials, traces, strict=True)
        for observation in observations_from_trial(
            trial,
            trace,
            judgment_by_trial[str(trial.trial_id)],
        )
    )
    write_observations(output_dir / "observations.jsonl", observations)
    return CollectionResult(
        observations,
        trials,
        Natural.parse(trace_hits),
        Natural.parse(judgment_hits),
    )


@beartype
def main(argv: Sequence[str] | None = None) -> int:
    """Collect one permutation-balanced study through OpenRouter."""
    parser = argparse.ArgumentParser(prog="reasonese-collect-data")
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
