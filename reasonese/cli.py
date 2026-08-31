"""Command-line interface for the offline experiment pipeline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from reasonese.bradley_terry import fit_bradley_terry
from reasonese.config import load_experiment_config, load_simulation_config
from reasonese.design import build_trials
from reasonese.io import (
    read_outcomes,
    read_responses,
    read_trials,
    write_json,
    write_jsonl,
)
from reasonese.scoring import score_responses
from reasonese.simulation import simulate_responses


def _path(value: str) -> Path:
    return Path(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reasonese",
        description="Controlled pairwise experiments on instruction representation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    design = subparsers.add_parser("design", help="Generate a counterbalanced trial design")
    design.add_argument("--config", type=_path, required=True)
    design.add_argument("--output", type=_path, required=True)

    simulate = subparsers.add_parser(
        "simulate", help="Generate explicitly synthetic responses for pipeline validation"
    )
    simulate.add_argument("--design", type=_path, required=True)
    simulate.add_argument("--config", type=_path, required=True)
    simulate.add_argument("--output", type=_path, required=True)

    score = subparsers.add_parser("score", help="Exact-match responses into pairwise outcomes")
    score.add_argument("--design", type=_path, required=True)
    score.add_argument("--responses", type=_path, required=True)
    score.add_argument("--output", type=_path, required=True)

    fit = subparsers.add_parser("fit", help="Fit a penalized Bradley-Terry ranking")
    fit.add_argument("--outcomes", type=_path, required=True)
    fit.add_argument("--output", type=_path, required=True)
    fit.add_argument("--reference")
    fit.add_argument("--ridge", type=float, default=1.0)
    return parser


def _summary(**values: object) -> None:
    print(json.dumps(values, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "design":
            config = load_experiment_config(arguments.config)
            trials = build_trials(config)
            write_jsonl(arguments.output, trials)
            _summary(design_id=config.design_id, output=str(arguments.output), trials=len(trials))
        elif arguments.command == "simulate":
            trials = read_trials(arguments.design)
            config = load_simulation_config(arguments.config)
            responses = simulate_responses(trials, config)
            write_jsonl(arguments.output, responses)
            _summary(
                model_id=config.model_id,
                output=str(arguments.output),
                responses=len(responses),
                source="synthetic",
            )
        elif arguments.command == "score":
            trials = read_trials(arguments.design)
            responses = read_responses(arguments.responses)
            outcomes = score_responses(trials, responses)
            write_jsonl(arguments.output, outcomes)
            _summary(
                decisive=sum(outcome.status == "decisive" for outcome in outcomes),
                invalid=sum(outcome.status == "invalid" for outcome in outcomes),
                output=str(arguments.output),
            )
        elif arguments.command == "fit":
            outcomes = read_outcomes(arguments.outcomes)
            result = fit_bradley_terry(
                outcomes,
                reference_condition=arguments.reference,
                ridge=arguments.ridge,
            )
            write_json(arguments.output, result.to_dict())
            _summary(
                converged=result.converged,
                decisive=result.decisive_trials,
                output=str(arguments.output),
                source=result.source,
            )
        else:  # pragma: no cover - argparse enforces the command choices
            raise AssertionError(f"unhandled command: {arguments.command}")
    except ValueError as error:
        parser.error(str(error))
    return 0
