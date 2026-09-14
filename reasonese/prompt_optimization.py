"""Bounded authoring-brief comparisons that stop before assistant execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from beartype import beartype

from reasonese.authoring_report import authoring_report
from reasonese.axes import Assistant, Author, Framing
from reasonese.cache import YamlMessageCache
from reasonese.check_messages import audit_messages
from reasonese.config import load_study_suite
from reasonese.conversation import (
    AUTHORING_BRIEFS,
    AuthoringBrief,
    ConversationSetup,
    authoring_request,
    construct_conversation,
)
from reasonese.instructions import load_instruction_pairs, pair_to_dict
from reasonese.manual_messages import ManualMessageLibrary
from reasonese.matchup import prompt_spec_to_dict
from reasonese.message_qa import message_qa_rubric_fingerprint
from reasonese.message_qa_cache import YamlMessageQaCache
from reasonese.openrouter import OpenRouterClient, RequestsTransport
from reasonese.planning import PromptSpec
from reasonese.probe_qa import (
    ProbeQaScorer,
    check_probe_qa,
    probe_qa_report,
    probe_qa_requests,
)
from reasonese.routing import CollectionRouting, add_route_arguments, routing_from_arguments
from reasonese.runner import materialize_specs
from reasonese.study import Study, build_trials, study_fingerprint

# The live comparison is four pairs x eight framings, with one fixed anchor
# per framing.  These are hard ceilings, not a search budget to be increased
# automatically.
MAX_STUDIES = 32
MAX_UNIQUE_INPUTS = 64
MAX_PROBE_REQUESTS = 128
_REQUIRED_FRAMINGS = frozenset(Framing)
_FORMAT_VERSION = 1
_LOGGER = logging.getLogger(__name__)


@beartype
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@beartype
def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


@beartype
def _append_failure(output: Path, stage: str, error: BaseException) -> None:
    with (output / "failures.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "stage": stage,
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )


@beartype
def _write_status(manifest_path: Path, manifest: dict[str, object], status: str) -> None:
    manifest["status"] = status
    _write_json(manifest_path, manifest)


@beartype
def _validate_suite(
    studies: tuple[Study, ...],
    *,
    assistant: Assistant,
    authors: tuple[Author, ...],
) -> tuple[dict[str, object], tuple[PromptSpec, ...]]:
    """Validate the fixed pilot shape and return work counts and unique inputs."""
    if not studies:
        raise ValueError("prompt optimization suite must contain at least one study")
    if len(studies) > MAX_STUDIES:
        raise ValueError(f"prompt optimization accepts at most {MAX_STUDIES} studies")
    if not authors or any(author is Author.USER for author in authors):
        raise ValueError("prompt optimization requires one or more model authors")
    if len(authors) != len(set(authors)):
        raise ValueError("prompt optimization authors must be unique")
    if any(study.assistant is not assistant for study in studies):
        raise ValueError("every study must use the selected prompt optimization assistant")
    if any(int(study.rollouts_per_permutation) != 1 for study in studies):
        raise ValueError("prompt optimization requires one rollout per permutation")
    if len(studies) != len(set(studies)):
        raise ValueError("prompt optimization studies must be distinct")

    specs = tuple(dict.fromkeys(spec for study in studies for spec in study.inputs))
    model_authors = {spec.author for spec in specs if spec.author is not Author.USER}
    if not model_authors <= set(authors):
        missing = ", ".join(sorted(str(author) for author in model_authors - set(authors)))
        raise ValueError(f"suite contains model authors not selected for this run: {missing}")
    framings = {spec.framing for spec in specs}
    if framings != _REQUIRED_FRAMINGS:
        missing = ", ".join(sorted(str(framing) for framing in _REQUIRED_FRAMINGS - framings))
        extra = ", ".join(sorted(str(framing) for framing in framings - _REQUIRED_FRAMINGS))
        details = f"; missing: {missing}" if missing else ""
        details += f"; unexpected: {extra}" if extra else ""
        raise ValueError("suite must cover all eight framing values" + details)

    counts = {
        "studies": len(studies),
        "unique_inputs": len(specs),
        "model_authored_inputs": sum(spec.author is not Author.USER for spec in specs),
        "message_qa_requests": len(specs),
        "probe_qa_requests": len(studies) * 4,
        "assistant_trials": 0,
        "response_judgment_requests": 0,
    }
    return counts, specs


@beartype
def _probe_identity(path: Path) -> dict[str, object]:
    """Record probe configuration and artifact identity without loading checkpoints."""
    if not path.is_file():
        raise ValueError(f"role-probe bundle does not exist: {path}")
    from reasonese.local_probe_qa import CAPTURE_POLICY, load_probe_bundles

    bundles = load_probe_bundles(path)
    return {
        "capture_policy": CAPTURE_POLICY,
        "config": {"path": str(path.resolve()), "sha256": _sha256(path)},
        "bundles": [
            {
                "assistant": str(bundle.assistant),
                "adapter": bundle.adapter.name,
                "checkpoint": str(bundle.checkpoint),
                "probe": str(bundle.probe_path),
                "probe_sha256": _sha256(bundle.probe_path),
            }
            for bundle in bundles
        ],
    }


@beartype
def _pair_identity(path: Path) -> tuple[dict[str, str], dict[str, object]]:
    """Load the authoritative pair bank and bind exact instruction text to pair IDs."""
    pairs = load_instruction_pairs(path)
    mapping = {
        str(instruction): str(pair.pair_id)
        for pair in pairs
        for instruction in pair.instructions
    }
    if len(mapping) != 2 * len(pairs):
        raise ValueError("instruction pair bank contains a reused instruction")
    return mapping, {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "pairs": [pair_to_dict(pair) for pair in pairs],
    }


@beartype
def _bind_pair_ids(
    specs: tuple[PromptSpec, ...], pair_ids: dict[str, str] | None
) -> dict[str, str]:
    """Bind each exact base instruction to its declared pair or fail closed."""
    if pair_ids is None:
        raise ValueError("prompt optimization requires exact instruction pair identity")
    if any(not value for value in pair_ids.values()):
        raise ValueError("instruction pair identity must map each instruction to one non-empty id")
    unknown = {str(spec.instruction) for spec in specs} - set(pair_ids)
    if unknown:
        raise ValueError("instruction pair identity is missing a suite instruction")
    return {str(spec.instruction): pair_ids[str(spec.instruction)] for spec in specs}


@beartype
def _validate_study_pair_ids(
    studies: tuple[Study, ...], pair_ids: dict[str, str]
) -> None:
    """Require each matchup to remain inside one declared instruction pair."""
    for study in studies:
        study_pair_ids = {
            pair_ids[str(spec.instruction)] for spec in study.inputs
        }
        if len(study_pair_ids) != 1:
            raise ValueError("each prompt optimization study must use one instruction pair")


@beartype
def _add_pair_ids(
    report: dict[str, object], pair_ids: dict[str, str]
) -> dict[str, object]:
    """Add pair coordinates to reusable reports without changing their schemas."""
    inputs = report.get("inputs")
    if isinstance(inputs, list):
        for raw_row in inputs:
            if isinstance(raw_row, dict):
                row = cast(dict[str, object], raw_row)
                instruction = row.get("instruction")
                if isinstance(instruction, str):
                    row["pair_id"] = pair_ids[instruction]
    scores = report.get("scores")
    if isinstance(scores, list):
        for raw_row in scores:
            if isinstance(raw_row, dict):
                row = cast(dict[str, object], raw_row)
                instruction = row.get("instruction")
                if isinstance(instruction, str):
                    row["pair_id"] = pair_ids[instruction]
    comparisons = report.get("comparisons")
    if isinstance(comparisons, list):
        for raw_row in comparisons:
            if not isinstance(raw_row, dict):
                continue
            row = cast(dict[str, object], raw_row)
            raw_inputs = row.get("inputs")
            if isinstance(raw_inputs, list):
                input_pair_ids = []
                for item in raw_inputs:
                    if not isinstance(item, dict):
                        continue
                    instruction = cast(dict[str, object], item).get("instruction")
                    if isinstance(instruction, str):
                        input_pair_ids.append(pair_ids[instruction])
                row["input_pair_ids"] = input_pair_ids
                row["pair_ids"] = sorted(
                    set(input_pair_ids)
                )
    return report


@beartype
def evaluate_prompt_brief(
    studies: tuple[Study, ...],
    *,
    brief: AuthoringBrief,
    output: Path,
    client: OpenRouterClient | None,
    manual_messages: ManualMessageLibrary,
    probe_scorer: ProbeQaScorer,
    assistant: Assistant,
    authors: tuple[Author, ...],
    suite_path: Path | None = None,
    pair_ids: dict[str, str] | None = None,
    pair_identity: dict[str, object] | None = None,
    probe_identity: dict[str, object] | None = None,
    prefer_batch: bool = True,
    routing: CollectionRouting | None = None,
) -> dict[str, object]:
    """Evaluate one immutable brief through authoring, message QA, and probe QA only."""
    if output.exists():
        if not output.is_dir():
            raise ValueError(f"prompt optimization output is not a directory: {output}")
        if any(output.iterdir()):
            raise ValueError("each prompt optimization candidate requires a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    routing = routing or CollectionRouting()
    counts, specs = _validate_suite(studies, assistant=assistant, authors=authors)
    bound_pair_ids = _bind_pair_ids(specs, pair_ids)
    _validate_study_pair_ids(studies, bound_pair_ids)
    if probe_identity is None:
        probe_identity = {"provided": False}
    manifest: dict[str, object] = {
        "format_version": _FORMAT_VERSION,
        "status": "running",
        "candidate": brief.to_dict(),
        "candidate_fingerprint": brief.fingerprint,
        "authors": [str(author) for author in authors],
        "assistant": str(assistant),
        "routing": {
            "preference": str(routing.preference),
            "prefer_batch": prefer_batch,
        },
        "suite": {
            "path": str(suite_path.resolve()) if suite_path is not None else None,
            "sha256": _sha256(suite_path) if suite_path is not None else None,
            "study_count": len(studies),
            "study_fingerprints": [study_fingerprint(study) for study in studies],
        },
        "instruction_pairs": pair_identity,
        "probe": probe_identity,
        "message_qa": {
            "rubric_sha256": message_qa_rubric_fingerprint(),
            "cache": "message_qa.yaml",
        },
        "limits": {
            "max_studies": MAX_STUDIES,
            "max_unique_inputs": MAX_UNIQUE_INPUTS,
            "max_probe_requests": MAX_PROBE_REQUESTS,
            "probe_requests_per_study": 4,
        },
        "work": counts,
        "stages": {
            "authoring": True,
            "message_qa": True,
            "probe_qa": True,
            "assistant_execution": False,
            "response_judging": False,
            "tool_calls": False,
        },
        "artifacts": {
            "manifest": "manifest.json",
            "authoring_requests": "authoring_requests.json",
            "generated_messages": "generated_messages.yaml",
            "message_qa": "message_qa.yaml",
            "authoring_report": "authoring_report.json",
            "probe_qa_report": "probe_qa_report.json",
            "summary": "summary.json",
            "failures": "failures.jsonl",
        },
    }
    _write_json(
        output / "authoring_requests.json",
        {
            "candidate_fingerprint": brief.fingerprint,
            "requests": [
                {
                    "input": prompt_spec_to_dict(spec),
                    "request": authoring_request(spec, brief=brief),
                }
                for spec in specs
                if spec.author is not Author.USER
            ],
        },
    )
    manifest["authoring_request_count"] = counts["model_authored_inputs"]
    manifest["authoring_requests_sha256"] = _sha256(output / "authoring_requests.json")
    manifest_path = output / "manifest.json"
    _write_json(manifest_path, manifest)
    message_cache = YamlMessageCache(output / "generated_messages.yaml")
    qa_cache = YamlMessageQaCache(output / "message_qa.yaml")
    stage = "paid_preflight"
    try:
        # Luna QA is fixed and chargeable; this guard runs before any uncached author call.
        routing.require_paid("uncached prompt optimization authoring and fixed Luna message QA")
        # Validate/load the local probe before spending on authoring or Luna QA.
        stage = "probe_preflight"
        probe_scorer.preflight((assistant,))
        stage = "authoring"
        generated = materialize_specs(
            specs,
            client,
            message_cache,
            manual_messages,
            prefer_batch=prefer_batch,
            routing=routing,
            authoring_brief=brief,
        )
        stage = "message_qa"
        qa_result = audit_messages(
            generated,
            qa_cache,
            client,
            routing=routing,
            prefer_batch=prefer_batch,
        )
        author_report = authoring_report(studies, qa_result.verdicts)
        author_report = _add_pair_ids(author_report, bound_pair_ids)
        _write_json(output / "authoring_report.json", author_report)
        by_spec = {verdict.spec: verdict for verdict in qa_result.verdicts}
        # Probe scores remain measured for every successfully materialized study,
        # including text that message QA rejected.  The two gates have separate
        # denominators; a rejected author input is reported as combined ineligible.
        active_studies = studies
        excluded_studies = tuple(
            study_fingerprint(study)
            for study in studies
            if not all(by_spec[spec].complies for spec in study.inputs)
        )
        setups: list[tuple[ConversationSetup, ConversationSetup]] = []
        for study in active_studies:
            by_message = {message.spec: message for message in generated}
            ordered = tuple(
                construct_conversation(
                    trial.matchup,
                    tuple(by_message[spec] for spec in trial.matchup.inputs),
                )
                for trial in build_trials(study)
                if int(trial.rollout) == 1
            )
            if len(ordered) != 2:
                raise ValueError("prompt optimization requires exactly two ordered setups")
            setups.append((ordered[0], ordered[1]))
        requests = tuple(
            request
            for study, setup in zip(active_studies, setups, strict=True)
            for request in probe_qa_requests(study, setup)
        )
        stage = "probe_qa"
        probe_verdicts = check_probe_qa(probe_scorer, requests)
        probe_report = _add_pair_ids(
            probe_qa_report(active_studies, probe_verdicts), bound_pair_ids
        )
        probe_report["planned_studies"] = len(studies)
        probe_report["probe_active_studies"] = len(active_studies)
        probe_report["message_qa_excluded_studies"] = list(excluded_studies)
        raw_comparisons = probe_report.get("comparisons")
        if not isinstance(raw_comparisons, list):
            raise ValueError("probe report lacks comparison rows")
        probe_failed = {
            cast(dict[str, object], row)["study_id"]
            for row in raw_comparisons
            if isinstance(row, dict) and cast(dict[str, object], row).get("excluded") is True
        }
        probe_report["combined_eligible_studies"] = sum(
            all(by_spec[spec].complies for spec in study.inputs)
            and study_fingerprint(study) not in probe_failed
            for study in studies
        )
        _write_json(output / "probe_qa_report.json", probe_report)
        summary = {
            "candidate": brief.name,
            "candidate_fingerprint": brief.fingerprint,
            "message_qa": {
                "messages": len(qa_result.verdicts),
                "cache_hits": int(qa_result.cache_hits),
                "complies": sum(verdict.complies for verdict in qa_result.verdicts),
                "failed": sum(not verdict.complies for verdict in qa_result.verdicts),
            },
            "probe_qa": probe_report["counts"],
            "message_qa_excluded_studies": list(excluded_studies),
            "assistant_execution": {"trials": 0, "performed": False},
        }
        _write_json(output / "summary.json", summary)
        manifest["results"] = summary
        _write_status(manifest_path, manifest, "completed")
        return summary
    except BaseException as error:
        _LOGGER.exception("prompt optimization failed during %s", stage)
        _append_failure(output, stage, error)
        _write_status(manifest_path, manifest, "failed")
        raise


@beartype
def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid prompt optimization JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"prompt optimization artifact must be an object: {path}")
    return value


@beartype
def _coordinate_key(row: dict[str, object], source: Path) -> tuple[str, str]:
    pair_id = row.get("pair_id")
    instruction = row.get("instruction")
    if not isinstance(pair_id, str) or not pair_id:
        raise ValueError(f"{source} lacks an exact pair_id")
    if not isinstance(instruction, str) or not instruction:
        raise ValueError(f"{source} lacks an exact instruction")
    return (
        pair_id,
        json.dumps(
            {
                key: row.get(key)
                for key in (
                    "instruction",
                    "framing",
                    "channel",
                    "author",
                    "study_id",
                    "permutation",
                    "position",
                )
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


@beartype
def _judge_rows(
    output: Path, filename: str, rows_key: str
) -> dict[tuple[str, str], object]:
    report = _read_json(output / filename)
    rows = report.get(rows_key)
    if not isinstance(rows, list):
        raise ValueError(f"{output / filename} must contain a {rows_key} list")
    result: dict[tuple[str, str], object] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{output / filename} contains a non-object row")
        row_data = cast(dict[str, object], row)
        key = _coordinate_key(row_data, output / filename)
        if key in result:
            raise ValueError(f"{output / filename} contains duplicate judge coordinates")
        result[key] = row.get("complies")
    return result


@beartype
def _expected_message_rows(output: Path) -> dict[tuple[str, str], None]:
    """Derive one exact QA coordinate for every planned input."""
    report = _read_json(output / "authoring_report.json")
    comparisons = report.get("comparisons")
    if not isinstance(comparisons, list):
        raise ValueError(f"{output / 'authoring_report.json'} must contain comparisons")
    expected: dict[tuple[str, str], None] = {}
    for raw_comparison in comparisons:
        if not isinstance(raw_comparison, dict):
            raise ValueError("authoring comparison rows must be objects")
        comparison = cast(dict[str, object], raw_comparison)
        raw_inputs = comparison.get("inputs")
        raw_pairs = comparison.get("input_pair_ids")
        if not isinstance(raw_inputs, list) or not isinstance(raw_pairs, list):
            raise ValueError("authoring comparisons must bind inputs to pair IDs")
        if len(raw_inputs) != len(raw_pairs):
            raise ValueError("authoring comparison inputs and pair IDs must align")
        for raw_input, raw_pair in zip(raw_inputs, raw_pairs, strict=True):
            if not isinstance(raw_input, dict) or not isinstance(raw_pair, str) or not raw_pair:
                raise ValueError("authoring comparison has an invalid input pair binding")
            row = cast(dict[str, object], raw_input).copy()
            row["pair_id"] = raw_pair
            expected[_coordinate_key(row, output / "authoring_report.json")] = None
    return expected


@beartype
def _judge_totals(
    rows: dict[tuple[str, str], object],
    pair_id: str,
    expected: set[tuple[str, str]],
    *,
    allow_descriptive: bool,
) -> dict[str, int]:
    selected_rows = {
        key: value for key, value in rows.items() if key[0] == pair_id
    }
    selected = list(selected_rows.values())
    invalid = [
        value
        for value in selected
        if (value is None and not allow_descriptive)
        or (value is not None and not isinstance(value, bool))
    ]
    if invalid:
        raise ValueError("judge rows must contain boolean or null complies values")
    descriptive = sum(value is None for value in selected)
    eligible = sum(isinstance(value, bool) for value in selected)
    passed = sum(value is True for value in selected)
    return {
        "pass_numerator": passed,
        "pass_denominator": eligible,
        "missing": len(
            {key for key in expected if key[0] == pair_id} - set(selected_rows)
        ),
        "descriptive": descriptive,
    }


@beartype
def _descriptive_probe_values(output: Path) -> dict[str, list[float]]:
    """Preserve raw reasoning scores for compressed, report-only probe rows."""
    report = _read_json(output / "probe_qa_report.json")
    raw_scores = report.get("scores")
    if not isinstance(raw_scores, list):
        raise ValueError(f"{output / 'probe_qa_report.json'} must contain a scores list")
    values: dict[str, list[float]] = {}
    for raw_score in raw_scores:
        if not isinstance(raw_score, dict):
            raise ValueError("probe score rows must be objects")
        score = cast(dict[str, object], raw_score)
        pair_id = score.get("pair_id")
        complies = score.get("complies")
        probability = score.get("reasoning_probability")
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError("probe score lacks an exact pair_id")
        if complies is None:
            if isinstance(probability, bool) or not isinstance(probability, (int, float)):
                raise ValueError("descriptive probe score lacks a numeric reasoning probability")
            value = float(probability)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("descriptive probe reasoning probability must lie between zero and one")
            values.setdefault(pair_id, []).append(value)
    return values


@beartype
def _expected_probe_rows(output: Path) -> dict[tuple[str, str], None]:
    """Derive all four probe coordinates per planned study from the suite report."""
    report = _read_json(output / "authoring_report.json")
    comparisons = report.get("comparisons")
    if not isinstance(comparisons, list):
        raise ValueError(f"{output / 'authoring_report.json'} lacks comparisons")
    expected: dict[tuple[str, str], None] = {}
    for raw_comparison in comparisons:
        if not isinstance(raw_comparison, dict):
            raise ValueError("authoring comparison rows must be objects")
        comparison = cast(dict[str, object], raw_comparison)
        raw_inputs = comparison.get("inputs")
        raw_pairs = comparison.get("input_pair_ids")
        study_id = comparison.get("study_id")
        if not isinstance(raw_inputs, list) or len(raw_inputs) != 2:
            raise ValueError("authoring comparison must contain two inputs")
        if not isinstance(raw_pairs, list) or len(raw_pairs) != 2:
            raise ValueError("authoring comparison must contain two pair IDs")
        if not isinstance(study_id, str) or not study_id:
            raise ValueError("authoring comparison must contain a study ID")
        inputs = cast(list[object], raw_inputs)
        pair_values = cast(list[object], raw_pairs)
        for permutation, order_indices in ((1, (0, 1)), (2, (1, 0))):
            for position, input_index in enumerate(order_indices, start=1):
                item = inputs[input_index]
                if not isinstance(item, dict) or not isinstance(
                    cast(dict[str, object], item).get("instruction"), str
                ):
                    raise ValueError("authoring comparison input is invalid")
                item_data = cast(dict[str, object], item)
                pair_id = pair_values[input_index]
                if not isinstance(pair_id, str) or not pair_id:
                    raise ValueError("authoring comparison cannot bind an input pair ID")
                key = (
                    pair_id,
                    json.dumps(
                        {
                            key: item_data.get(key)
                            for key in (
                                "instruction",
                                "framing",
                                "channel",
                                "author",
                                "study_id",
                                "permutation",
                                "position",
                            )
                        }
                        | {
                            "study_id": study_id,
                            "permutation": permutation,
                            "position": position,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
                if key in expected:
                    raise ValueError("authoring comparisons duplicate a probe coordinate")
                expected[key] = None
    return expected


@beartype
def _format_comparison_markdown(
    rows: list[dict[str, object]],
    baseline_name: str,
    candidate_name: str,
    overall: dict[str, object],
) -> str:
    def format_cell(counts: dict[str, int]) -> str:
        denominator = counts["pass_denominator"]
        rate = "N/A" if denominator == 0 else f"{100 * counts['pass_numerator'] / denominator:.1f}%"
        return f"{counts['pass_numerator']}/{denominator} ({rate})"

    lines = [
        "| Instruction pair | Judge | "
        f"{baseline_name} passed/eligible (rate) | {candidate_name} passed/eligible (rate) |",
        "|---|---|---:|---:|",
    ]
    for row in rows:
        baseline = cast(dict[str, int], row["baseline"])
        candidate = cast(dict[str, int], row["candidate"])
        lines.append(
            f"| {row['pair_id']} | {row['judge']} | "
            f"{format_cell(baseline)} | {format_cell(candidate)} |"
        )
    for judge, key in (("message-QA (GPT-5.6 Luna)", "message_qa"), ("Nemotron role probe", "probe")):
        baseline = cast(dict[str, int], cast(dict[str, object], overall["baseline"])[key])
        candidate = cast(dict[str, int], cast(dict[str, object], overall["candidate"])[key])
        lines.append(
            f"| Overall | {judge} | "
            f"{format_cell(baseline)} | {format_cell(candidate)} |"
        )
    return "\n".join(lines) + "\n"


@beartype
def compare_prompt_outputs(
    baseline_output: Path, candidate_output: Path
) -> dict[str, object]:
    """Compare two completed candidate directories without making provider calls."""
    baseline = _read_json(baseline_output / "manifest.json")
    candidate = _read_json(candidate_output / "manifest.json")
    if baseline.get("status") != "completed" or candidate.get("status") != "completed":
        raise ValueError("prompt comparison requires completed baseline and candidate manifests")
    for field in ("suite", "assistant", "routing", "probe", "instruction_pairs"):
        if baseline.get(field) != candidate.get(field):
            raise ValueError(f"baseline and candidate {field} identities differ")
    baseline_qa = baseline.get("message_qa")
    candidate_qa = candidate.get("message_qa")
    if not isinstance(baseline_qa, dict) or not isinstance(candidate_qa, dict):
        raise ValueError("prompt manifests lack message-QA identity")
    if baseline_qa.get("rubric_sha256") != candidate_qa.get("rubric_sha256"):
        raise ValueError("baseline and candidate message-QA rubrics differ")
    baseline_candidate = baseline.get("candidate")
    candidate_candidate = candidate.get("candidate")
    if not isinstance(baseline_candidate, dict) or not isinstance(candidate_candidate, dict):
        raise ValueError("prompt manifests lack candidate identities")
    baseline_name = baseline_candidate.get("name")
    candidate_name = candidate_candidate.get("name")
    if not isinstance(baseline_name, str) or not isinstance(candidate_name, str):
        raise ValueError("prompt candidates must have names")
    if baseline_name == candidate_name:
        raise ValueError("baseline and candidate names must differ")

    qa_baseline = _judge_rows(baseline_output, "authoring_report.json", "inputs")
    qa_candidate = _judge_rows(candidate_output, "authoring_report.json", "inputs")
    probe_baseline = _judge_rows(baseline_output, "probe_qa_report.json", "scores")
    probe_candidate = _judge_rows(candidate_output, "probe_qa_report.json", "scores")
    descriptive_baseline = _descriptive_probe_values(baseline_output)
    descriptive_candidate = _descriptive_probe_values(candidate_output)
    expected_qa_baseline = set(_expected_message_rows(baseline_output))
    expected_qa_candidate = set(_expected_message_rows(candidate_output))
    expected_probe_baseline = set(_expected_probe_rows(baseline_output))
    expected_probe_candidate = set(_expected_probe_rows(candidate_output))
    if expected_qa_baseline != expected_qa_candidate:
        raise ValueError("baseline and candidate message-QA coordinates differ")
    if set(qa_baseline) != expected_qa_baseline or set(qa_candidate) != expected_qa_candidate:
        raise ValueError("message-QA report coordinates do not match the planned inputs")
    if expected_probe_baseline != expected_probe_candidate:
        raise ValueError("baseline and candidate probe-QA coordinates differ")
    if set(probe_baseline) != expected_probe_baseline or set(probe_candidate) != expected_probe_candidate:
        raise ValueError("probe-QA report coordinates do not match the planned spans")
    all_pairs = sorted(
        {
            pair_id
            for rows in (qa_baseline, qa_candidate, probe_baseline, probe_candidate)
            for pair_id, _ in rows
        }
    )
    result_rows: list[dict[str, object]] = []
    for pair_id in all_pairs:
        for judge, before, after in (
            ("message-QA (GPT-5.6 Luna)", qa_baseline, qa_candidate),
            ("Nemotron role probe", probe_baseline, probe_candidate),
        ):
            expected = (
                expected_qa_baseline
                if judge.startswith("message-QA")
                else expected_probe_baseline
            )
            before_values = _judge_totals(
                before,
                pair_id,
                expected,
                allow_descriptive=judge.startswith("Nemotron"),
            )
            after_values = _judge_totals(
                after,
                pair_id,
                expected,
                allow_descriptive=judge.startswith("Nemotron"),
            )
            result_row: dict[str, object] = {
                "pair_id": pair_id,
                "judge": judge,
                "baseline": before_values,
                "candidate": after_values,
            }
            if judge.startswith("Nemotron"):
                result_row["descriptive_reasoning_probabilities"] = {
                    "baseline": descriptive_baseline.get(pair_id, []),
                    "candidate": descriptive_candidate.get(pair_id, []),
                }
                result_row["descriptive_reasoning_probability_counts"] = {
                    "baseline": len(descriptive_baseline.get(pair_id, [])),
                    "candidate": len(descriptive_candidate.get(pair_id, [])),
                }
            result_rows.append(result_row)
    def all_rows(rows: dict[tuple[str, str], object]) -> dict[tuple[str, str], object]:
        return {(("__all__"), key): value for (_, key), value in rows.items()}

    def all_expected(rows: set[tuple[str, str]]) -> set[tuple[str, str]]:
        return {(("__all__"), key) for (_, key) in rows}

    overall = {
        "baseline": {
            "message_qa": _judge_totals(
                all_rows(qa_baseline),
                "__all__",
                all_expected(expected_qa_baseline),
                allow_descriptive=False,
            ),
            "probe": _judge_totals(
                all_rows(probe_baseline),
                "__all__",
                all_expected(expected_probe_baseline),
                allow_descriptive=True,
            ),
        },
        "candidate": {
            "message_qa": _judge_totals(
                all_rows(qa_candidate),
                "__all__",
                all_expected(expected_qa_baseline),
                allow_descriptive=False,
            ),
            "probe": _judge_totals(
                all_rows(probe_candidate),
                "__all__",
                all_expected(expected_probe_baseline),
                allow_descriptive=True,
            ),
        },
    }
    return {
        "baseline": {"name": baseline_name, "output": str(baseline_output)},
        "candidate": {"name": candidate_name, "output": str(candidate_output)},
        "message_qa_rubric_sha256": baseline_qa.get("rubric_sha256"),
        "rows": result_rows,
        "overall": overall,
        "descriptive_probe": {
            "baseline": descriptive_baseline,
            "candidate": descriptive_candidate,
        },
        "descriptive_probe_counts": {
            "baseline": {pair_id: len(values) for pair_id, values in descriptive_baseline.items()},
            "candidate": {pair_id: len(values) for pair_id, values in descriptive_candidate.items()},
        },
        "markdown": _format_comparison_markdown(
            result_rows, baseline_name, candidate_name, cast(dict[str, object], overall)
        ),
    }


@beartype
def main(argv: Sequence[str] | None = None) -> int:
    """Run one explicitly selected bounded authoring brief."""
    parser = argparse.ArgumentParser(prog="reasonese-optimize-prompt")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--brief", choices=tuple(AUTHORING_BRIEFS), required=True)
    parser.add_argument("--assistant", type=Assistant, choices=tuple(Assistant), default=Assistant.NEMOTRON_3_5_LIGHTNING)
    parser.add_argument(
        "--author",
        action="append",
        type=Author,
        choices=tuple(author for author in Author if author is not Author.USER),
        help="repeat to select model-authored inputs; default: model authors present in the suite",
    )
    parser.add_argument("--user-messages", type=Path, default=Path("prompts/user"))
    parser.add_argument("--role-probes", type=Path, required=True)
    parser.add_argument("--probe-execution-device", default="cuda:0")
    parser.add_argument(
        "--no-batch",
        action="store_true",
        help="use synchronous transport for authoring and message QA; free author routes are synchronous either way",
    )
    add_route_arguments(parser)
    args = parser.parse_args(argv)
    try:
        studies = load_study_suite(args.suite)
        pair_ids, pair_identity = _pair_identity(args.pairs)
        suite_authors = tuple(
            dict.fromkeys(
                spec.author
                for study in studies
                for spec in study.inputs
                if spec.author is not Author.USER
            )
        )
        authors = tuple(args.author) if args.author else suite_authors
        brief = AUTHORING_BRIEFS[args.brief]
        probe_identity = _probe_identity(args.role_probes)
        from reasonese.local_probe_qa import LocalProbeQaScorer

        scorer = LocalProbeQaScorer(
            args.role_probes,
            args.output / "probe_qa_cache.json",
            execution_device=args.probe_execution_device,
        )
        routing = routing_from_arguments(args)
        routing.announce(authors, (args.assistant,), prefer_batch=not args.no_batch)
        api_key = os.environ.get("OPENROUTER_API_KEY")
        client = OpenRouterClient(RequestsTransport(api_key)) if api_key is not None else None
        result = evaluate_prompt_brief(
            studies,
            brief=brief,
            output=args.output,
            client=client,
            manual_messages=ManualMessageLibrary(args.user_messages),
            probe_scorer=scorer,
            assistant=args.assistant,
            authors=authors,
            suite_path=args.suite,
            pair_ids=pair_ids,
            pair_identity=pair_identity,
            probe_identity=probe_identity,
            prefer_batch=not args.no_batch,
            routing=routing,
        )
    except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({"output": str(args.output), "routes": routing.summary(), **result}, sort_keys=True))
    return 0


@beartype
def compare_main(argv: Sequence[str] | None = None) -> int:
    """Render one fixed-rubric baseline/candidate comparison table offline."""
    parser = argparse.ArgumentParser(prog="reasonese-compare-prompts")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = compare_prompt_outputs(args.baseline, args.candidate)
        if args.output is not None:
            _write_json(args.output, result)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0
