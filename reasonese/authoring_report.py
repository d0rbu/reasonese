"""Describe author-QA attrition separately from assistant outcomes."""

from __future__ import annotations

from collections import defaultdict

from beartype import beartype

from reasonese.matchup import prompt_spec_to_dict
from reasonese.message_qa import MessageQaVerdict
from reasonese.observations import cell_id
from reasonese.openrouter import JsonObject
from reasonese.study import Cell, Study, build_trials, study_fingerprint


@beartype
def authoring_report(
    studies: tuple[Study, ...], verdicts: tuple[MessageQaVerdict, ...]
) -> JsonObject:
    """Report this invocation's unique inputs and planned comparison graph.

    Marginal comparison counts include a comparison once per distinct axis value
    on either endpoint; they describe exposure to exclusions, not blame.
    """
    by_spec = {verdict.spec: verdict for verdict in verdicts}
    specs = tuple(dict.fromkeys(spec for study in studies for spec in study.inputs))
    if set(by_spec) != set(specs):
        raise ValueError("authoring report verdicts must match the planned inputs")
    for verdict in verdicts:
        if verdict != by_spec[verdict.spec]:
            raise ValueError("conflicting authoring verdicts for one input")

    inputs: list[JsonObject] = []
    input_counts: dict[str, dict[str, list[int]]] = {
        axis: defaultdict(lambda: [0, 0])
        for axis in ("instruction", "author", "framing", "channel")
    }
    for spec in specs:
        verdict = by_spec[spec]
        coordinates = prompt_spec_to_dict(spec)
        inputs.append(
            {
                **coordinates,
                "content": str(verdict.content),
                "complies": verdict.complies,
                "issues": list(verdict.issues),
                "qa_response_id": verdict.response.get("id"),
            }
        )
        for axis, counts in input_counts.items():
            counts[str(coordinates[axis])][0] += 1
            counts[str(coordinates[axis])][1] += int(not verdict.complies)

    comparisons: list[JsonObject] = []
    comparison_counts: dict[str, dict[str, list[int]]] = {
        axis: defaultdict(lambda: [0, 0, 0, 0])
        for axis in ("instruction", "author", "framing", "channel", "assistant")
    }
    for study in studies:
        rejected = [index for index, spec in enumerate(study.inputs) if not by_spec[spec].complies]
        excluded = bool(rejected)
        trials = build_trials(study)
        comparisons.append(
            {
                "study_id": study_fingerprint(study),
                "assistant": str(study.assistant),
                "inputs": [prompt_spec_to_dict(spec) for spec in study.inputs],
                "cell_ids": [str(cell_id(Cell(spec, study.assistant))) for spec in study.inputs],
                "trial_ids": [str(trial.trial_id) for trial in trials],
                "excluded": excluded,
                "reason": "author_message_qa_failed" if excluded else None,
                "failed_input_indices": rejected,
            }
        )
        for axis, counts in comparison_counts.items():
            values = (
                {str(study.assistant)}
                if axis == "assistant"
                else {str(getattr(spec, axis)) for spec in study.inputs}
            )
            for value in values:
                row = counts[value]
                row[0] += 1
                row[1] += int(excluded)
                row[2] += len(trials)
                row[3] += len(trials) if excluded else 0

    return {
        "counts": {
            "unique_inputs": len(specs),
            "failed_inputs": sum(not by_spec[spec].complies for spec in specs),
            "planned_comparisons": len(studies),
            "excluded_comparisons": sum(row["excluded"] for row in comparisons),
            "planned_trials": sum(len(row["trial_ids"]) for row in comparisons),
            "excluded_trials": sum(len(row["trial_ids"]) for row in comparisons if row["excluded"]),
        },
        "inputs_by_axis": {
            axis: [
                {axis: value, "inputs": counts[0], "failed_inputs": counts[1]}
                for value, counts in sorted(groups.items())
            ]
            for axis, groups in input_counts.items()
        },
        "comparisons_by_axis": {
            axis: [
                {
                    axis: value,
                    "planned_comparisons": counts[0],
                    "excluded_comparisons": counts[1],
                    "planned_trials": counts[2],
                    "excluded_trials": counts[3],
                }
                for value, counts in sorted(groups.items())
            ]
            for axis, groups in comparison_counts.items()
        },
        "inputs": inputs,
        "comparisons": comparisons,
    }
