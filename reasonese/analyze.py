"""Write Bradley-Terry, axis, and order-effect analyses for collected data."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path

from beartype import beartype

from reasonese.analysis import AnalysisBundle, RankedCell, TableRow, analyze_observations
from reasonese.instructions import (
    PairMembership,
    instruction_index,
    load_instruction_pairs,
)
from reasonese.observations import Observation, load_observations


def _ranking_row(ranked: RankedCell, membership: PairMembership) -> TableRow:
    return {
        "component": ranked.component_index,
        "rank": ranked.rank,
        "cell_id": str(ranked.cell_id),
        "pair": str(membership.pair.pair_id),
        "side": str(membership.side),
        "skill": str(membership.pair.skill),
        "conflict": str(membership.pair.conflict),
        "instruction": str(ranked.cell.spec.instruction),
        "framing": str(ranked.cell.spec.framing),
        "channel": str(ranked.cell.spec.channel),
        "author": str(ranked.cell.spec.author),
        "assistant": str(ranked.cell.assistant),
        "bt_score": ranked.score,
        "standard_error": ranked.standard_error,
        "bootstrap_low": ranked.bootstrap_low,
        "bootstrap_high": ranked.bootstrap_high,
        "observations": ranked.observations,
        "completions": ranked.completions,
        "completion_rate": ranked.completion_rate,
    }


def _write_csv(path: Path, rows: tuple[TableRow, ...]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_float(value: object) -> str:
    if value is None:
        return "NA"
    if not isinstance(value, int | float):
        raise TypeError("expected a numeric report value")
    return f"{float(value):.4f}"


def _write_report(
    path: Path,
    bundle: AnalysisBundle,
    l2: float,
    index: dict[str, PairMembership],
) -> None:
    lines = [
        "# reasonese analysis",
        "",
        "## Summary",
        "",
        f"- Cells: {bundle.diagnostics['cells']}",
        f"- Trials: {bundle.diagnostics['trials']}",
        f"- Observations: {bundle.diagnostics['observations']}",
        f"- Pairwise comparisons: {bundle.fit.comparison_count}",
        f"- Tied comparisons (half-win each): {bundle.fit.tie_count}",
        f"- L2 penalty: {l2}",
        f"- Components: {len(bundle.fit.connected_components)}",
        "- Components match (pair, assistant): "
        f"{bundle.diagnostics['components_match_pair_assistant']}",
        f"- Both-completed trials: {bundle.diagnostics['both_completed_trials']}",
        f"- Neither-completed trials: {bundle.diagnostics['neither_completed_trials']}",
        f"- Position counts balanced: {bundle.diagnostics['position_balanced']}",
        "",
        "Instruction is not a treatment axis. A trial only holds the two instructions of one "
        "mutually exclusive pair, so the comparison graph has one component per (pair, "
        "assistant) and scores are identified only inside a component. The ranks below are "
        "within-component and carry no meaning across components. All-true and all-false "
        "within-trial pairs contribute 0.5 outcomes instead of being discarded, and bootstrap "
        "intervals resample whole trials.",
        "",
        "## Within-component cell ordering",
        "",
        "| Component | Rank | Cell | Pair | Side | Framing | Channel | Author | Assistant | BT score | 95% bootstrap | Completion |",
        "|---:|---:|---|---|---|---|---|---|---|---:|---:|---:|",
    ]
    for ranked in bundle.fit.ranking:
        membership = index[str(ranked.cell.spec.instruction)]
        interval = (
            f"{_format_float(ranked.bootstrap_low)} to {_format_float(ranked.bootstrap_high)}"
        )
        lines.append(
            f"| {ranked.component_index} | {ranked.rank} | `{ranked.cell_id}` | "
            f"{membership.pair.pair_id} | {membership.side} | "
            f"{ranked.cell.spec.framing} | {ranked.cell.spec.channel} | "
            f"{ranked.cell.spec.author} | {ranked.cell.assistant} | {ranked.score:.4f} | "
            f"{interval} | {ranked.completions}/{ranked.observations} |"
        )

    lines.extend(
        [
            "",
            "## Axis comparisons",
            "",
            "Marginal completion rates are descriptive; they do not by themselves isolate "
            "causal axis effects when the collected cells are not a balanced factorial design.",
            "",
            "| Axis | Value | Cells | Observations | Completion rate | Mean BT score |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in bundle.axis_summary:
        lines.append(
            f"| {row['axis']} | {row['value']} | {row['cells']} | {row['observations']} | "
            f"{_format_float(row['completion_rate'])} | "
            f"{_format_float(row['mean_bt_score'])} |"
        )

    lines.extend(
        [
            "",
            "## Strata",
            "",
            "Assistant, skill, conflict type, and pair are constant within a trial, so no "
            "Bradley-Terry contrast can separate their levels. Only descriptive completion "
            "rates are reported.",
            "",
            "| Stratum | Value | Cells | Observations | Completion rate | 95% Wilson |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in bundle.stratum_summary:
        interval = f"{_format_float(row['wilson_low'])} to {_format_float(row['wilson_high'])}"
        lines.append(
            f"| {row['stratum']} | {row['value']} | {row['cells']} | {row['observations']} | "
            f"{_format_float(row['completion_rate'])} | {interval} |"
        )

    lines.extend(
        [
            "",
            "## Pair exclusivity",
            "",
            "A pair is exclusive in practice when almost every trial completes exactly one of "
            "its two instructions. A high both-completed rate is a defect in the pair, since "
            "one response satisfied both. A high neither-completed rate means the trial was "
            "too hard rather than not exclusive. Bradley-Terry scores both as ties.",
            "",
            "| Pair | Skill | Conflict | Trials | Exactly one | Both | Neither |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in bundle.pair_exclusivity:
        lines.append(
            f"| {row['pair']} | {row['skill']} | {row['conflict']} | {row['trials']} | "
            f"{_format_float(row['exactly_one_rate'])} | "
            f"{_format_float(row['both_completed_rate'])} | "
            f"{_format_float(row['neither_completed_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## Strongest order sensitivities",
            "",
            "These are descriptive position-rate ranges and correlations, intended as sanity "
            "checks rather than multiplicity-corrected significance tests.",
            "",
            "| Kind | Axis/cell | Value | Observations | Rate range | Position correlation |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in bundle.order_sensitivity[:20]:
        label = row.get("axis", "cell")
        lines.append(
            f"| {row['kind']} | {label} | {row['value']} | {row['observations']} | "
            f"{_format_float(row['position_rate_range'])} | "
            f"{_format_float(row['position_correlation'])} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "See the CSV files for the complete axis contrasts, cell-by-position results, "
            "axis-by-position results, and regularization sensitivity. `diagnostics.json` "
            "contains comparison connectivity and per-cell position balance.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


@beartype
def write_analysis(
    output_dir: Path,
    bundle: AnalysisBundle,
    l2: float,
    index: dict[str, PairMembership],
) -> None:
    """Write all analysis tables, diagnostics, and a readable report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "ranking.csv",
        tuple(
            _ranking_row(ranked, index[str(ranked.cell.spec.instruction)])
            for ranked in bundle.fit.ranking
        ),
    )
    _write_csv(output_dir / "axis_summary.csv", bundle.axis_summary)
    _write_csv(output_dir / "axis_comparisons.csv", bundle.axis_comparisons)
    _write_csv(output_dir / "stratum_summary.csv", bundle.stratum_summary)
    _write_csv(output_dir / "pair_exclusivity.csv", bundle.pair_exclusivity)
    _write_csv(output_dir / "position_summary.csv", bundle.position_summary)
    _write_csv(output_dir / "cell_position_effects.csv", bundle.cell_position_effects)
    _write_csv(output_dir / "axis_position_effects.csv", bundle.axis_position_effects)
    _write_csv(output_dir / "order_sensitivity.csv", bundle.order_sensitivity)
    _write_csv(
        output_dir / "regularization_sensitivity.csv",
        bundle.regularization_sensitivity,
    )
    (output_dir / "diagnostics.json").write_text(
        json.dumps(bundle.diagnostics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir / "report.md", bundle, l2, index)


@beartype
def main(argv: Sequence[str] | None = None) -> int:
    """Analyze one or more collected observation files."""
    parser = argparse.ArgumentParser(prog="reasonese-analyze")
    parser.add_argument("--observations", type=Path, nargs="+", required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        observations: tuple[Observation, ...] = tuple(
            observation for path in args.observations for observation in load_observations(path)
        )
        pairs = load_instruction_pairs(args.pairs)
        index = {
            str(instruction): membership
            for instruction, membership in instruction_index(pairs).items()
        }
        bundle = analyze_observations(
            observations,
            pairs,
            args.l2,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        write_analysis(args.output, bundle, args.l2, index)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))

    print(
        json.dumps(
            {
                "both_completed_trials": bundle.diagnostics["both_completed_trials"],
                "cells": len(bundle.fit.ranking),
                "components": len(bundle.fit.connected_components),
                "components_match_pair_assistant": bundle.diagnostics[
                    "components_match_pair_assistant"
                ],
                "neither_completed_trials": bundle.diagnostics["neither_completed_trials"],
                "observations": len(observations),
                "output": str(args.output),
                "position_balanced": bundle.diagnostics["position_balanced"],
                "trials": bundle.diagnostics["trials"],
            },
            sort_keys=True,
        )
    )
    return 0
