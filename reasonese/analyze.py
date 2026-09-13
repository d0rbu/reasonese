"""Write Bradley-Terry, feature-lasso, axis, and order-effect analyses for collected data."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path

from beartype import beartype

from reasonese.analysis import (
    AnalysisBundle,
    RankedCell,
    TableRow,
    analyze_observations,
    pair_memberships,
)
from reasonese.instructions import (
    PairMembership,
    instruction_index,
    load_instruction_pairs,
)
from reasonese.lasso import (
    PATH_MIN_RATIO,
    FeatureLasso,
    fit_feature_lasso,
    lasso_diagnostics,
    lasso_tables,
    selected_features,
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


def _lasso_lines(lasso: FeatureLasso) -> list[str]:
    """Render the feature-lasso section: what was fitted, what entered, and when."""
    references = lasso.references
    tables = lasso_tables(lasso)
    fitted = [row for row in tables.features if row["status"] == "fitted"]
    never_differ = [row for row in tables.features if row["status"] == "never differs"]
    aliased = [row for row in tables.features if row["status"] == "aliased"]
    lines = [
        "",
        "## Feature lasso",
        "",
        "Every cell's strength is refitted on the same within-trial comparisons as a `(pair, "
        "assistant)` side offset plus a sparse sum of feature effects. Feature coefficients "
        "carry an L1 penalty, so a feature stays at zero until the comparisons support it; "
        "the offsets keep the L2 penalty. Features are treatment contrasts against "
        f"`{references['framing']}` framing, the `{references['channel']}` channel, the "
        f"`{references['author']}` author, and the `{references['assistant']}` assistant, "
        "plus `self_author`, `same_family`, `first_position`, and two-way interactions. "
        "Columns are not standardized, so a feature that rarely differs inside a trial needs "
        "a larger effect to enter.",
        "",
        f"- Comparisons: {lasso.comparisons}; blocks: {len(lasso.blocks)}",
        f"- Candidate features: {len(lasso.features)} ({len(fitted)} fitted, "
        f"{len(never_differ)} never differ inside a trial, {len(aliased)} aliased)",
        f"- Held-out loss is the mean log loss per comparison; offsets alone give "
        f"{_format_float(lasso.null_loss)} on the training data.",
    ]
    if lasso.design_rank < len(fitted):
        lines.append(
            f"- The fitted columns have rank {lasso.design_rank}, so they are linearly "
            "dependent: the fitted probabilities are unique but the coefficients are not, "
            "and the entry order can depend on the solver's path."
        )
    if not lasso.lambdas:
        lines.append(
            "- No penalty path was fitted: no fitted feature is correlated with the outcome "
            "once the offsets alone are fitted."
        )
        return lines

    lines.append(
        f"- Penalty path: {len(lasso.lambdas)} values from lambda_max = "
        f"{lasso.lambda_max:.4f} down to {PATH_MIN_RATIO:g} lambda_max."
    )
    validation = lasso.cross_validation
    selected = lasso.selected
    if validation is None or selected is None:
        lines.append(
            "- No cross-validation was run; the coefficients below are from the least "
            "penalized end of the path."
        )
    else:
        path_min = tables.path[validation.index_min]
        path_1se = tables.path[validation.index_1se]
        lines.append(
            f"- {validation.folds}-fold cross-validation: held-out loss "
            f"{_format_float(validation.mean_loss[0])} with offsets only, "
            f"{_format_float(path_min['cv_mean_loss'])} at lambda_min "
            f"({path_min['nonzero']} features), "
            f"{_format_float(path_1se['cv_mean_loss'])} at lambda_1se "
            f"({path_1se['nonzero']} features). The coefficients below are at lambda_1se, "
            "the largest penalty within one standard error of the minimum."
        )
    if not all(lasso.converged):
        lines.append("- At least one path point did not converge; see `lasso_path.csv`.")
    lines.extend(
        [
            "",
            "| Feature | Group | Enters at lambda/lambda_max | Coefficient (selected) | "
            "Coefficient (lambda_min) | Differs in |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    entered = sorted(
        (row for row in fitted if row["entry_index"] is not None),
        key=lambda row: (int(str(row["entry_index"])), str(row["feature"])),
    )
    for row in entered:
        lines.append(
            f"| `{row['feature']}` | {row['group']} | "
            f"{_format_float(row['entry_lambda_ratio'])} | "
            f"{_format_float(row['coefficient_selected'])} | "
            f"{_format_float(row['coefficient_cv_min'])} | "
            f"{row['differing_comparisons']} |"
        )
    missing = [str(row["feature"]) for row in fitted if row["entry_index"] is None]
    if missing:
        lines.extend(["", "Never enters: " + ", ".join(f"`{name}`" for name in missing) + "."])
    if aliased:
        pairs = ", ".join(
            f"`{row['feature']}` = {'-' if row['alias_sign'] == -1 else ''}`{row['alias_of']}`"
            for row in aliased
        )
        lines.extend(["", f"Aliased, identical inside every trial: {pairs}."])
    lines.extend(
        [
            "",
            "Lasso coefficients are shrunk toward zero and carry no standard errors. Read the "
            "entry order and the cross-validation curve as a guide to which contrasts deserve "
            "a closer look, not as tests. `lasso_features.csv` lists every candidate, and "
            "`lasso_blocks.csv` gives each pair's side offset under each assistant.",
        ]
    )
    return lines


def _write_report(
    path: Path,
    bundle: AnalysisBundle,
    l2: float,
    index: dict[str, PairMembership],
    lasso: FeatureLasso,
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

    lines.extend(_lasso_lines(lasso))

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
            "axis-by-position results, regularization sensitivity, and the full lasso path. "
            "`diagnostics.json` contains comparison connectivity, per-cell position balance, "
            "and the lasso's screening and cross-validation summary.",
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
    lasso: FeatureLasso,
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
    tables = lasso_tables(lasso)
    _write_csv(output_dir / "lasso_path.csv", tables.path)
    _write_csv(output_dir / "lasso_coefficients.csv", tables.coefficients)
    _write_csv(output_dir / "lasso_features.csv", tables.features)
    _write_csv(output_dir / "lasso_blocks.csv", tables.blocks)
    diagnostics = {**bundle.diagnostics, "feature_lasso": lasso_diagnostics(lasso)}
    (output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir / "report.md", bundle, l2, index, lasso)


@beartype
def main(argv: Sequence[str] | None = None) -> int:
    """Analyze one or more collected observation files."""
    parser = argparse.ArgumentParser(prog="reasonese-analyze")
    parser.add_argument("--observations", type=Path, nargs="+", required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--lasso-folds", type=int, default=5)
    parser.add_argument("--lasso-path-length", type=int, default=40)
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
        lasso = fit_feature_lasso(
            observations,
            pair_memberships(observations, pairs),
            args.l2,
            folds=args.lasso_folds,
            path_length=args.lasso_path_length,
            seed=args.seed,
        )
        write_analysis(args.output, bundle, args.l2, index, lasso)
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
                "lasso_selected_features": (
                    None if lasso.selected is None else len(selected_features(lasso))
                ),
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
