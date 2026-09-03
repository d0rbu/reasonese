"""Curate a candidate instruction-pair bank and write a spot-check report."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from beartype import beartype
from phantom.interval import Natural

from reasonese.instructions import (
    InstructionPair,
    PairId,
    SimilarityRow,
    coverage,
    cross_pair_similarities,
    load_instruction_pairs,
    scaffold_manual_variants,
)
from reasonese.openrouter import OpenRouterClient, RequestsTransport
from reasonese.pair_check_cache import YamlPairCheckCache
from reasonese.pair_checks import MAX_DIFFICULTY, MIN_DIFFICULTY, PairCheck, check_pairs

DEFAULT_SIMILARITY_THRESHOLD = 0.6


@beartype
@dataclass(frozen=True, slots=True)
class CurationResult:
    """Loaded pairs, their audits when run, and deterministic diversity diagnostics."""

    pairs: tuple[InstructionPair, ...]
    checks: tuple[PairCheck, ...]
    similarities: tuple[SimilarityRow, ...]
    check_cache_hits: Natural
    similarity_threshold: float

    @property
    def failing_pair_ids(self) -> tuple[PairId, ...]:
        """Return identifiers of audited pairs that failed any bank criterion."""
        return tuple(check.pair.pair_id for check in self.checks if not check.passes)

    @property
    def overlapping_rows(self) -> tuple[SimilarityRow, ...]:
        """Return instructions whose nearest other-pair instruction exceeds the threshold."""
        return tuple(
            row for row in self.similarities if row.similarity >= self.similarity_threshold
        )

    @property
    def passes(self) -> bool:
        """Return whether every audited pair satisfied the bank criteria."""
        return not self.failing_pair_ids


@beartype
def audit_pairs(
    pairs: tuple[InstructionPair, ...],
    cache: YamlPairCheckCache,
    client: OpenRouterClient | None,
    *,
    prefer_batch: bool = True,
) -> tuple[tuple[PairCheck, ...], Natural]:
    """Audit pairs, judging only those whose exact texts have no cached audit."""
    cached = cache.load()
    check_by_id: dict[PairId, PairCheck] = {}
    for pair in pairs:
        match = next((check for check in cached if check.matches(pair)), None)
        if match is not None:
            check_by_id[pair.pair_id] = match
    missing = tuple(pair for pair in pairs if pair.pair_id not in check_by_id)
    if missing:
        if client is None:
            raise ValueError("OPENROUTER_API_KEY is required for uncached pair checks")
        new_checks = check_pairs(missing, client, prefer_batch=prefer_batch)
        cache.put_many(new_checks)
        check_by_id.update({check.pair.pair_id: check for check in new_checks})
    return (
        tuple(check_by_id[pair.pair_id] for pair in pairs),
        Natural.parse(len(pairs) - len(missing)),
    )


@beartype
def curate(
    pairs: tuple[InstructionPair, ...],
    cache: YamlPairCheckCache,
    client: OpenRouterClient | None,
    *,
    run_checks: bool,
    similarity_threshold: float,
    prefer_batch: bool = True,
) -> CurationResult:
    """Run deterministic diagnostics and, when requested, the cached LLM audit."""
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity threshold must be between 0 and 1")
    checks: tuple[PairCheck, ...] = ()
    hits = Natural.parse(0)
    if run_checks:
        checks, hits = audit_pairs(pairs, cache, client, prefer_batch=prefer_batch)
    return CurationResult(
        pairs,
        checks,
        cross_pair_similarities(pairs),
        hits,
        similarity_threshold,
    )


def _escape(text: str) -> str:
    return text.replace("|", "\\|")


@beartype
def write_report(path: Path, result: CurationResult) -> None:
    """Write a Markdown report for manual spot-checking of the candidate bank."""
    check_by_id = {check.pair.pair_id: check for check in result.checks}
    lines = [
        "# Instruction bank curation",
        "",
        "## Summary",
        "",
        f"- Pairs: {len(result.pairs)}",
        f"- Audited pairs: {len(result.checks)}",
        f"- Audit cache hits: {int(result.check_cache_hits)}",
        f"- Failing pairs: {len(result.failing_pair_ids)}",
        f"- Acceptable difficulty band: {MIN_DIFFICULTY} to {MAX_DIFFICULTY}",
        f"- Lexical overlap threshold: {result.similarity_threshold}",
        f"- Instructions over the overlap threshold: {len(result.overlapping_rows)}",
        "",
        "## Coverage",
        "",
        "| Skill | Conflict | Pairs |",
        "|---|---|---:|",
    ]
    for (skill, conflict), count in sorted(coverage(result.pairs).items()):
        lines.append(f"| {skill} | {conflict} | {count} |")

    lines.extend(
        [
            "",
            "## Audit",
            "",
            "| Pair | Skill | Conflict | Verdict | Difficulty | Reasons |",
            "|---|---|---|---|---|---|",
        ]
    )
    for pair in result.pairs:
        check = check_by_id.get(pair.pair_id)
        if check is None:
            verdict, difficulty, reasons = "not audited", "", ""
        else:
            verdict = "pass" if check.passes else "FAIL"
            difficulty = f"{int(check.first.difficulty)} / {int(check.second.difficulty)}"
            reasons = "; ".join((*check.failure_reasons(), *(str(i) for i in check.issues)))
        lines.append(
            f"| `{pair.pair_id}` | {pair.skill} | {pair.conflict} | {verdict} | {difficulty} | "
            f"{_escape(reasons)} |"
        )

    lines.extend(
        [
            "",
            "## Lexical overlap across pairs",
            "",
            "Rows list each instruction's nearest instruction in a different pair.",
            "",
            "| Pair | Slot | Nearest pair | Jaccard |",
            "|---|---|---|---:|",
        ]
    )
    for row in result.similarities:
        flag = " (over threshold)" if row.similarity >= result.similarity_threshold else ""
        lines.append(
            f"| `{row.pair_id}` | {row.slot} | `{row.nearest_pair_id}` | "
            f"{row.similarity:.2f}{flag} |"
        )

    lines.extend(["", "## Spot check", ""])
    for pair in result.pairs:
        lines.extend(
            [
                f"### `{pair.pair_id}`",
                "",
                f"- Skill: {pair.skill}",
                f"- Conflict: {pair.conflict}",
                f"- Rationale: {pair.rationale}",
                "",
                f"**First.** {pair.first}",
                "",
                f"**Second.** {pair.second}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


@beartype
def main(argv: Sequence[str] | None = None) -> int:
    """Audit a candidate instruction-pair bank and fail when any pair misses a criterion."""
    parser = argparse.ArgumentParser(prog="reasonese-curate-instructions")
    parser.add_argument("--pairs", type=Path, default=Path("configs/instruction_pairs.yaml"))
    parser.add_argument("--output", type=Path, default=Path("out/instructions"))
    parser.add_argument("--check-cache", type=Path, default=None)
    parser.add_argument("--no-checks", action="store_true")
    parser.add_argument("--no-batch", action="store_true")
    parser.add_argument(
        "--similarity-threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD
    )
    parser.add_argument("--scaffold-user-prompts", type=Path, default=None)
    args = parser.parse_args(argv)

    cache_path = args.check_cache if args.check_cache is not None else args.output / "pair_checks.yaml"
    try:
        pairs = load_instruction_pairs(args.pairs)
        api_key = os.environ.get("OPENROUTER_API_KEY")
        client = OpenRouterClient(RequestsTransport(api_key)) if api_key is not None else None
        result = curate(
            pairs,
            YamlPairCheckCache(cache_path),
            client,
            run_checks=not args.no_checks,
            similarity_threshold=args.similarity_threshold,
            prefer_batch=not args.no_batch,
        )
        write_report(args.output / "report.md", result)
        scaffolded: tuple[Path, ...] = ()
        if args.scaffold_user_prompts is not None:
            scaffolded = scaffold_manual_variants(args.scaffold_user_prompts, pairs)
    except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as error:
        parser.error(str(error))

    print(
        json.dumps(
            {
                "audited": len(result.checks),
                "check_cache_hits": int(result.check_cache_hits),
                "failing": [str(pair_id) for pair_id in result.failing_pair_ids],
                "over_overlap_threshold": len(result.overlapping_rows),
                "pairs": len(pairs),
                "report": str(args.output / "report.md"),
                "scaffolded": [str(path) for path in scaffolded],
            },
            sort_keys=True,
        )
    )
    return 0 if result.passes else 1
