"""Judge one cached matchup response through GPT-5.6 Luna batch."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from beartype import beartype

from reasonese.cache import YamlTraceCache
from reasonese.config import load_matchup
from reasonese.judging import Judgment, judge_trace
from reasonese.judgment_cache import YamlJudgmentCache
from reasonese.openrouter import OpenRouterClient, RequestsTransport


@beartype
@dataclass(frozen=True, slots=True)
class JudgeRunResult:
    """A cached or newly generated judgment."""

    judgment: Judgment
    cache_hit: bool


@beartype
def run_judge(
    trace_cache: YamlTraceCache,
    judgment_cache: YamlJudgmentCache,
    matchup_path: Path,
    client: OpenRouterClient | None,
) -> JudgeRunResult:
    """Load one trace, return its cached judgment, or judge it."""
    matchup = load_matchup(matchup_path)
    trace = trace_cache.get(matchup)
    if trace is None:
        raise ValueError("trace cache does not contain the requested matchup")
    cached = judgment_cache.get(trace)
    if cached is not None:
        return JudgeRunResult(cached, True)
    if client is None:
        raise ValueError("OPENROUTER_API_KEY is required for an uncached judgment")
    judgment = judge_trace(trace, client)
    judgment_cache.put(judgment)
    return JudgeRunResult(judgment, False)


@beartype
def main(argv: Sequence[str] | None = None) -> int:
    """Judge every instruction in one cached conversation trace."""
    parser = argparse.ArgumentParser(prog="reasonese-judge-responses")
    parser.add_argument("--matchup", type=Path, required=True)
    parser.add_argument("--trace-cache", type=Path, default=Path("out/conversation_traces.yaml"))
    parser.add_argument("--judgment-cache", type=Path, default=Path("out/judgments.yaml"))
    args = parser.parse_args(argv)

    try:
        trace_cache = YamlTraceCache(args.trace_cache)
        judgment_cache = YamlJudgmentCache(args.judgment_cache)
        api_key = os.environ.get("OPENROUTER_API_KEY")
        client = OpenRouterClient(RequestsTransport(api_key)) if api_key is not None else None
        result = run_judge(trace_cache, judgment_cache, args.matchup, client)
    except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as error:
        parser.error(str(error))

    print(
        json.dumps(
            {
                "cache_hit": result.cache_hit,
                "completed": [verdict.completed for verdict in result.judgment.verdicts],
                "judgment_cache": str(args.judgment_cache),
                "judge": "openai/gpt-5.6-luna:batch",
            },
            sort_keys=True,
        )
    )
    return 0
