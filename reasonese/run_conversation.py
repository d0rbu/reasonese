"""Run one YAML matchup through OpenRouter."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from beartype import beartype

from reasonese.cache import YamlMessageCache, YamlTraceCache
from reasonese.config import load_matchup
from reasonese.openrouter import OpenRouterClient, RequestsTransport
from reasonese.runner import run_matchup


@beartype
def main(argv: Sequence[str] | None = None) -> int:
    """Materialize, execute, and cache one matchup."""
    parser = argparse.ArgumentParser(prog="reasonese-run-conversation")
    parser.add_argument("--matchup", type=Path, required=True)
    parser.add_argument("--message-cache", type=Path, default=Path("out/generated_messages.yaml"))
    parser.add_argument("--trace-cache", type=Path, default=Path("out/conversation_traces.yaml"))
    parser.add_argument("--no-batch", action="store_true")
    args = parser.parse_args(argv)

    try:
        matchup = load_matchup(args.matchup)
        message_cache = YamlMessageCache(args.message_cache)
        trace_cache = YamlTraceCache(args.trace_cache)
        cached = trace_cache.get(matchup)
        if cached is not None:
            result = {
                "assistant": str(matchup.assistant),
                "cache_hit": True,
                "messages": len(cached.setup.messages),
                "response_id": cached.response.get("id"),
                "trace_cache": str(args.trace_cache),
            }
            print(json.dumps(result, sort_keys=True))
            return 0

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if api_key is None:
            raise ValueError("OPENROUTER_API_KEY is required for an uncached matchup")
        client = OpenRouterClient(RequestsTransport(api_key))
        run = run_matchup(
            matchup,
            client,
            message_cache,
            trace_cache,
            prefer_batch=not args.no_batch,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))

    print(
        json.dumps(
            {
                "assistant": str(matchup.assistant),
                "cache_hit": run.cache_hit,
                "messages": len(run.trace.setup.messages),
                "response_id": run.trace.response.get("id"),
                "trace_cache": str(args.trace_cache),
            },
            sort_keys=True,
        )
    )
    return 0
