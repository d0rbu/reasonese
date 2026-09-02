"""Run one YAML matchup through OpenRouter."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from beartype import beartype

from reasonese.cache import YamlMessageCache, YamlTraceCache
from reasonese.check_messages import audit_messages, require_compliant_messages
from reasonese.config import load_matchup
from reasonese.conversation import GeneratedMessage
from reasonese.manual_messages import ManualMessageLibrary
from reasonese.message_qa_cache import YamlMessageQaCache
from reasonese.openrouter import OpenRouterClient, RequestsTransport
from reasonese.runner import run_matchup


@beartype
def main(argv: Sequence[str] | None = None) -> int:
    """Materialize, execute, and cache one matchup."""
    parser = argparse.ArgumentParser(prog="reasonese-run-conversation")
    parser.add_argument("--matchup", type=Path, required=True)
    parser.add_argument("--message-cache", type=Path, default=Path("out/generated_messages.yaml"))
    parser.add_argument("--message-qa-cache", type=Path, default=Path("out/message_qa.yaml"))
    parser.add_argument("--trace-cache", type=Path, default=Path("out/conversation_traces.yaml"))
    parser.add_argument("--user-messages", type=Path, default=Path("prompts/user"))
    parser.add_argument("--no-batch", action="store_true")
    args = parser.parse_args(argv)

    try:
        matchup = load_matchup(args.matchup)
        message_cache = YamlMessageCache(args.message_cache)
        qa_cache = YamlMessageQaCache(args.message_qa_cache)
        trace_cache = YamlTraceCache(args.trace_cache)
        manual_messages = ManualMessageLibrary(args.user_messages)
        api_key = os.environ.get("OPENROUTER_API_KEY")
        client = OpenRouterClient(RequestsTransport(api_key)) if api_key is not None else None
        cached = trace_cache.get(matchup)
        if cached is not None and manual_messages.matches(cached.setup):
            cached_messages = tuple(
                GeneratedMessage(spec, cached.setup.content_for_input(index), None)
                for index, spec in enumerate(matchup.inputs)
            )
            require_compliant_messages(audit_messages(cached_messages, qa_cache, client))
            result = {
                "assistant": str(matchup.assistant),
                "cache_hit": True,
                "message_qa_cache": str(args.message_qa_cache),
                "messages": len(cached.setup.messages),
                "response_id": cached.response.get("id"),
                "trace_cache": str(args.trace_cache),
            }
            print(json.dumps(result, sort_keys=True))
            return 0

        if client is None:
            raise ValueError("OPENROUTER_API_KEY is required for an uncached matchup")
        run = run_matchup(
            matchup,
            client,
            message_cache,
            trace_cache,
            qa_cache,
            manual_messages,
            prefer_batch=not args.no_batch,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))

    print(
        json.dumps(
            {
                "assistant": str(matchup.assistant),
                "cache_hit": run.cache_hit,
                "message_qa_cache": str(args.message_qa_cache),
                "messages": len(run.trace.setup.messages),
                "response_id": run.trace.response.get("id"),
                "trace_cache": str(args.trace_cache),
            },
            sort_keys=True,
        )
    )
    return 0
