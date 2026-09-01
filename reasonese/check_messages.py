"""Audit cached materialized messages through GPT-5.6 Luna batch."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from beartype import beartype
from phantom.interval import Natural

from reasonese.cache import YamlMessageCache
from reasonese.conversation import GeneratedMessage
from reasonese.message_qa import MessageQaVerdict, check_messages
from reasonese.message_qa_cache import YamlMessageQaCache
from reasonese.openrouter import OpenRouterClient, RequestsTransport
from reasonese.planning import PromptSpec


@beartype
@dataclass(frozen=True, slots=True)
class MessageQaRunResult:
    """Ordered QA verdicts and the number reused from cache."""

    verdicts: tuple[MessageQaVerdict, ...]
    cache_hits: Natural


@beartype
def run_message_qa(
    message_cache: YamlMessageCache,
    qa_cache: YamlMessageQaCache,
    client: OpenRouterClient | None,
) -> MessageQaRunResult:
    """Load and audit every message in one generated-message cache."""
    messages = message_cache.load()
    if not messages:
        raise ValueError("message cache does not contain any generated messages")
    return audit_messages(messages, qa_cache, client)


@beartype
def audit_messages(
    messages: tuple[GeneratedMessage, ...],
    qa_cache: YamlMessageQaCache,
    client: OpenRouterClient | None,
) -> MessageQaRunResult:
    """Audit messages, judging only exact uncached text."""
    if not messages:
        raise ValueError("at least one generated message is required for QA")
    unique_by_spec: dict[PromptSpec, GeneratedMessage] = {}
    for message in messages:
        existing = unique_by_spec.get(message.spec)
        if existing is not None and existing.content != message.content:
            raise ValueError("one datapoint cannot have multiple message contents in one QA run")
        unique_by_spec[message.spec] = message
    unique_messages = tuple(unique_by_spec.values())
    cached_by_spec = {verdict.spec: verdict for verdict in qa_cache.load()}
    verdict_by_spec = {
        message.spec: cached
        for message in unique_messages
        if (cached := cached_by_spec.get(message.spec)) is not None and cached.matches(message)
    }
    missing = tuple(message for message in unique_messages if message.spec not in verdict_by_spec)
    if missing:
        if client is None:
            raise ValueError("OPENROUTER_API_KEY is required for uncached message QA")
        new_verdicts = check_messages(missing, client)
        qa_cache.put_many(new_verdicts)
        verdict_by_spec.update({verdict.spec: verdict for verdict in new_verdicts})
    return MessageQaRunResult(
        tuple(verdict_by_spec[message.spec] for message in messages),
        Natural.parse(len(unique_messages) - len(missing)),
    )


@beartype
def require_compliant_messages(result: MessageQaRunResult) -> None:
    """Fail closed when any audited message does not satisfy its datapoint instructions."""
    failures = tuple(verdict for verdict in result.verdicts if not verdict.complies)
    if failures:
        coordinates = "; ".join(
            f"{verdict.spec.instruction!s} [{verdict.spec.framing}, {verdict.spec.channel}, "
            f"{verdict.spec.author}]"
            for verdict in failures
        )
        raise ValueError(f"message QA failed for: {coordinates}")


@beartype
def main(argv: Sequence[str] | None = None) -> int:
    """Audit cached messages and fail when any message is noncompliant."""
    parser = argparse.ArgumentParser(prog="reasonese-check-messages")
    parser.add_argument("--message-cache", type=Path, default=Path("out/generated_messages.yaml"))
    parser.add_argument("--qa-cache", type=Path, default=Path("out/message_qa.yaml"))
    args = parser.parse_args(argv)

    try:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        client = OpenRouterClient(RequestsTransport(api_key)) if api_key is not None else None
        result = run_message_qa(
            YamlMessageCache(args.message_cache),
            YamlMessageQaCache(args.qa_cache),
            client,
        )
    except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as error:
        parser.error(str(error))

    print(
        json.dumps(
            {
                "cache_hits": int(result.cache_hits),
                "complies": [verdict.complies for verdict in result.verdicts],
                "judge": "openai/gpt-5.6-luna:batch",
                "messages": len(result.verdicts),
                "qa_cache": str(args.qa_cache),
            },
            sort_keys=True,
        )
    )
    return 0 if all(verdict.complies for verdict in result.verdicts) else 1
