"""Generate framed inputs, build a conversation, and run its assistant."""

from __future__ import annotations

from dataclasses import dataclass

from beartype import beartype

from reasonese.axes import Author
from reasonese.cache import YamlMessageCache, YamlTraceCache
from reasonese.conversation import (
    ConversationTrace,
    GeneratedMessage,
    GeneratedText,
    authoring_request,
    construct_conversation,
)
from reasonese.matchup import Matchup
from reasonese.openrouter import OpenRouterClient, model_route, response_content


@beartype
@dataclass(frozen=True, slots=True)
class RunResult:
    """A conversation trace and whether it came entirely from cache."""

    trace: ConversationTrace
    cache_hit: bool


@beartype
def materialize_messages(
    matchup: Matchup,
    client: OpenRouterClient,
    cache: YamlMessageCache,
    *,
    prefer_batch: bool,
) -> tuple[GeneratedMessage, ...]:
    """Generate each distinct uncached input, grouped by author model."""
    materialized = {message.spec: message for message in cache.load()}
    missing = tuple(dict.fromkeys(spec for spec in matchup.inputs if spec not in materialized))
    new_messages: list[GeneratedMessage] = []

    for spec in missing:
        if spec.author is Author.USER:
            message = GeneratedMessage(spec, GeneratedText.parse(str(spec.instruction)), None)
            materialized[spec] = message
            new_messages.append(message)

    model_authors = tuple(author for author in Author if author is not Author.USER)
    for author in model_authors:
        authored_specs = tuple(spec for spec in missing if spec.author is author)
        if not authored_specs:
            continue
        responses = client.complete_many(
            model_route(author),
            tuple(authoring_request(spec) for spec in authored_specs),
            prefer_batch=prefer_batch,
        )
        for spec, response in zip(authored_specs, responses, strict=True):
            message = GeneratedMessage(
                spec,
                GeneratedText.parse(response_content(response)),
                response,
            )
            materialized[spec] = message
            new_messages.append(message)

    if new_messages:
        cache.put_many(tuple(new_messages))
    return tuple(materialized[spec] for spec in matchup.inputs)


@beartype
def run_matchup(
    matchup: Matchup,
    client: OpenRouterClient,
    message_cache: YamlMessageCache,
    trace_cache: YamlTraceCache,
    *,
    prefer_batch: bool,
) -> RunResult:
    """Return a cached trace or execute the complete matchup through OpenRouter."""
    cached = trace_cache.get(matchup)
    if cached is not None:
        return RunResult(cached, True)

    generated = materialize_messages(
        matchup,
        client,
        message_cache,
        prefer_batch=prefer_batch,
    )
    setup = construct_conversation(matchup, generated)
    response = client.complete(
        model_route(matchup.assistant).model_id,
        {
            "messages": setup.openrouter_messages(),
            "reasoning": {"enabled": True, "exclude": False},
        },
    )
    trace = ConversationTrace(setup, response)
    trace_cache.put(trace)
    return RunResult(trace, False)
