"""Generate framed inputs, build a conversation, and run its assistant."""

from __future__ import annotations

from dataclasses import dataclass

from beartype import beartype

from reasonese.axes import Author
from reasonese.cache import YamlMessageCache, YamlTraceCache
from reasonese.conversation import (
    ConversationSetup,
    ConversationTrace,
    GeneratedMessage,
    GeneratedText,
    ToolStep,
    authoring_request,
    construct_conversation,
)
from reasonese.manual_messages import ManualMessageLibrary
from reasonese.matchup import Matchup
from reasonese.openrouter import OpenRouterClient, OpenRouterModelId, model_route, response_content
from reasonese.tools import (
    ASSISTANT_TOOLS,
    ToolRuntime,
    assistant_message_from_response,
    tool_calls_from_response,
)

_MAX_LOCAL_TOOL_STEPS = 8


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
    manual_messages: ManualMessageLibrary,
    *,
    prefer_batch: bool,
) -> tuple[GeneratedMessage, ...]:
    """Generate each distinct uncached input, grouped by author model."""
    materialized = {message.spec: message for message in cache.load()}
    new_messages: list[GeneratedMessage] = []

    user_specs = tuple(dict.fromkeys(spec for spec in matchup.inputs if spec.author is Author.USER))
    for spec in user_specs:
        message = GeneratedMessage(spec, manual_messages.message_for(spec), None)
        if materialized.get(spec) != message:
            materialized[spec] = message
            new_messages.append(message)

    missing = tuple(dict.fromkeys(spec for spec in matchup.inputs if spec not in materialized))

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
def run_assistant(
    setup: ConversationSetup,
    model_id: OpenRouterModelId,
    client: OpenRouterClient,
) -> ConversationTrace:
    """Run one assistant, executing bounded local function calls until it answers."""
    messages = setup.openrouter_messages()
    steps: list[ToolStep] = []
    with ToolRuntime(setup.readme_contents()) as tools:
        for _ in range(_MAX_LOCAL_TOOL_STEPS + 1):
            response = client.complete(
                model_id,
                {
                    "messages": messages,
                    "tools": list(ASSISTANT_TOOLS),
                    "parallel_tool_calls": False,
                    "temperature": 0.7,
                    "reasoning": {"enabled": True, "exclude": False},
                },
            )
            calls = tool_calls_from_response(response)
            if not calls:
                return ConversationTrace(setup, response, tuple(steps))
            if len(steps) == _MAX_LOCAL_TOOL_STEPS:
                raise RuntimeError(
                    f"assistant exceeded {_MAX_LOCAL_TOOL_STEPS} local tool-call steps"
                )
            results = tuple(tools.execute(call) for call in calls)
            steps.append(ToolStep(response, results))
            messages.append(assistant_message_from_response(response))
            messages.extend(result.openrouter_dict() for result in results)
    raise AssertionError("bounded assistant loop ended unexpectedly")


@beartype
def run_matchup(
    matchup: Matchup,
    client: OpenRouterClient,
    message_cache: YamlMessageCache,
    trace_cache: YamlTraceCache,
    manual_messages: ManualMessageLibrary,
    *,
    prefer_batch: bool,
) -> RunResult:
    """Return a cached trace or execute the complete matchup through OpenRouter."""
    cached = trace_cache.get(matchup)
    if cached is not None and manual_messages.matches(cached.setup):
        return RunResult(cached, True)

    generated = materialize_messages(
        matchup,
        client,
        message_cache,
        manual_messages,
        prefer_batch=prefer_batch,
    )
    setup = construct_conversation(matchup, generated)
    trace = run_assistant(setup, model_route(matchup.assistant).model_id, client)
    trace_cache.put(trace)
    return RunResult(trace, False)
