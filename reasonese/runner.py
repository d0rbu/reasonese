"""Generate framed inputs, build a conversation, and run its assistant."""

from __future__ import annotations

from contextlib import ExitStack
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
from reasonese.openrouter import (
    JsonObject,
    ModelRoute,
    OpenRouterClient,
    OpenRouterModelId,
    model_route,
    response_content,
)
from reasonese.tools import (
    ASSISTANT_TOOLS,
    ToolRuntime,
    assistant_message_from_response,
    tool_calls_from_response,
)

_MAX_LOCAL_TOOL_STEPS = 8


def _assistant_request(messages: list[JsonObject]) -> JsonObject:
    return {
        "messages": list(messages),
        "tools": list(ASSISTANT_TOOLS),
        "parallel_tool_calls": False,
        "temperature": 0.7,
        "reasoning": {"enabled": True, "exclude": False},
    }


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
    route = ModelRoute(model_id, None)
    return run_assistants((setup,), route, client, prefer_batch=False)[0]


@beartype
def run_assistants(
    setups: tuple[ConversationSetup, ...],
    route: ModelRoute,
    client: OpenRouterClient,
    *,
    prefer_batch: bool,
) -> tuple[ConversationTrace, ...]:
    """Run many independent assistants, batching each active function-tool round."""
    if not setups:
        return ()
    messages = {index: setup.openrouter_messages() for index, setup in enumerate(setups)}
    steps = {index: [] for index in range(len(setups))}
    completed: dict[int, ConversationTrace] = {}
    pending = list(range(len(setups)))

    with ExitStack() as stack:
        runtimes = {
            index: stack.enter_context(ToolRuntime(setup.readme_contents()))
            for index, setup in enumerate(setups)
        }
        while pending:
            responses = client.complete_many(
                route,
                tuple(_assistant_request(messages[index]) for index in pending),
                prefer_batch=prefer_batch,
            )
            next_pending: list[int] = []
            for index, response in zip(pending, responses, strict=True):
                calls = tool_calls_from_response(response)
                if not calls:
                    completed[index] = ConversationTrace(
                        setups[index], response, tuple(steps[index])
                    )
                    continue
                if len(steps[index]) == _MAX_LOCAL_TOOL_STEPS:
                    raise RuntimeError(
                        f"assistant exceeded {_MAX_LOCAL_TOOL_STEPS} local tool-call steps"
                    )
                results = tuple(runtimes[index].execute(call) for call in calls)
                steps[index].append(ToolStep(response, results))
                messages[index].append(assistant_message_from_response(response))
                messages[index].extend(result.openrouter_dict() for result in results)
                next_pending.append(index)
            pending = next_pending
    return tuple(completed[index] for index in range(len(setups)))


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
