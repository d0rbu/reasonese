"""Generate framed inputs, build a conversation, and run its assistant."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import ExitStack
from dataclasses import dataclass

from beartype import beartype

from reasonese.axes import Author
from reasonese.cache import YamlMessageCache, YamlTraceCache
from reasonese.check_messages import audit_messages, require_compliant_messages
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
from reasonese.message_qa_cache import YamlMessageQaCache
from reasonese.openrouter import (
    CompletionGroup,
    JsonObject,
    ModelRoute,
    OpenRouterClient,
    OpenRouterModelId,
    model_route,
    response_content,
)
from reasonese.planning import PromptSpec
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
@dataclass(frozen=True, slots=True)
class AssistantRunGroup:
    """Independent conversation setups sharing one assistant model route."""

    route: ModelRoute
    setups: tuple[ConversationSetup, ...]


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
    return materialize_specs(
        matchup.inputs,
        client,
        cache,
        manual_messages,
        prefer_batch=prefer_batch,
    )


@beartype
def materialize_specs(
    specs: tuple[PromptSpec, ...],
    client: OpenRouterClient,
    cache: YamlMessageCache,
    manual_messages: ManualMessageLibrary,
    *,
    prefer_batch: bool,
) -> tuple[GeneratedMessage, ...]:
    """Materialize arbitrary prompt specs with one shared model-grouped cache pass."""
    materialized = {message.spec: message for message in cache.load()}
    new_messages: list[GeneratedMessage] = []

    user_specs = tuple(dict.fromkeys(spec for spec in specs if spec.author is Author.USER))
    for spec in user_specs:
        message = GeneratedMessage(spec, manual_messages.message_for(spec), None)
        if materialized.get(spec) != message:
            materialized[spec] = message
            new_messages.append(message)

    missing = tuple(dict.fromkeys(spec for spec in specs if spec not in materialized))

    grouped_specs: list[tuple[PromptSpec, ...]] = []
    completion_groups: list[CompletionGroup] = []
    model_authors = tuple(author for author in Author if author is not Author.USER)
    for author in model_authors:
        authored_specs = tuple(spec for spec in missing if spec.author is author)
        if not authored_specs:
            continue
        grouped_specs.append(authored_specs)
        completion_groups.append(
            CompletionGroup(
                model_route(author),
                tuple(authoring_request(spec) for spec in authored_specs),
            )
        )

    grouped_responses = client.complete_many_grouped(
        tuple(completion_groups),
        prefer_batch=prefer_batch,
    )
    for authored_specs, responses in zip(grouped_specs, grouped_responses, strict=True):
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
    return tuple(materialized[spec] for spec in specs)


@beartype
def run_assistant(
    setup: ConversationSetup,
    model_id: OpenRouterModelId,
    client: OpenRouterClient,
) -> ConversationTrace:
    """Run one assistant, executing bounded local function calls until it answers."""
    route = ModelRoute(model_id, None)
    return run_assistants((setup,), route, client)[0]


@beartype
def run_assistants(
    setups: tuple[ConversationSetup, ...],
    route: ModelRoute,
    client: OpenRouterClient,
) -> tuple[ConversationTrace, ...]:
    """Run many independent assistants through completion-driven tool loops."""
    return run_assistant_groups((AssistantRunGroup(route, setups),), client)[0]


@beartype
def run_assistant_groups(
    groups: tuple[AssistantRunGroup, ...],
    client: OpenRouterClient,
) -> tuple[tuple[ConversationTrace, ...], ...]:
    """Run assistant-model groups without blocking fast tool continuations on slow peers."""
    messages = {
        (group_index, setup_index): setup.openrouter_messages()
        for group_index, group in enumerate(groups)
        for setup_index, setup in enumerate(group.setups)
    }
    steps: dict[tuple[int, int], list[ToolStep]] = {key: [] for key in messages}
    completed: dict[tuple[int, int], ConversationTrace] = {}
    if not messages:
        return tuple(() for _ in groups)

    with ExitStack() as stack:
        runtimes = {
            (group_index, setup_index): stack.enter_context(ToolRuntime(setup.readme_contents()))
            for group_index, group in enumerate(groups)
            for setup_index, setup in enumerate(group.setups)
        }
        with ThreadPoolExecutor(max_workers=min(client.sync_workers, len(messages))) as executor:

            def submit(key: tuple[int, int]) -> Future[JsonObject]:
                group_index, _ = key
                return executor.submit(
                    client.complete,
                    groups[group_index].route.model_id,
                    _assistant_request(messages[key]),
                )

            pending = {submit(key): key for key in messages}
            while pending:
                finished, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                for future in finished:
                    key = pending.pop(future)
                    group_index, setup_index = key
                    response = future.result()
                    calls = tool_calls_from_response(response)
                    if not calls:
                        completed[key] = ConversationTrace(
                            groups[group_index].setups[setup_index],
                            response,
                            tuple(steps[key]),
                        )
                        continue
                    if len(steps[key]) == _MAX_LOCAL_TOOL_STEPS:
                        raise RuntimeError(
                            f"assistant exceeded {_MAX_LOCAL_TOOL_STEPS} local tool-call steps"
                        )
                    results = tuple(runtimes[key].execute(call) for call in calls)
                    steps[key].append(ToolStep(response, results))
                    messages[key].append(assistant_message_from_response(response))
                    messages[key].extend(result.openrouter_dict() for result in results)
                    pending[submit(key)] = key
    return tuple(
        tuple(completed[(group_index, setup_index)] for setup_index in range(len(group.setups)))
        for group_index, group in enumerate(groups)
    )


@beartype
def run_matchup(
    matchup: Matchup,
    client: OpenRouterClient,
    message_cache: YamlMessageCache,
    trace_cache: YamlTraceCache,
    qa_cache: YamlMessageQaCache,
    manual_messages: ManualMessageLibrary,
    *,
    prefer_batch: bool,
) -> RunResult:
    """Return a cached trace or execute the complete matchup through OpenRouter."""
    cached = trace_cache.get(matchup)
    if cached is not None and manual_messages.matches(cached.setup):
        cached_messages = tuple(
            GeneratedMessage(spec, cached.setup.content_for_input(index), None)
            for index, spec in enumerate(matchup.inputs)
        )
        require_compliant_messages(audit_messages(cached_messages, qa_cache, client))
        return RunResult(cached, True)

    generated = materialize_messages(
        matchup,
        client,
        message_cache,
        manual_messages,
        prefer_batch=prefer_batch,
    )
    require_compliant_messages(audit_messages(generated, qa_cache, client))
    setup = construct_conversation(matchup, generated)
    trace = run_assistant(setup, model_route(matchup.assistant).model_id, client)
    trace_cache.put(trace)
    return RunResult(trace, False)
