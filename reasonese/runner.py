"""Generate framed inputs, build a conversation, and run its assistant."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from threading import Lock

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
from reasonese.manual_messages import ManualMessageLibrary, ManualMessageSnapshot
from reasonese.matchup import Matchup
from reasonese.message_qa_cache import YamlMessageQaCache
from reasonese.openrouter import (
    CompletionGroup,
    CompletionTransport,
    JsonObject,
    ModelRoute,
    OpenRouterClient,
    OpenRouterModelId,
    RouteProvenance,
    completion_provenance,
    response_content,
    select_route,
)
from reasonese.planning import PromptSpec
from reasonese.routing import CollectionRouting
from reasonese.scheduling import ScheduledRequest
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
    manual_messages: ManualMessageLibrary | ManualMessageSnapshot,
    *,
    prefer_batch: bool,
    routing: CollectionRouting | None = None,
) -> tuple[GeneratedMessage, ...]:
    """Generate each distinct uncached input, grouped by author model."""
    return materialize_specs(
        matchup.inputs,
        client,
        cache,
        manual_messages,
        prefer_batch=prefer_batch,
        routing=routing,
    )


@beartype
def materialize_specs(
    specs: tuple[PromptSpec, ...],
    client: OpenRouterClient,
    cache: YamlMessageCache,
    manual_messages: ManualMessageLibrary | ManualMessageSnapshot,
    *,
    prefer_batch: bool,
    routing: CollectionRouting | None = None,
) -> tuple[GeneratedMessage, ...]:
    """Materialize arbitrary prompt specs with one shared model-grouped cache pass."""
    routing = routing or CollectionRouting()
    manual_snapshot = (
        manual_messages.snapshot(specs)
        if isinstance(manual_messages, ManualMessageLibrary)
        else manual_messages
    )
    materialized = {message.spec: message for message in cache.load()}
    new_messages: dict[PromptSpec, GeneratedMessage] = {}
    message_lock = Lock()

    user_specs = tuple(dict.fromkeys(spec for spec in specs if spec.author is Author.USER))
    for spec in user_specs:
        message = GeneratedMessage(spec, manual_snapshot.message_for(spec), None)
        if materialized.get(spec) != message:
            materialized[spec] = message
            new_messages[spec] = message

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
                select_route(author, routing.preference),
                tuple(authoring_request(spec) for spec in authored_specs),
            )
        )

    for group in completion_groups:
        if not str(group.route.model_id).endswith(":free"):
            routing.require_paid("uncached paid authoring")

    group_provenances = tuple(
        completion_provenance(group.route, group.bodies, prefer_batch=prefer_batch)
        for group in completion_groups
    )

    def collect_message(group_index: int, body_index: int, response: JsonObject) -> None:
        spec = grouped_specs[group_index][body_index]
        message = GeneratedMessage(
            spec,
            GeneratedText.parse(response_content(response)),
            response,
            group_provenances[group_index],
        )
        with message_lock:
            new_messages[spec] = message

    try:
        client.complete_many_grouped(
            tuple(completion_groups),
            prefer_batch=prefer_batch,
            on_response=collect_message,
        )
    finally:
        if new_messages:
            cache.put_many(tuple(new_messages.values()))
    materialized.update(new_messages)
    for spec in dict.fromkeys(specs):
        message = materialized[spec]
        routing.record(
            "author",
            spec.author,
            "manual" if spec.author is Author.USER else "new" if spec in new_messages else "cache",
            message.provenance,
            message.response,
        )
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
    *,
    on_trace: Callable[[int, int, ConversationTrace], None] | None = None,
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
    runtimes: dict[tuple[int, int], ToolRuntime] = {}
    available_runtimes: list[ToolRuntime] = []
    runtime_stack = ExitStack()
    runtime_lock = Lock()

    def runtime_for(key: tuple[int, int]) -> ToolRuntime:
        with runtime_lock:
            runtime = runtimes.get(key)
            if runtime is not None:
                return runtime
            group_index, setup_index = key
            readme_contents = groups[group_index].setups[setup_index].readme_contents()
            if available_runtimes:
                runtime = available_runtimes.pop()
                runtime.reset(readme_contents)
            else:
                runtime = runtime_stack.enter_context(ToolRuntime(readme_contents))
            runtimes[key] = runtime
            return runtime

    def request_for(key: tuple[int, int]) -> ScheduledRequest[JsonObject]:
        group_index, _ = key
        return client.completion_request(
            groups[group_index].route.model_id,
            _assistant_request(messages[key]),
            lambda response: receive(key, response),
        )

    def receive(key: tuple[int, int], response: JsonObject) -> ScheduledRequest[JsonObject] | None:
        group_index, setup_index = key
        calls = tool_calls_from_response(response)
        if not calls:
            response_content(response)
            trace = ConversationTrace(
                groups[group_index].setups[setup_index],
                response,
                tuple(steps[key]),
                RouteProvenance(groups[group_index].route.model_id, CompletionTransport.SYNC),
            )
            with runtime_lock:
                completed[key] = trace
                runtime = runtimes.pop(key, None)
                if runtime is not None:
                    available_runtimes.append(runtime)
            if on_trace is not None:
                on_trace(group_index, setup_index, trace)
            return None
        if len(steps[key]) == _MAX_LOCAL_TOOL_STEPS:
            raise RuntimeError(f"assistant exceeded {_MAX_LOCAL_TOOL_STEPS} local tool-call steps")
        runtime = runtime_for(key)
        results = tuple(runtime.execute(call) for call in calls)
        steps[key].append(ToolStep(response, results))
        messages[key].append(assistant_message_from_response(response))
        messages[key].extend(result.openrouter_dict() for result in results)
        return request_for(key)

    with runtime_stack:
        client.scheduler.run(request_for(key) for key in messages)

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
    routing: CollectionRouting | None = None,
) -> RunResult:
    """Return a cached trace or execute the complete matchup through OpenRouter."""
    routing = routing or CollectionRouting()
    manual_snapshot = manual_messages.snapshot(matchup.inputs)
    cached = trace_cache.get(matchup)
    if cached is not None and manual_snapshot.matches(cached.setup):
        cached_messages = tuple(
            GeneratedMessage(spec, cached.setup.content_for_input(index), None)
            for index, spec in enumerate(matchup.inputs)
        )
        require_compliant_messages(
            audit_messages(cached_messages, qa_cache, client, routing=routing)
        )
        routing.record("assistant", matchup.assistant, "cache", cached.provenance, cached.response)
        record_cached_authors((cached,), message_cache, routing)
        return RunResult(cached, True)

    routing.require_paid(
        "uncached assistant work (chargeable web search), including required message QA"
    )
    generated = materialize_messages(
        matchup,
        client,
        message_cache,
        manual_snapshot,
        prefer_batch=prefer_batch,
        routing=routing,
    )
    require_compliant_messages(audit_messages(generated, qa_cache, client, routing=routing))
    setup = construct_conversation(matchup, generated)
    trace = run_assistant(
        setup, select_route(matchup.assistant, routing.preference).model_id, client
    )
    trace_cache.put(trace)
    routing.record("assistant", matchup.assistant, "new", trace.provenance, trace.response)
    return RunResult(trace, False)


def record_cached_authors(
    traces: tuple[ConversationTrace, ...], cache: YamlMessageCache, routing: CollectionRouting
) -> None:
    """Report only author provenance supported by matching cached text, once per message."""
    if not traces:
        return
    messages = {message.spec: message for message in cache.load()}
    delivered = dict.fromkeys(
        (spec, trace.setup.content_for_input(index))
        for trace in traces
        for index, spec in enumerate(trace.setup.matchup.inputs)
    )
    for spec, content in delivered:
        message = messages.get(spec)
        if message is not None and message.content == content:
            routing.record("author", spec.author, "cache", message.provenance, message.response)
        else:
            routing.record(
                "author",
                spec.author,
                "manual" if spec.author is Author.USER else "cache",
                None,
                None,
            )
