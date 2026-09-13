from __future__ import annotations

from collections import Counter
from pathlib import Path
from threading import Event, Lock

import pytest
import requests

from reasonese.axes import Assistant, Author, Channel
from reasonese.cache import YamlMessageCache
from reasonese.collect_data import CollectionTask, collect_studies
from reasonese.conversation import ConversationTrace, GeneratedText
from reasonese.manual_messages import ManualMessageLibrary
from reasonese.message_qa_cache import YamlMessageQaCache
from reasonese.openrouter import (
    CompletionGroup,
    JsonObject,
    ModelRoute,
    OpenRouterClient,
    OpenRouterModelId,
    RoutePreference,
)
from reasonese.routing import CollectionRouting
from reasonese.runner import AssistantRunGroup, materialize_specs, run_assistant_groups
from reasonese.scheduling import ModelLimiter
from reasonese.study import build_trials
from reasonese.study_cache import SqliteStudyCache
from reasonese.tools import ToolCall, ToolResult
from tests.test_cache_runner_cli import _setup_with_user_content, _tool_chat
from tests.test_scheduling import http_error
from tests.test_study_orchestration import (
    _batch_result,
    _chat,
    _judge_batch,
    _manual_library,
    _message_qa_batch,
    _spec,
    _study,
)


class RecoveringCollectionTransport:
    def __init__(self) -> None:
        self.limited = True
        self.calls: Counter[str] = Counter()
        self.batch_schemas: list[str] = []
        self.lock = Lock()

    def post_json(self, path: str, body: JsonObject) -> JsonObject:
        model = body["model"]
        with self.lock:
            self.calls[model] += 1
            count = self.calls[model]
        if path == "/api/beta/batches":
            schema = body["requests"][0]["body"]["response_format"]["json_schema"]["name"]
            self.batch_schemas.append(schema)
            if schema == "message_compliance_verdict":
                return _message_qa_batch(len(body["requests"]))
            assert schema == "instruction_completion_verdict"
            return _judge_batch((True,) * len(body["requests"]))
        if self.limited and "gemma" in model:
            raise http_error()
        return _chat("complete answer", f"{model}-{count}")

    def get_json(self, path: str) -> JsonObject:
        raise AssertionError(path)


@pytest.mark.parametrize("shared", [False, True])
def test_exhausted_model_preserves_healthy_trials_and_resume_avoids_repeating_them(
    tmp_path: Path, shared: bool
) -> None:
    healthy = _study(3, Assistant.NEMOTRON_3_5_LIGHTNING)
    limited = _study(3, Assistant.GEMMA_4_31B_IT)
    tasks = (
        CollectionTask(limited, tmp_path / "limited"),
        CollectionTask(healthy, tmp_path / "healthy"),
    )
    manual = _manual_library(tmp_path, limited, healthy)
    message_cache = YamlMessageCache(tmp_path / "messages.yaml")
    qa_cache = YamlMessageQaCache(tmp_path / "qa.yaml")
    shared_cache = SqliteStudyCache(tmp_path / "shared.sqlite3") if shared else None
    transport = RecoveringCollectionTransport()

    def collect(client: OpenRouterClient | None):
        return collect_studies(
            tasks,
            client,
            manual,
            message_cache,
            qa_cache,
            prefer_batch=True,
            routing=CollectionRouting(RoutePreference.FREE, True),
            shared_cache=shared_cache,
        )

    with pytest.raises(requests.HTTPError, match="429"):
        collect(OpenRouterClient(transport, rate_limit_retries=0))

    healthy_slug = "nvidia/nemotron-3.5-lightning:free"
    assert transport.calls[healthy_slug] == 6
    assert transport.batch_schemas == ["message_compliance_verdict"]
    assert all(not (task.output_dir / "observations.jsonl").exists() for task in tasks)
    for task, expected in zip(tasks, (0, 6), strict=True):
        cache = shared_cache or SqliteStudyCache(task.output_dir / "collection.sqlite3")
        trials = build_trials(task.study)
        assert len(cache.load_traces(trials)) == expected
        assert cache.load_judgments(trials) == {}

    transport.limited = False
    resumed = collect(OpenRouterClient(transport))
    assert [result.trace_cache_hits for result in resumed] == [0, 6]
    assert [len(result.observations) for result in resumed] == [12, 12]
    assert transport.calls[healthy_slug] == 6
    assert transport.batch_schemas == [
        "message_compliance_verdict", "instruction_completion_verdict"
    ]

    calls_before_warm = transport.calls.copy()
    warm = collect(None)
    assert [result.trace_cache_hits for result in warm] == [6, 6]
    assert [result.judgment_cache_hits for result in warm] == [6, 6]
    assert [result.observations for result in warm] == [result.observations for result in resumed]
    assert transport.calls == calls_before_warm


def test_authoring_persists_successful_model_and_retries_only_missing_specs(tmp_path: Path) -> None:
    specs = (
        _spec("First instruction.", Channel.USER, Author.GEMMA_4_31B_IT),
        _spec("Second instruction.", Channel.USER, Author.NEMOTRON_3_5_LIGHTNING),
    )
    cache = YamlMessageCache(tmp_path / "messages.yaml")
    transport = RecoveringCollectionTransport()
    manual = ManualMessageLibrary(tmp_path / "manual")
    with pytest.raises(requests.HTTPError, match="429"):
        materialize_specs(
            specs,
            OpenRouterClient(transport, rate_limit_retries=0),
            cache,
            manual,
            prefer_batch=False,
        )
    cached = cache.load()
    assert tuple(message.spec for message in cached) == (specs[1],)
    assert cached[0].provenance is not None
    assert str(cached[0].provenance.requested_model_id) == "nvidia/nemotron-3.5-lightning:free"

    transport.limited = False
    materialized = materialize_specs(
        specs, OpenRouterClient(transport), cache, manual, prefer_batch=False
    )
    assert tuple(message.spec for message in materialized) == specs
    assert materialized[1] == cached[0]
    assert transport.calls["nvidia/nemotron-3.5-lightning:free"] == 1
    assert transport.calls["google/gemma-4-31b-it:free"] == 2


@pytest.mark.parametrize("status", [429, 403])
def test_accepted_author_batch_is_collected_and_cached_after_sync_model_failure(
    tmp_path: Path, status: int,
) -> None:
    batch_accepted = Event()
    polled: list[str] = []

    class Transport:
        def post_json(self, path: str, body: JsonObject) -> JsonObject:
            if path == "/api/beta/batches":
                batch_accepted.set()
                return {"id": "accepted-author", "status": "in_progress"}
            assert batch_accepted.wait(2)
            raise http_error(status)

        def get_json(self, path: str) -> JsonObject:
            polled.append(path)
            return {
                "id": "accepted-author",
                "status": "completed",
                "results": [_batch_result("request-0", _chat("batch wording", "author"))],
            }

    specs = (
        _spec("Batch instruction.", Channel.USER, Author.QWEN3_8_2_4T),
        _spec("Sync instruction.", Channel.USER, Author.NEMOTRON_3_5_LIGHTNING),
    )
    cache = YamlMessageCache(tmp_path / "messages.yaml")
    with pytest.raises(requests.HTTPError, match=str(status)):
        materialize_specs(
            specs,
            OpenRouterClient(Transport(), rate_limit_retries=0, poll_interval_seconds=0.0),
            cache,
            ManualMessageLibrary(tmp_path / "manual"),
            prefer_batch=True,
            routing=CollectionRouting(RoutePreference.BATCH, True),
        )
    assert polled == ["/api/beta/batches/accepted-author"]
    assert len(cache.load()) == 1
    assert cache.load()[0].spec == specs[0]
    assert cache.load()[0].response == _chat("batch wording", "author")


def test_slow_local_tool_does_not_block_another_models_queued_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    healthy_finished = Event()
    closed: list[bool] = []

    class SlowRuntime:
        def __init__(self, readme_contents: tuple[GeneratedText, ...]) -> None:
            assert readme_contents

        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> None:
            closed.append(True)

        def execute(self, call: ToolCall) -> ToolResult:
            assert healthy_finished.wait(2), "local tool blocked the healthy model's next request"
            return ToolResult(call.call_id, GeneratedText.parse("tool result"))

    class Transport:
        def __init__(self) -> None:
            self.healthy_count = 0
            self.tool_bodies: list[JsonObject] = []

        def post_json(self, path: str, body: JsonObject) -> JsonObject:
            if body["model"] == "example/tool":
                self.tool_bodies.append(body)
                if body["messages"][-1]["role"] == "tool":
                    return _chat("tool final", "tool-final")
                return _tool_chat()
            self.healthy_count += 1
            if self.healthy_count == 3:
                healthy_finished.set()
            return _chat("healthy final", f"healthy-{self.healthy_count}")

        def get_json(self, path: str) -> JsonObject:
            raise AssertionError(path)

    monkeypatch.setattr("reasonese.runner.ToolRuntime", SlowRuntime)
    setup = _setup_with_user_content("request")
    transport = Transport()
    traces = run_assistant_groups(
        (
            AssistantRunGroup(ModelRoute(OpenRouterModelId.parse("example/tool"), None), (setup,)),
            AssistantRunGroup(
                ModelRoute(OpenRouterModelId.parse("example/healthy"), None), (setup,) * 3
            ),
        ),
        OpenRouterClient(transport, sync_workers=1),
    )
    assert healthy_finished.is_set()
    assert closed == [True]
    assert [trace.response["id"] for trace in traces[1]] == ["healthy-1", "healthy-2", "healthy-3"]
    assert traces[0][0].tool_steps[0].response == _tool_chat()
    assert transport.tool_bodies[1]["messages"][-1] == {
        "role": "tool", "tool_call_id": "live-call", "content": "tool result"
    }


def test_successful_inflight_trace_is_delivered_when_its_model_exhausts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peer_started = Event()
    limit_observed = Event()
    observed: list[tuple[int, int, ConversationTrace]] = []
    original_limited = ModelLimiter.limited

    def track_limit(self: ModelLimiter, now: float, retry_after: float) -> None:
        original_limited(self, now, retry_after)
        limit_observed.set()

    monkeypatch.setattr(ModelLimiter, "limited", track_limit)

    class Transport:
        def post_json(self, path: str, body: JsonObject) -> JsonObject:
            content = next(m["content"] for m in body["messages"] if m["role"] == "user")
            if content == "limited":
                assert peer_started.wait(2)
                raise http_error()
            peer_started.set()
            assert limit_observed.wait(2)
            return _chat("successful peer", "peer")

        def get_json(self, path: str) -> JsonObject:
            raise AssertionError(path)

    with pytest.raises(requests.HTTPError, match="429"):
        run_assistant_groups(
            (
                AssistantRunGroup(
                    ModelRoute(OpenRouterModelId.parse("example/model"), None),
                    (_setup_with_user_content("limited"), _setup_with_user_content("success")),
                ),
            ),
            OpenRouterClient(Transport(), rate_limit_retries=0),
            on_trace=lambda group, index, trace: observed.append((group, index, trace)),
        )
    assert [(group, index, trace.response["id"]) for group, index, trace in observed] == [
        (0, 1, "peer")
    ]


def test_batch_failure_does_not_replace_original_submission_failure() -> None:
    original = http_error()

    class Transport:
        def post_json(self, path: str, body: JsonObject) -> JsonObject:
            if path == "/api/beta/batches":
                return {"id": "accepted", "status": "failed"}
            raise original

        def get_json(self, path: str) -> JsonObject:
            raise AssertionError(path)

    route = ModelRoute(
        OpenRouterModelId.parse("example/batch"), OpenRouterModelId.parse("example/batch:batch")
    )
    with pytest.raises(requests.HTTPError) as caught:
        OpenRouterClient(Transport(), rate_limit_retries=0).complete_many_grouped(
            (
                CompletionGroup(route, ({},)),
                CompletionGroup(ModelRoute(OpenRouterModelId.parse("example/sync"), None), ({},)),
            ),
            prefer_batch=True,
        )
    assert caught.value is original
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert "ended with status failed" in str(caught.value.__cause__)


def test_empty_assistant_groups_preserve_shape_without_requests() -> None:
    transport = RecoveringCollectionTransport()
    client = OpenRouterClient(transport)
    assert run_assistant_groups((), client) == ()
    assert run_assistant_groups(
        (
            AssistantRunGroup(ModelRoute(OpenRouterModelId.parse("example/one"), None), ()),
            AssistantRunGroup(ModelRoute(OpenRouterModelId.parse("example/two"), None), ()),
        ),
        client,
    ) == ((), ())
    assert transport.calls == {}
    assert client.scheduler.limiters == {}


@pytest.mark.parametrize("malformed_kind", ["content", "duplicate"])
def test_malformed_author_batch_preserves_valid_results_from_other_accepted_batches(
    tmp_path: Path, malformed_kind: str,
) -> None:
    polled: list[str] = []

    class Transport:
        def post_json(self, path: str, body: JsonObject) -> JsonObject:
            assert path == "/api/beta/batches"
            if body["model"].startswith("qwen/"):
                responses = [
                    _batch_result("request-0", _chat("", "empty")),
                    _batch_result("request-1", {}),
                    _batch_result("request-2", _chat("valid sibling", "sibling")),
                ]
                if malformed_kind == "duplicate":
                    responses = [
                        _batch_result("request-0", _chat("original", "original")),
                        _batch_result("request-0", _chat("conflicting", "conflicting")),
                        _batch_result("request-1", _chat("another", "another")),
                        _batch_result("request-2", _chat("valid sibling", "sibling")),
                    ]
                return {
                    "id": "malformed-author",
                    "status": "completed",
                    "results": responses,
                }
            return {"id": "healthy-author", "status": "in_progress"}

        def get_json(self, path: str) -> JsonObject:
            polled.append(path)
            return {
                "id": "healthy-author",
                "status": "completed",
                "results": [_batch_result("request-0", _chat("valid peer", "peer"))],
            }

    specs = (
        *(_spec(f"Task {index}.", Channel.USER, Author.QWEN3_8_2_4T) for index in range(3)),
        _spec("Peer task.", Channel.USER, Author.GEMMA_4_31B_IT),
    )
    cache = YamlMessageCache(tmp_path / "messages.yaml")
    error = "assistant content is empty" if malformed_kind == "content" else "duplicate custom_id"
    with pytest.raises(ValueError, match=error):
        materialize_specs(
            specs,
            OpenRouterClient(Transport(), poll_interval_seconds=0.0),
            cache,
            ManualMessageLibrary(tmp_path / "manual"),
            prefer_batch=True,
            routing=CollectionRouting(RoutePreference.BATCH, True),
        )
    assert polled == ["/api/beta/batches/healthy-author"]
    if malformed_kind == "content":
        assert tuple(message.spec for message in cache.load()) == specs[2:]
        assert tuple(message.response for message in cache.load()) == (
            _chat("valid sibling", "sibling"), _chat("valid peer", "peer")
        )
    else:
        assert tuple(message.spec for message in cache.load()) == specs[3:]
        assert tuple(message.response for message in cache.load()) == (_chat("valid peer", "peer"),)


@pytest.mark.parametrize("failure_kind", ["terminal", "poll", "timeout", "all_poll"])
def test_failed_batch_does_not_discard_other_accepted_batch_results(failure_kind: str) -> None:
    polled: list[str] = []
    received: list[tuple[int, int, JsonObject]] = []

    class Transport:
        def post_json(self, path: str, body: JsonObject) -> JsonObject:
            case = body["requests"][0]["body"]["case"]
            if case == "healthy" and failure_kind != "poll":
                return {
                    "id": case,
                    "status": "completed",
                    "results": [_batch_result("request-0", {"id": "healthy-result"})],
                }
            return {
                "id": case,
                "status": "failed" if failure_kind == "terminal" else "in_progress",
            }

        def get_json(self, path: str) -> JsonObject:
            polled.append(path)
            if "failed" in path:
                raise requests.HTTPError(path)
            return {
                "id": "healthy",
                "status": "completed",
                "results": [_batch_result("request-0", {"id": "healthy-result"})],
            }

    cases = ("failed-0", "failed-1") if failure_kind == "all_poll" else (
        "failed-0", "failed-1", "healthy"
    )
    groups = tuple(
        CompletionGroup(
            ModelRoute(
                OpenRouterModelId.parse(f"example/{case}"),
                OpenRouterModelId.parse(f"example/{case}:batch"),
            ),
            ({"case": case},),
        )
        for case in cases
    )
    expected_exception = {
        "terminal": RuntimeError,
        "poll": requests.HTTPError,
        "timeout": TimeoutError,
        "all_poll": requests.HTTPError,
    }[failure_kind]
    with pytest.raises(expected_exception, match="failed-0"):
        OpenRouterClient(
            Transport(),
            poll_interval_seconds=0.0,
            batch_timeout_seconds=0.0 if failure_kind == "timeout" else 100.0,
        ).complete_many_grouped(
            groups,
            prefer_batch=True,
            on_response=lambda group, index, response: received.append((group, index, response)),
        )
    assert received == ([] if failure_kind == "all_poll" else [(2, 0, {"id": "healthy-result"})])
    assert polled == (
        [f"/api/beta/batches/{case}" for case in cases]
        if failure_kind in ("poll", "all_poll")
        else []
    )
