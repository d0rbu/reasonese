from __future__ import annotations

from collections.abc import Iterator
from threading import Lock
from time import sleep
from typing import Any, cast

import pytest

from reasonese.axes import Assistant, Author
from reasonese.openrouter import (
    CompletionGroup,
    JsonObject,
    ModelRoute,
    OpenRouterClient,
    OpenRouterModelId,
    RequestsTransport,
    model_route,
    response_content,
)


def _chat(content: str) -> JsonObject:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


class FakeTransport:
    def __init__(
        self,
        posts: list[JsonObject] | None = None,
        gets: list[JsonObject] | None = None,
    ) -> None:
        self.post_responses = posts or []
        self.get_responses = gets or []
        self.post_calls: list[tuple[str, JsonObject]] = []
        self.get_calls: list[str] = []
        self.calls: list[tuple[str, str]] = []

    def post_json(self, path: str, body: JsonObject) -> JsonObject:
        self.post_calls.append((path, body))
        self.calls.append(("POST", path))
        return self.post_responses.pop(0)

    def get_json(self, path: str) -> JsonObject:
        self.get_calls.append(path)
        self.calls.append(("GET", path))
        return self.get_responses.pop(0)


def test_model_routes_match_current_openrouter_slugs() -> None:
    assert str(model_route(Assistant.QWEN3_8_FLASH).model_id) == "qwen/qwen3.8-flash"
    assert model_route(Author.QWEN3_8_FLASH).batch_model_id is None
    assert str(model_route(Assistant.INKLING_SMALL).batch_model_id).endswith(":batch")
    with pytest.raises(ValueError, match="user author"):
        model_route(Author.USER)


def test_sync_completion_adds_the_selected_model() -> None:
    transport = FakeTransport(posts=[_chat("done")])
    client = OpenRouterClient(transport, sync_workers=1)
    model = OpenRouterModelId.parse("example/model")

    assert client.complete(model, {"messages": []}) == _chat("done")
    assert transport.post_calls == [
        ("/api/v1/chat/completions", {"messages": [], "model": "example/model"})
    ]


def test_complete_many_falls_back_to_sync_without_a_batch_variant() -> None:
    transport = FakeTransport(posts=[_chat("one"), _chat("two")])
    route = ModelRoute(OpenRouterModelId.parse("example/model"), None)
    client = OpenRouterClient(transport, sync_workers=1)

    assert client.complete_many(route, ({"messages": []}, {"messages": []}), prefer_batch=True) == (
        _chat("one"),
        _chat("two"),
    )
    assert client.complete_many(route, (), prefer_batch=True) == ()


def test_server_tools_fall_back_to_sync_when_a_batch_route_exists() -> None:
    transport = FakeTransport(posts=[_chat("searched")])
    route = ModelRoute(
        OpenRouterModelId.parse("example/model"),
        OpenRouterModelId.parse("example/model:batch"),
    )
    body = {"messages": [], "tools": [{"type": "openrouter:web_search"}]}

    result = OpenRouterClient(transport).complete_many(
        route,
        (body,),
        prefer_batch=True,
    )

    assert result == (_chat("searched"),)
    assert transport.post_calls == [
        ("/api/v1/chat/completions", {**body, "model": "example/model"})
    ]


class ConcurrentTransport:
    def __init__(self) -> None:
        self.lock = Lock()
        self.active = 0
        self.max_active = 0

    def post_json(self, path: str, body: JsonObject) -> JsonObject:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        sleep(0.03)
        with self.lock:
            self.active -= 1
        return _chat(body["messages"][0]["content"])

    def get_json(self, path: str) -> JsonObject:
        raise AssertionError(f"unexpected GET {path}")


def test_sync_completion_groups_run_concurrently_and_restore_order() -> None:
    transport = ConcurrentTransport()
    route = ModelRoute(OpenRouterModelId.parse("example/model"), None)
    bodies = tuple(
        {"messages": [{"role": "user", "content": content}]} for content in ("one", "two", "three")
    )

    results = OpenRouterClient(transport, sync_workers=3).complete_many(
        route,
        bodies,
        prefer_batch=True,
    )

    assert results == (_chat("one"), _chat("two"), _chat("three"))
    assert transport.max_active == 3


def test_sync_worker_count_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        OpenRouterClient(FakeTransport(), sync_workers=0)


def _batch_result(custom_id: str, content: str) -> JsonObject:
    return {
        "custom_id": custom_id,
        "response": {"status_code": 200, "body": _chat(content)},
        "error": None,
    }


def test_batch_completion_polls_and_restores_submission_order() -> None:
    transport = FakeTransport(
        posts=[{"id": "batch-1", "status": "validating"}],
        gets=[
            {"id": "batch-1", "status": "in_progress"},
            {
                "id": "batch-1",
                "status": "completed",
                "results": [_batch_result("request-1", "two"), _batch_result("request-0", "one")],
            },
        ],
    )
    sleeps: list[float] = []
    route = ModelRoute(
        OpenRouterModelId.parse("example/model"),
        OpenRouterModelId.parse("example/model:batch"),
    )
    client = OpenRouterClient(
        transport,
        poll_interval_seconds=0.25,
        sleep=sleeps.append,
        monotonic=lambda: 0.0,
    )

    results = client.complete_many(
        route,
        ({"messages": [{"role": "user", "content": "one"}]}, {"messages": []}),
        prefer_batch=True,
    )

    assert results == (_chat("one"), _chat("two"))
    assert sleeps == [0.25, 0.25]
    path, payload = transport.post_calls[0]
    assert path == "/api/beta/batches"
    assert list(payload) == ["endpoint", "model", "requests"]
    assert payload["model"] == "example/model"
    assert payload["requests"] == [
        {
            "custom_id": "request-0",
            "body": {
                "messages": [{"role": "user", "content": "one"}],
                "model": "example/model",
            },
        },
        {
            "custom_id": "request-1",
            "body": {"messages": [], "model": "example/model"},
        },
    ]
    assert transport.get_calls == ["/api/beta/batches/batch-1"] * 2


def test_grouped_completion_submits_all_batches_before_polling() -> None:
    transport = FakeTransport(
        posts=[
            {"id": "batch-1", "status": "validating"},
            {"id": "batch-2", "status": "validating"},
        ],
        gets=[
            {
                "id": "batch-1",
                "status": "completed",
                "results": [
                    _batch_result("request-1", "one-b"),
                    _batch_result("request-0", "one-a"),
                ],
            },
            {
                "id": "batch-2",
                "status": "completed",
                "results": [_batch_result("request-0", "two-a")],
            },
        ],
    )
    route = ModelRoute(
        OpenRouterModelId.parse("example/model"),
        OpenRouterModelId.parse("example/model:batch"),
    )
    sleeps: list[float] = []
    client = OpenRouterClient(
        transport,
        poll_interval_seconds=0.25,
        sleep=sleeps.append,
        monotonic=lambda: 0.0,
    )

    results = client.complete_many_grouped(
        (
            CompletionGroup(
                route, ({"messages": [{"role": "user", "content": "one-a"}]}, {"messages": []})
            ),
            CompletionGroup(route, ({"messages": [{"role": "user", "content": "two-a"}]},)),
        ),
        prefer_batch=True,
    )

    assert results == ((_chat("one-a"), _chat("one-b")), (_chat("two-a"),))
    assert transport.calls == [
        ("POST", "/api/beta/batches"),
        ("POST", "/api/beta/batches"),
        ("GET", "/api/beta/batches/batch-1"),
        ("GET", "/api/beta/batches/batch-2"),
    ]
    assert sleeps == [0.25]


def test_grouped_completion_starts_batches_before_synchronous_groups() -> None:
    transport = FakeTransport(
        posts=[
            {"id": "batch", "status": "validating"},
            _chat("sync"),
        ],
        gets=[
            {
                "id": "batch",
                "status": "completed",
                "results": [_batch_result("request-0", "batch")],
            }
        ],
    )
    sync_route = ModelRoute(OpenRouterModelId.parse("example/sync"), None)
    batch_route = ModelRoute(
        OpenRouterModelId.parse("example/batch"),
        OpenRouterModelId.parse("example/batch:batch"),
    )
    client = OpenRouterClient(transport, sleep=lambda _: None, monotonic=lambda: 0.0)

    results = client.complete_many_grouped(
        (
            CompletionGroup(sync_route, ({"messages": []},)),
            CompletionGroup(batch_route, ({"messages": []},)),
            CompletionGroup(batch_route, ()),
        ),
        prefer_batch=True,
    )

    assert results == ((_chat("sync"),), (_chat("batch"),), ())
    assert transport.calls[:2] == [
        ("POST", "/api/beta/batches"),
        ("POST", "/api/v1/chat/completions"),
    ]


def test_grouped_completion_matches_sequential_batch_payloads_and_results() -> None:
    route = ModelRoute(
        OpenRouterModelId.parse("example/model"),
        OpenRouterModelId.parse("example/model:batch"),
    )
    groups = (
        CompletionGroup(route, ({"messages": [{"role": "user", "content": "a"}]},)),
        CompletionGroup(
            route,
            (
                {"messages": [{"role": "user", "content": "b"}]},
                {"messages": [{"role": "user", "content": "c"}]},
            ),
        ),
    )
    terminal_batches = [
        {
            "id": "batch-1",
            "status": "completed",
            "results": [_batch_result("request-0", "a")],
        },
        {
            "id": "batch-2",
            "status": "completed",
            "results": [_batch_result("request-0", "b"), _batch_result("request-1", "c")],
        },
    ]
    sequential_transport = FakeTransport(posts=list(terminal_batches))
    grouped_transport = FakeTransport(posts=list(terminal_batches))

    sequential = tuple(
        OpenRouterClient(sequential_transport).complete_many(
            group.route,
            group.bodies,
            prefer_batch=True,
        )
        for group in groups
    )
    grouped = OpenRouterClient(grouped_transport).complete_many_grouped(
        groups,
        prefer_batch=True,
    )

    assert grouped == sequential
    assert grouped_transport.post_calls == sequential_transport.post_calls


@pytest.mark.parametrize(
    ("created", "error"),
    [
        ({"status": "validating"}, "missing an id"),
        ({"id": "batch", "status": "failed"}, "ended with status failed"),
        ({"id": "batch", "status": "completed"}, "has no results"),
        ({"id": "batch", "status": "completed", "results": ["bad"]}, "not an object"),
        (
            {
                "id": "batch",
                "status": "completed",
                "results": [{"custom_id": "request-0", "response": None, "error": "bad"}],
            },
            "failed",
        ),
        (
            {
                "id": "batch",
                "status": "completed",
                "results": [{"custom_id": 1, "response": {}, "error": None}],
            },
            "malformed",
        ),
        (
            {
                "id": "batch",
                "status": "completed",
                "results": [
                    {
                        "custom_id": "request-0",
                        "response": {"status_code": 500, "body": {}},
                        "error": None,
                    }
                ],
            },
            "returned",
        ),
        (
            {
                "id": "batch",
                "status": "completed",
                "results": [_batch_result("wrong", "bad")],
            },
            "do not match",
        ),
    ],
)
def test_batch_completion_rejects_bad_terminal_responses(created: JsonObject, error: str) -> None:
    route = ModelRoute(
        OpenRouterModelId.parse("example/model"),
        OpenRouterModelId.parse("example/model:batch"),
    )
    client = OpenRouterClient(FakeTransport(posts=[created]))
    with pytest.raises((RuntimeError, ValueError), match=error):
        client.complete_many(route, ({"messages": []},), prefer_batch=True)


def test_batch_completion_times_out() -> None:
    route = ModelRoute(
        OpenRouterModelId.parse("example/model"),
        OpenRouterModelId.parse("example/model:batch"),
    )
    client = OpenRouterClient(
        FakeTransport(posts=[{"id": "batch", "status": "validating"}]),
        batch_timeout_seconds=0.0,
        monotonic=lambda: 1.0,
    )
    with pytest.raises(TimeoutError, match="did not finish"):
        client.complete_many(route, ({"messages": []},), prefer_batch=True)


@pytest.mark.parametrize(
    "response",
    [{}, {"choices": []}, _chat(""), {"choices": [{"message": {"content": 1}}]}],
)
def test_response_content_rejects_missing_or_empty_text(response: JsonObject) -> None:
    with pytest.raises(ValueError, match="content"):
        response_content(response)


def test_response_content_strips_text() -> None:
    assert response_content(_chat("  answer  ")) == "answer"


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.raised = False

    def raise_for_status(self) -> None:
        self.raised = True

    def json(self) -> object:
        return self.payload


class FakeSession:
    def __init__(self, responses: Iterator[FakeResponse]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append(("post", url, kwargs))
        return next(self.responses)

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append(("get", url, kwargs))
        return next(self.responses)


def test_requests_transport_posts_and_gets_authenticated_json() -> None:
    first = FakeResponse({"one": 1})
    second = FakeResponse({"two": 2})
    session = FakeSession(iter((first, second)))
    transport = RequestsTransport("secret", base_url="https://example.test/")
    cast(Any, transport)._session = session

    assert transport.post_json("/post", {"x": 1}) == {"one": 1}
    assert transport.get_json("/get") == {"two": 2}
    assert first.raised and second.raised
    assert session.calls[0][1] == "https://example.test/post"
    assert session.calls[0][2]["headers"]["Authorization"] == "Bearer secret"


def test_requests_transport_rejects_blank_keys_and_non_object_json() -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        RequestsTransport(" ")
    response = FakeResponse([])
    transport = RequestsTransport("secret")
    cast(Any, transport)._session = FakeSession(iter((response,)))
    with pytest.raises(ValueError, match="non-object"):
        transport.get_json("/bad")
