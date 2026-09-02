from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import pytest

from reasonese.axes import Assistant, Author
from reasonese.openrouter import (
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

    def post_json(self, path: str, body: JsonObject) -> JsonObject:
        self.post_calls.append((path, body))
        return self.post_responses.pop(0)

    def get_json(self, path: str) -> JsonObject:
        self.get_calls.append(path)
        return self.get_responses.pop(0)


def test_model_routes_match_current_openrouter_slugs() -> None:
    assert str(model_route(Assistant.QWEN3_8_FLASH).model_id) == "qwen/qwen3.8-flash"
    assert model_route(Author.QWEN3_8_FLASH).batch_model_id is None
    assert str(model_route(Assistant.INKLING_SMALL).batch_model_id).endswith(":batch")
    with pytest.raises(ValueError, match="user author"):
        model_route(Author.USER)


def test_sync_completion_adds_the_selected_model() -> None:
    transport = FakeTransport(posts=[_chat("done")])
    client = OpenRouterClient(transport)
    model = OpenRouterModelId.parse("example/model")

    assert client.complete(model, {"messages": []}) == _chat("done")
    assert transport.post_calls == [
        ("/api/v1/chat/completions", {"messages": [], "model": "example/model"})
    ]


def test_complete_many_falls_back_to_sync_without_a_batch_variant() -> None:
    transport = FakeTransport(posts=[_chat("one"), _chat("two")])
    route = ModelRoute(OpenRouterModelId.parse("example/model"), None)
    client = OpenRouterClient(transport)

    assert client.complete_many(route, ({"messages": []}, {"messages": []}), prefer_batch=True) == (
        _chat("one"),
        _chat("two"),
    )
    assert client.complete_many(route, (), prefer_batch=True) == ()


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
