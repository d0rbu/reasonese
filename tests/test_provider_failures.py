"""Provider failures must not masquerade as text, tools, or cached author messages."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from threading import Lock

import pytest
import requests

from reasonese.axes import Author, Channel, Framing, Instruction
from reasonese.cache import YamlMessageCache
from reasonese.manual_messages import ManualMessageLibrary
from reasonese.openrouter import (
    CompletionGroup,
    JsonObject,
    ModelRoute,
    OpenRouterClient,
    OpenRouterModelId,
    RoutePreference,
    response_content,
)
from reasonese.planning import PromptSpec
from reasonese.routing import CollectionRouting
from reasonese.runner import materialize_specs
from reasonese.scheduling import ProviderRequestError


def chat(text: str) -> JsonObject:
    return {"id": "response", "choices": [{"message": {"content": text}}]}


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def sleep(self, delay: float) -> None:
        self.now += delay

    def monotonic(self) -> float:
        return self.now


class Transport:
    def __init__(self, responses: list[JsonObject | Exception]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, JsonObject]] = []

    def post_json(self, path: str, body: JsonObject) -> JsonObject:
        self.calls.append((path, body))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def get_json(self, path: str) -> JsonObject:
        raise AssertionError(path)


def client(transport: Transport) -> OpenRouterClient:
    clock = Clock()
    return OpenRouterClient(transport, sleep=clock.sleep, monotonic=clock.monotonic)


ERRORS: tuple[JsonObject, ...] = (
    {"error": {"code": 503, "message": "unavailable"}},
    {"choices": [{"error": {"code": 504, "message": "Server tool request failed", "metadata": {"error_type": "timeout"}}, "message": {"content": None}}]},
    {"choices": [{"finish_reason": "error", "message": {"content": "misleading partial answer"}}]},
    {"choices": [{"native_finish_reason": "error", "message": {"content": "misleading partial answer"}}]},
    {"error": "stream interrupted", "choices": [{"message": {"content": "partial"}}]},
    chat(""),
)


@pytest.mark.parametrize("response", ERRORS)
def test_errors_log_and_retry_exact_request_before_callback(response: JsonObject, caplog: pytest.LogCaptureFixture) -> None:
    transport = Transport([response, chat("success")])
    seen = []
    model = OpenRouterModelId.parse("example/model:free")
    c = client(transport)
    c.scheduler.run((c.completion_request(model, {"messages": []}, seen.append),))
    assert seen == [chat("success")]
    assert transport.calls[0] == transport.calls[1]
    assert len([record for record in caplog.records if record.exc_info]) == 1
    assert c.scheduler.monotonic() == 1
    with pytest.raises(ProviderRequestError):
        response_content(response)


@pytest.mark.parametrize("response", ERRORS)
def test_error_exhaustion_does_not_call_consumer(response: JsonObject) -> None:
    transport = Transport([response] * 3)
    c = client(transport)
    seen = []
    with pytest.raises(ProviderRequestError):
        c.scheduler.run((c.completion_request(OpenRouterModelId.parse("example/model:free"), {}, seen.append),))
    assert len(transport.calls) == 3
    assert seen == []
    assert c.scheduler.monotonic() == 3


@pytest.mark.parametrize("status,attempts", [(400, 1), (401, 1), (403, 1), (408, 3), (500, 3), (504, 3)])
def test_embedded_status_controls_retry(status: int, attempts: int) -> None:
    response = {"error": {"code": status, "message": "provider failure"}}
    transport = Transport([response] * attempts)
    with pytest.raises(ProviderRequestError):
        client(transport).complete(OpenRouterModelId.parse("example/model:free"), {})
    assert len(transport.calls) == attempts


def test_embedded_429_uses_adaptive_limiter_and_provider_cooldown() -> None:
    transport = Transport([{"error": {"code": "429", "metadata": {"headers": {"Retry-After": "120"}}}}, chat("success")])
    c = client(transport)
    c.complete(OpenRouterModelId.parse("example/model:free"), {})
    assert c.scheduler.monotonic() == 120
    assert c.scheduler.limiters["example/model:free"].generation == 1


@pytest.mark.parametrize("error", [requests.Timeout("read timeout"), requests.ConnectionError("disconnected")])
def test_transport_failure_retries_only_completion_request(error: Exception) -> None:
    transport = Transport([error, chat("success")])
    assert client(transport).complete(OpenRouterModelId.parse("example/model:free"), {}) == chat("success")
    assert transport.calls[0] == transport.calls[1]


def test_http_504_completion_retries_but_batch_submission_does_not() -> None:
    response = requests.Response()
    response.status_code = 504
    error = requests.HTTPError("timeout", response=response)
    transport = Transport([error, chat("success")])
    client(transport).complete(OpenRouterModelId.parse("example/model"), {})
    assert len(transport.calls) == 2
    batch_transport = Transport([error])
    route = ModelRoute(OpenRouterModelId.parse("example/model"), OpenRouterModelId.parse("example/model:batch"))
    with pytest.raises(requests.HTTPError):
        client(batch_transport).complete_many(route, ({},), prefer_batch=True)
    assert len(batch_transport.calls) == 1


def test_callback_failure_never_replays_successful_request() -> None:
    transport = Transport([chat("success")])
    c = client(transport)
    def receive(response: JsonObject) -> None:
        raise ProviderRequestError("consumer failed")
    with pytest.raises(ProviderRequestError, match="consumer failed"):
        c.scheduler.run((c.completion_request(OpenRouterModelId.parse("example/model"), {}, receive),))
    assert len(transport.calls) == 1


def test_authoring_saves_successful_peer_and_only_resumes_missing_spec(tmp_path: Path) -> None:
    specs = tuple(PromptSpec(Instruction.parse(text), Framing.NORMAL, Channel.USER, Author.NEMOTRON_3_5_LIGHTNING) for text in ("Bad.", "Good."))
    class AuthorTransport(Transport):
        def __init__(self) -> None:
            super().__init__([])
            self.counts: Counter[str] = Counter()
            self.lock = Lock()
            self.fail = True

        def post_json(self, path: str, body: JsonObject) -> JsonObject:
            text = "Bad." if "<request>\nBad." in body["messages"][0]["content"] else "Good."
            with self.lock:
                self.counts[text] += 1
            return ERRORS[1] if text == "Bad." and self.fail else chat(text)

    transport = AuthorTransport()
    cache = YamlMessageCache(tmp_path / "messages.yaml")
    manual = ManualMessageLibrary(tmp_path / "manual")
    routing = CollectionRouting(RoutePreference.FREE, False)
    with pytest.raises(ProviderRequestError):
        materialize_specs(specs, client(transport), cache, manual, prefer_batch=False, routing=routing)
    assert cache.get(specs[0]) is None
    saved = cache.get(specs[1])
    assert saved is not None
    assert transport.counts == {"Bad.": 3, "Good.": 1}
    transport.fail = False
    messages = materialize_specs(specs, client(transport), cache, manual, prefer_batch=False, routing=routing)
    assert messages[1] == saved
    assert transport.counts == {"Bad.": 4, "Good.": 1}


@pytest.mark.parametrize("item_error", [True, False])
def test_batch_keeps_successful_items_but_never_delivers_failed_item(item_error: bool) -> None:
    bad = {"custom_id": "request-0", "error": {"code": 504}, "response": None} if item_error else {
        "custom_id": "request-0", "error": None, "response": {"status_code": 200, "body": ERRORS[2]}}
    batch = {"id": "batch", "status": "completed", "results": [bad, {
        "custom_id": "request-1", "error": None, "response": {"status_code": 200, "body": chat("success")}}]}
    transport = Transport([batch])
    seen = []
    route = ModelRoute(OpenRouterModelId.parse("example/model"), OpenRouterModelId.parse("example/model:batch"))
    with pytest.raises(ProviderRequestError):
        client(transport).complete_many_grouped((CompletionGroup(route, ({}, {})),), prefer_batch=True,
            on_response=lambda group, index, response: seen.append((group, index, response)))
    assert seen == [(0, 1, chat("success"))]
    assert len(transport.calls) == 1
