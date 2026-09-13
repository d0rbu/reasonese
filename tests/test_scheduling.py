from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from email.utils import format_datetime
from threading import Event, Lock
from time import monotonic, sleep
from typing import cast

import pytest
import requests

from reasonese.openrouter import (
    CompletionGroup,
    JsonObject,
    ModelRoute,
    OpenRouterClient,
    OpenRouterModelId,
    RequestsTransport,
)
from reasonese.runner import AssistantRunGroup, run_assistant_groups
from reasonese.scheduling import ModelLimiter, ModelScheduler, ScheduledRequest, retry_after_seconds
from tests.test_cache_runner_cli import _chat, _setup_with_user_content, _tool_chat


def http_error(status: int = 429, retry_after: str | None = None) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status
    if retry_after is not None:
        response.headers["Retry-After"] = retry_after
    return requests.HTTPError(f"HTTP {status}", response=response)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, 0.0),
        ("bad", 0.0),
        ("nan", 0.0),
        ("inf", 0.0),
        ("-1", 0.0),
        ("0.25", 0.25),
        ("3600", 3600.0),
    ],
)
def test_retry_after_seconds(value: str | None, expected: float) -> None:
    assert retry_after_seconds(value) == expected


def test_retry_after_http_date(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("reasonese.scheduling.time.time", lambda: 1000.0)
    assert retry_after_seconds(format_datetime(datetime.fromtimestamp(1120, UTC))) == 120
    assert retry_after_seconds(format_datetime(datetime.fromtimestamp(900, UTC))) == 0


def test_limiter_backs_off_and_recovers_without_stale_success_undoing_backoff() -> None:
    limiter = ModelLimiter(8)
    for _ in range(100):
        limiter.succeeded(0)
    assert limiter.window == 8
    limiter.limited(10, 120)
    assert limiter.window == 4
    assert limiter.ready_at == 130
    limiter.succeeded(0)
    assert limiter.window == 4
    assert limiter.interval == 1
    limiter.limited(11, 0)
    assert limiter.window == 2
    assert limiter.interval == 2
    assert limiter.ready_at == 130
    for _ in range(100):
        limiter.succeeded(limiter.generation)
    assert limiter.window == 8
    assert limiter.interval == 0
    assert limiter.ready_at == 130
    for _ in range(100):
        limiter.limited(130, 0)
    assert limiter.window == 1
    assert limiter.interval == 30


def test_bounded_retry_honors_provider_delay_and_preserves_client_state() -> None:
    clock = Clock()
    attempts: list[float] = []
    received: list[str] = []
    scheduler = ModelScheduler(4, 2, clock.sleep, clock.monotonic)

    def send() -> str:
        attempts.append(clock.now)
        if len(attempts) < 3:
            raise http_error(retry_after="120")
        return "done"

    scheduler.run((ScheduledRequest("model/a", send, received.append),))
    assert attempts == [0, 120, 240]
    assert received == ["done"]
    limiter = scheduler.limiters["model/a"]
    assert limiter.interval == pytest.approx(1.6)
    scheduler.run((ScheduledRequest("model/a", send, received.append),))
    assert attempts[-1] == 242
    assert scheduler.limiters["model/a"] is limiter


@pytest.mark.parametrize("retries", [0, 2])
def test_exhausted_model_does_not_abort_healthy_queued_work(retries: int) -> None:
    clock = Clock()
    attempts: Counter[str] = Counter()
    received: list[int] = []
    scheduler = ModelScheduler(1, retries, clock.sleep, clock.monotonic)

    def limited() -> int:
        attempts["limited"] += 1
        raise http_error()

    def healthy() -> int:
        attempts["healthy"] += 1
        return attempts["healthy"]

    with pytest.raises(requests.HTTPError, match="429"):
        scheduler.run(
            [
                ScheduledRequest("model/limited", limited, received.append),
                ScheduledRequest("model/limited", limited, received.append),
                *(ScheduledRequest("model/healthy", healthy, received.append) for _ in range(8)),
            ]
        )
    assert attempts == {"limited": retries + 1, "healthy": 8}
    assert received == list(range(1, 9))


@pytest.mark.parametrize(
    "error",
    [
        http_error(403),
        http_error(500),
        requests.Timeout(),
        requests.HTTPError("no response"),
        ValueError("bad JSON"),
    ],
)
def test_other_failures_are_not_retried(error: Exception) -> None:
    attempts: list[bool] = []

    def send() -> str:
        attempts.append(True)
        raise error

    with pytest.raises(type(error)):
        ModelScheduler().run((ScheduledRequest("model/a", send, lambda _: None),))
    assert attempts == [True]


def test_independent_model_capacity_and_successful_ramp() -> None:
    active: Counter[str] = Counter()
    peaks: Counter[str] = Counter()
    starts: Counter[str] = Counter()
    first_wave: dict[str, int] = {}
    other_started = Event()
    lock = Lock()
    scheduler = ModelScheduler(4)

    def send(model: str) -> str:
        with lock:
            starts[model] += 1
            active[model] += 1
            peaks[model] = max(peaks[model], active[model])
            if model == "b":
                other_started.set()
        assert other_started.wait(2), "first model monopolized worker capacity"
        sleep(0.015)
        with lock:
            first_wave.setdefault(model, starts[model])
            active[model] -= 1
        return model

    scheduler.run(
        ScheduledRequest(model, lambda model=model: send(model), lambda _: None)
        for model in ("a", "b")
        for _ in range(30)
    )
    assert first_wave == {"a": 2, "b": 2}
    assert peaks == {"a": 4, "b": 4}


def test_model_cooldown_wakes_while_another_model_is_still_in_flight() -> None:
    retry_started = Event()
    attempts = 0
    scheduler = ModelScheduler(1)
    # Shorten only the policy interval for this real-clock wakeup test.
    limiter = ModelLimiter(1)
    limiter.ready_at = monotonic() + 0.04
    scheduler.limiters["cooling"] = limiter

    def cooling() -> bool:
        nonlocal attempts
        attempts += 1
        retry_started.set()
        return True

    def slow() -> bool:
        assert retry_started.wait(2), "scheduler waited for a slow future past the cooldown"
        return True

    scheduler.run(
        (
            ScheduledRequest("cooling", cooling, lambda _: None),
            ScheduledRequest("slow", slow, lambda _: None),
        )
    )
    assert attempts == 1


def test_real_transport_reports_429_to_client_and_retries_exact_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    calls: list[JsonObject] = []

    def post(self: requests.Session, url: str, **kwargs: object) -> requests.Response:
        body = kwargs["json"]
        assert isinstance(body, dict)
        calls.append(cast(JsonObject, body))
        response = requests.Response()
        response.status_code = 429 if len(calls) == 1 else 200
        response.headers["Retry-After"] = "3600"
        response._content = b'{"answer": "done"}'
        return response

    monkeypatch.setattr(requests.Session, "post", post)
    client = OpenRouterClient(
        RequestsTransport("test-key"), sleep=clock.sleep, monotonic=clock.monotonic
    )
    model = OpenRouterModelId.parse("google/gemma-4-31b-it:free")
    assert client.complete(model, {"messages": []}) == {"answer": "done"}
    assert clock.sleeps == [3600]
    assert calls == [{"messages": [], "model": str(model)}] * 2


def test_grouped_authoring_keeps_healthy_model_moving_during_429() -> None:
    clock = Clock()
    events: list[str] = []
    lock = Lock()

    class Transport:
        def post_json(self, path: str, body: JsonObject) -> JsonObject:
            with lock:
                model = body["model"]
                events.append(model)
                if model == "example/limited:free" and events.count(model) == 1:
                    raise http_error(retry_after="60")
                return {"content": body["index"]}

        def get_json(self, path: str) -> JsonObject:
            raise AssertionError(path)

    client = OpenRouterClient(
        Transport(), sync_workers=1, sleep=clock.sleep, monotonic=clock.monotonic
    )
    models = ("example/limited:free", "example/healthy:free")
    groups = tuple(
        CompletionGroup(
            ModelRoute(OpenRouterModelId.parse(model), None), tuple({"index": i} for i in range(5))
        )
        for model in models
    )
    responses = client.complete_many_grouped(groups, prefer_batch=False)
    assert responses == (tuple({"content": i} for i in range(5)),) * 2
    assert events[1:6] == [models[1]] * 5
    assert client.scheduler.limiters[models[0]].generation == 1
    assert client.scheduler.limiters[models[1]].generation == 0


def test_empty_scheduler_has_no_model_state() -> None:
    scheduler = ModelScheduler()
    scheduler.run(())
    assert scheduler.limiters == {}


def test_continuation_cannot_change_model() -> None:
    continuation = ScheduledRequest("other", lambda: "done", lambda _: None)
    with pytest.raises(ValueError, match="retain its model"):
        ModelScheduler().run((ScheduledRequest("first", lambda: "done", lambda _: continuation),))


def test_tool_continuation_uses_model_cooldown_and_a_fresh_retry_budget() -> None:
    clock = Clock()
    calls: list[JsonObject] = []
    healthy_finished = Event()

    class Transport:
        def post_json(self, path: str, body: JsonObject) -> JsonObject:
            if body["model"] == "example/healthy:free":
                healthy_finished.set()
                return _chat("healthy final")
            calls.append(body)
            if len(calls) in (1, 3):
                raise http_error(retry_after="60")
            assert healthy_finished.is_set()
            return _tool_chat() if len(calls) == 2 else _chat("limited final")

        def get_json(self, path: str) -> JsonObject:
            raise AssertionError(path)

    client = OpenRouterClient(
        Transport(),
        sync_workers=1,
        rate_limit_retries=1,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )
    setup = _setup_with_user_content("request")
    traces = run_assistant_groups(
        tuple(
            AssistantRunGroup(ModelRoute(OpenRouterModelId.parse(model), None), (setup,))
            for model in ("example/limited:free", "example/healthy:free")
        ),
        client,
    )
    assert calls[0] == calls[1]
    assert calls[2] == calls[3]
    assert calls[2]["messages"][-1]["role"] == "tool"
    assert len(traces[0][0].tool_steps) == 1
    assert traces[0][0].response == _chat("limited final")
    assert traces[1][0].response == _chat("healthy final")


def test_batch_submission_429_does_not_hold_synchronous_model() -> None:
    clock = Clock()
    healthy_finished = Event()
    submitted: list[JsonObject] = []

    class Transport:
        def post_json(self, path: str, body: JsonObject) -> JsonObject:
            if path == "/api/v1/chat/completions":
                healthy_finished.set()
                return {"content": "sync"}
            submitted.append(body)
            if len(submitted) == 1:
                raise http_error()
            assert healthy_finished.is_set()
            return {
                "id": "batch",
                "status": "completed",
                "results": [
                    {
                        "custom_id": "request-0",
                        "response": {"status_code": 200, "body": {"content": "batch"}},
                    }
                ],
            }

        def get_json(self, path: str) -> JsonObject:
            raise AssertionError(path)

    client = OpenRouterClient(
        Transport(), sync_workers=1, sleep=clock.sleep, monotonic=clock.monotonic
    )
    responses = client.complete_many_grouped(
        (
            CompletionGroup(
                ModelRoute(
                    OpenRouterModelId.parse("example/batch"),
                    OpenRouterModelId.parse("example/batch:batch"),
                ),
                ({},),
            ),
            CompletionGroup(ModelRoute(OpenRouterModelId.parse("example/sync"), None), ({},)),
        ),
        prefer_batch=True,
    )
    assert responses == (({"content": "batch"},), ({"content": "sync"},))
    assert submitted[0] == submitted[1]
