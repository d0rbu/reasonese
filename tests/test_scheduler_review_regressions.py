from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import Future, ThreadPoolExecutor, wait
from textwrap import dedent
from threading import Event, Lock
from typing import Any

import pytest
import requests

import reasonese.scheduling as scheduling
from reasonese.scheduling import ModelLimiter, ModelScheduler, ScheduledRequest
from tests.test_scheduling import Clock, http_error


def _limited() -> requests.HTTPError:
    return http_error(retry_after="60")


class _ImmediateExecutor(ThreadPoolExecutor):
    def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Future[Any]:
        future = super().submit(fn, *args, **kwargs)
        # Force completion before submit returns, leaving errors for the scheduler.
        finished, _ = wait((future,), timeout=2)
        assert future in finished
        return future


@pytest.mark.parametrize("timezone", ["UTC-9", "UTC+6"])
def test_all_http_date_formats_use_utc_in_non_utc_processes(timezone: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            dedent(
                """\
                import json
                import time
                from datetime import UTC, datetime
                from reasonese.scheduling import retry_after_seconds

                time.tzset()
                now = datetime(2026, 9, 12, 12, 0, tzinfo=UTC).timestamp()
                time.time = lambda: now
                values = [
                    "Sat, 12 Sep 2026 12:02:00 GMT",
                    "Saturday, 12-Sep-26 12:02:00 GMT",
                    "Sat Sep 12 12:02:00 2026",
                ]
                print(json.dumps([retry_after_seconds(value) for value in values]))
                """
            ),
        ],
        env={**os.environ, "TZ": timezone},
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert json.loads(result.stdout) == [120, 120, 120]


def test_slow_receive_does_not_block_another_models_queued_http() -> None:
    callback_started = Event()
    healthy_finished = Event()
    received: list[str] = []

    def receive_slow(result: str) -> None:
        callback_started.set()
        assert healthy_finished.wait(2), "local work blocked the other model's next request"
        received.append(result)

    def healthy_first() -> str:
        assert callback_started.wait(2)
        return "healthy first"

    def healthy_second() -> str:
        healthy_finished.set()
        return "healthy second"

    ModelScheduler(1).run(
        (
            ScheduledRequest("slow", lambda: "slow result", receive_slow),
            ScheduledRequest("healthy", healthy_first, received.append),
            ScheduledRequest("healthy", healthy_second, received.append),
        )
    )
    assert sorted(received) == ["healthy first", "healthy second", "slow result"]


def test_completed_429_is_drained_before_admitting_fresh_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_bad = Event()
    bad_registered = Event()
    bad_future: list[Future[Any]] = []
    admitted: list[str] = []

    def bad() -> str:
        assert release_bad.wait(2)
        raise _limited()

    def good() -> str:
        return "good"

    def fresh() -> str:
        return "must not be admitted"

    class RecordingExecutor(ThreadPoolExecutor):
        def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Future[Any]:
            if fn in (bad, good, fresh):
                admitted.append(fn.__name__)
            future = super().submit(fn, *args, **kwargs)
            if fn is bad:
                bad_future.append(future)
                bad_registered.set()
            return future

    def receive_good(result: str) -> None:
        assert result == "good"
        assert bad_registered.wait(2)
        release_bad.set()
        finished, _ = wait(bad_future, timeout=2)
        assert finished == set(bad_future)

    monkeypatch.setattr(scheduling, "ThreadPoolExecutor", RecordingExecutor)
    with pytest.raises(requests.HTTPError, match="429"):
        ModelScheduler(2, 0).run(
            (
                ScheduledRequest("same", bad, lambda _: None),
                ScheduledRequest("same", good, receive_good),
                ScheduledRequest("same", fresh, lambda _: None),
            )
        )
    assert admitted == ["bad", "good"]


def test_immediate_responses_do_not_starve_another_ready_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    starts: list[str] = []

    def send(model: str) -> str:
        starts.append(model)
        return model

    monkeypatch.setattr(scheduling, "ThreadPoolExecutor", _ImmediateExecutor)
    ModelScheduler(1).run(
        (
            *(ScheduledRequest("first", lambda: send("first"), lambda _: None) for _ in range(8)),
            ScheduledRequest("second", lambda: send("second"), lambda _: None),
        )
    )
    assert starts.index("second") < 8, "first model drained its queue before second model started"
    assert starts.count("first") == 8
    assert starts.count("second") == 1


@pytest.mark.parametrize("separate_runs", [False, True])
def test_reusing_request_spec_preserves_independent_retry_budgets(separate_runs: bool) -> None:
    clock = Clock()
    attempts: list[int] = []
    received: list[str] = []

    def send() -> str:
        attempts.append(len(attempts) + 1)
        if len(attempts) % 2:
            raise _limited()
        return "done"

    scheduler = ModelScheduler(1, 1, clock.sleep, clock.monotonic)
    request = ScheduledRequest("model", send, received.append)
    if separate_runs:
        scheduler.run((request,))
        scheduler.run((request,))
    else:
        scheduler.run((request, request))
    assert attempts == [1, 2, 3, 4]
    assert received == ["done", "done"]


def test_callback_429_never_retries_an_already_successful_http_request() -> None:
    attempts: list[bool] = []
    callbacks: list[str] = []
    error = _limited()

    def send() -> str:
        attempts.append(True)
        return "successful HTTP result"

    def receive(result: str) -> None:
        callbacks.append(result)
        raise error

    scheduler = ModelScheduler()
    with pytest.raises(requests.HTTPError) as raised:
        scheduler.run((ScheduledRequest("model", send, receive),))
    assert raised.value is error
    assert attempts == [True]
    assert callbacks == ["successful HTTP result"]
    assert scheduler.limiters["model"].generation == 0


def test_exhausted_model_delivers_late_success_without_sending_its_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    good_started = Event()
    failure_recorded = Event()
    received: list[str] = []
    continuation_calls: list[bool] = []
    original_limited = ModelLimiter.limited

    def record_limit(self: ModelLimiter, now: float, retry_after: float) -> None:
        original_limited(self, now, retry_after)
        failure_recorded.set()

    monkeypatch.setattr(ModelLimiter, "limited", record_limit)

    def bad() -> str:
        assert good_started.wait(2)
        raise _limited()

    def good() -> str:
        good_started.set()
        assert failure_recorded.wait(2)
        return "late success"

    def continuation() -> str:
        continuation_calls.append(True)
        return "unreachable"

    def receive(result: str) -> ScheduledRequest[str]:
        received.append(result)
        return ScheduledRequest("same", continuation, lambda _: None)

    with pytest.raises(requests.HTTPError, match="429"):
        ModelScheduler(2, 0).run(
            (
                ScheduledRequest("same", bad, lambda _: None),
                ScheduledRequest("same", good, receive),
            )
        )
    assert received == ["late success"]
    assert continuation_calls == []


def test_already_completed_fatal_error_prevents_other_models_first_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[str] = []
    error = ValueError("fatal response")

    def fatal() -> str:
        attempts.append("fatal")
        raise error

    def fresh() -> str:
        attempts.append("fresh")
        return "must not be submitted"

    monkeypatch.setattr(scheduling, "ThreadPoolExecutor", _ImmediateExecutor)
    with pytest.raises(ValueError) as raised:
        ModelScheduler(1).run(
            (
                ScheduledRequest("fatal", fatal, lambda _: None),
                ScheduledRequest("other", fresh, lambda _: None),
            )
        )
    assert raised.value is error
    assert attempts == ["fatal"]


def test_fatal_error_drains_inflight_success_but_stops_fresh_retries_and_continuations() -> None:
    all_started = Event()
    fatal_observed = Event()
    lock = Lock()
    attempts: list[str] = []
    received: list[str] = []
    response = requests.Response()
    response.status_code = 403

    class ObservedHTTPError(requests.HTTPError):
        def __getattribute__(self, name: str) -> Any:
            # Release late work only once the scheduler classifies the fatal HTTP error.
            if name == "response":
                fatal_observed.set()
            return super().__getattribute__(name)

    fatal_error = ObservedHTTPError("fatal 403", response=response)
    callback_error = ValueError("later callback failure")

    def started(model: str) -> None:
        with lock:
            attempts.append(model)
            if len(attempts) == 4:
                all_started.set()

    def fatal() -> str:
        started("fatal")
        assert all_started.wait(2)
        raise fatal_error

    def limited() -> str:
        started("limited")
        assert all_started.wait(2)
        raise _limited()

    def continuing() -> str:
        started("continuing")
        return "continuing success"

    def late() -> str:
        started("late")
        assert fatal_observed.wait(2)
        return "late success"

    def forbidden() -> str:
        attempts.append("forbidden")
        return "must not be submitted"

    def receive_continuation(result: str) -> ScheduledRequest[str]:
        assert fatal_observed.wait(2)
        received.append(result)
        return ScheduledRequest("continuing", forbidden, lambda _: None)

    def receive_late(result: str) -> None:
        received.append(result)
        raise callback_error

    initial = (
        ScheduledRequest("fatal", fatal, lambda _: None),
        ScheduledRequest("limited", limited, lambda _: None),
        ScheduledRequest("continuing", continuing, receive_continuation),
        ScheduledRequest("late", late, receive_late),
    )
    fresh = tuple(ScheduledRequest(request.model, forbidden, lambda _: None) for request in initial)
    with pytest.raises(requests.HTTPError) as raised:
        ModelScheduler(1, 1).run((*initial, *fresh))
    assert raised.value is fatal_error
    assert sorted(attempts) == ["continuing", "fatal", "late", "limited"]
    assert sorted(received) == ["continuing success", "late success"]
