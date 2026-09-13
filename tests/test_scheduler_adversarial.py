"""Offline scheduling regressions with controlled completion order, without timing races."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ALL_COMPLETED, Future, ThreadPoolExecutor, wait
from queue import Queue
from random import Random
from threading import Event, Lock

import pytest
import requests

from reasonese.openrouter import (
    CompletionGroup,
    JsonObject,
    ModelRoute,
    OpenRouterClient,
    OpenRouterModelId,
)
from reasonese.scheduling import ModelLimiter, ModelScheduler, ScheduledRequest
from tests.test_scheduling import Clock, http_error


@pytest.fixture
def accelerated_clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    """Keep real worker/future behavior while making policy cooldowns instantaneous."""
    clock = Clock()

    def virtual_wait[Result](
        futures: Iterable[Future[Result]],
        *,
        timeout: float | None,
        return_when: str,
    ) -> tuple[set[Future[Result]], set[Future[Result]]]:
        if timeout is not None:
            clock.sleep(timeout)
            timeout = 0.0
        return wait(futures, timeout=timeout, return_when=return_when)

    monkeypatch.setattr("reasonese.scheduling.wait", virtual_wait)
    return clock


@pytest.mark.parametrize("other_inflight", [False, True])
def test_huge_finite_retry_after_retains_deadline_and_bounds_platform_waits(
    monkeypatch: pytest.MonkeyPatch, other_inflight: bool
) -> None:
    class StopAfterBoundedWait(Exception):
        pass

    peer_started = Event()
    release_peer = Event()
    observed: list[tuple[str, float]] = []
    attempts: list[str] = []

    def check_timeout(kind: str, seconds: float | None) -> None:
        # Release before assertions or the sentinel: executor shutdown must always
        # be able to join the peer, including when a broken timeout is detected.
        release_peer.set()
        assert seconds is not None and 0 < seconds <= 60, "the platform wait must be chunked"
        assert scheduler.limiters["limited"].ready_at == 1e300
        observed.append((kind, seconds))
        raise StopAfterBoundedWait

    def bounded_sleep(seconds: float) -> None:
        check_timeout("sleep", seconds)

    scheduler = ModelScheduler(1, 1, bounded_sleep, lambda: 0.0)

    def bounded_wait[Result](
        futures: Iterable[Future[Result]],
        *,
        timeout: float | None,
        return_when: str,
    ) -> tuple[set[Future[Result]], set[Future[Result]]]:
        if scheduler.limiters["limited"].generation:
            check_timeout("wait", timeout)
        return wait(futures, timeout=timeout, return_when=return_when)

    def limited() -> str:
        attempts.append("limited")
        if other_inflight:
            assert peer_started.wait(5)
        raise http_error(retry_after="1e300")

    def peer() -> str:
        peer_started.set()
        assert release_peer.wait(5), "the bounded-wait observer never released the peer"
        return "peer"

    requests_to_run = [ScheduledRequest("limited", limited, lambda _: None)]
    if other_inflight:
        requests_to_run.append(ScheduledRequest("peer", peer, lambda _: None))
    monkeypatch.setattr("reasonese.scheduling.wait", bounded_wait)
    try:
        with pytest.raises(StopAfterBoundedWait):
            scheduler.run(requests_to_run)
    finally:
        release_peer.set()

    assert attempts == ["limited"], "the enormous cooldown must not trigger an early retry"
    assert len(observed) == 1
    assert observed[0][0] == ("wait" if other_inflight else "sleep")
    assert scheduler.limiters["limited"].ready_at == 1e300


@pytest.mark.parametrize("peer_is_limited", [False, True])
def test_exhaustion_collects_late_success_without_restarting_failed_model(
    monkeypatch: pytest.MonkeyPatch, accelerated_clock: Clock, peer_is_limited: bool
) -> None:
    """An exhausted retry can finish before another attempt from the same generation."""
    clock = accelerated_clock
    scheduler = ModelScheduler(8, 1, clock.sleep, clock.monotonic)
    # Reach the concurrency needed for a retry alongside an earlier in-flight request.
    scheduler.run(ScheduledRequest("limited", lambda: None, lambda _: None) for _ in range(100))
    assert scheduler.limiters["limited"].window == 8

    release_peer = Event()
    peer_started = Event()
    attempts: Counter[str] = Counter()
    received: list[str] = []
    errors: list[requests.HTTPError] = []
    original_limited = ModelLimiter.limited

    def observe_backoff(self: ModelLimiter, now: float, retry_after: float) -> None:
        original_limited(self, now, retry_after)
        if self is scheduler.limiters["limited"] and self.generation == 2:
            # The exhausted future is already in the scheduler's completed set. The
            # peer cannot be in that set because it has not yet been released.
            release_peer.set()

    monkeypatch.setattr(ModelLimiter, "limited", observe_backoff)

    def limited() -> str:
        attempts["limited"] += 1
        assert peer_started.wait(5), "the peer must be in flight before backoff"
        error = http_error()
        errors.append(error)
        raise error

    def peer() -> str:
        attempts["peer"] += 1
        peer_started.set()
        assert release_peer.wait(5), "an in-flight peer blocked the retry"
        if peer_is_limited:
            raise http_error()
        return "completed peer"

    def collect_peer(response: str) -> ScheduledRequest[str]:
        received.append(response)

        def unexpected_continuation() -> str:
            pytest.fail("exhausted models must not start a late continuation")

        return ScheduledRequest("limited", unexpected_continuation, received.append)

    with pytest.raises(requests.HTTPError) as raised:
        scheduler.run(
            (
                ScheduledRequest("limited", limited, received.append),
                ScheduledRequest("limited", peer, collect_peer),
                *(
                    ScheduledRequest("healthy", lambda i=i: f"healthy-{i}", received.append)
                    for i in range(12)
                ),
            )
        )

    assert raised.value is errors[-1]
    assert attempts == {"limited": 2, "peer": 1}
    expected = {f"healthy-{i}" for i in range(12)}
    if not peer_is_limited:
        expected.add("completed peer")
    assert set(received) == expected
    assert len(received) == len(expected)
    assert scheduler.limiters["limited"].generation == 2 + peer_is_limited
    assert scheduler.limiters["healthy"].generation == 0


def test_simultaneous_success_cannot_soften_an_observed_429(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both responses are ready together, with success earlier in submission order."""
    second_started = Event()
    received: list[str] = []
    scheduler = ModelScheduler(2, 0)

    def complete_wave[Result](
        futures: Iterable[Future[Result]],
        *,
        timeout: float | None,
        return_when: str,
    ) -> tuple[set[Future[Result]], set[Future[Result]]]:
        # Join the already-admitted first wave to make simultaneity deterministic.
        # No subsequent admission is needed after the model exhausts its budget.
        return wait(futures, timeout=5, return_when=ALL_COMPLETED)

    def successful() -> str:
        assert second_started.wait(5)
        return "already completed"

    def limited() -> str:
        second_started.set()
        raise http_error()

    monkeypatch.setattr("reasonese.scheduling.wait", complete_wave)
    with pytest.raises(requests.HTTPError):
        scheduler.run(
            (
                ScheduledRequest("model", successful, received.append),
                ScheduledRequest("model", limited, received.append),
            )
        )

    assert received == ["already completed"]
    limiter = scheduler.limiters["model"]
    assert limiter.generation == 1
    assert limiter.window == 1
    assert limiter.interval == 1


def test_slow_callback_preserves_other_model_capacity_and_continuation_priority() -> None:
    """A local tool can wait for healthy work without occupying its model's workers."""
    scheduler = ModelScheduler(1)
    healthy_finished = Event()
    healthy_results: list[int] = []
    slow_starts: list[str] = []

    def slow_send(kind: str) -> str:
        slow_starts.append(kind)
        if kind != "initial":
            assert healthy_finished.is_set(), "processing did not retain its model's slot"
        return kind

    def process_slow(response: str) -> ScheduledRequest[str] | None:
        if response == "initial":
            assert healthy_finished.wait(5), "a callback blocked another model's queue"
            return ScheduledRequest("slow", lambda: slow_send("continuation"), process_slow)
        return None

    def process_healthy(response: int) -> None:
        healthy_results.append(response)
        if len(healthy_results) == 8:
            healthy_finished.set()

    scheduler.run(
        (
            ScheduledRequest("slow", lambda: slow_send("initial"), process_slow),
            ScheduledRequest("slow", lambda: slow_send("queued"), process_slow),
            *(ScheduledRequest("healthy", lambda i=i: i, process_healthy) for i in range(8)),
        )
    )

    assert healthy_results == list(range(8))
    assert slow_starts == ["initial", "continuation", "queued"]


@pytest.mark.parametrize("workers", [1, 2, 8])
@pytest.mark.parametrize("seed", [0, 93])
def test_interleaved_model_retries_and_continuations_are_lossless(
    accelerated_clock: Clock, workers: int, seed: int
) -> None:
    """Every logical step gets its own retry budget, result, and predecessor context."""
    clock = accelerated_clock
    scheduler = ModelScheduler(workers, 2, clock.sleep, clock.monotonic)
    models = ("model/a", "model/b", "model/c")
    random = Random(seed)
    failures = {
        (model, chain, step): random.randrange(3)
        for model in models
        for chain in range(10)
        for step in range(3)
    }
    attempts: Counter[tuple[str, int, int]] = Counter()
    completed: set[tuple[str, int, int]] = set()
    delivered: list[tuple[str, int]] = []
    lock = Lock()

    def request_for(model: str, chain: int, step: int) -> ScheduledRequest[tuple[str, int, int]]:
        key = (model, chain, step)

        def send() -> tuple[str, int, int]:
            with lock:
                attempts[key] += 1
                attempt = attempts[key]
            if attempt <= failures[key]:
                raise http_error(retry_after="3" if chain % 2 else None)
            return key

        def receive(result: tuple[str, int, int]) -> ScheduledRequest[tuple[str, int, int]] | None:
            assert result == key, "a response was associated with the wrong request"
            with lock:
                assert key not in completed, "a retry delivered a duplicate result"
                if step:
                    assert (model, chain, step - 1) in completed
                completed.add(key)
                if step == 2:
                    delivered.append((model, chain))
                    return None
            return request_for(model, chain, step + 1)

        return ScheduledRequest(model, send, receive)

    scheduler.run(request_for(model, chain, 0) for model in models for chain in range(10))

    assert dict(attempts) == {key: failures[key] + 1 for key in failures}
    assert completed == set(failures)
    assert len(delivered) == 30
    assert set(delivered) == {(model, chain) for model in models for chain in range(10)}
    for model in models:
        limiter = scheduler.limiters[model]
        assert limiter.generation == sum(
            count for key, count in failures.items() if key[0] == model
        )
        assert 1 <= limiter.window <= workers
        assert 0 <= limiter.interval <= 30


def test_duplicate_model_groups_share_capacity_and_preserve_identity() -> None:
    """Force out-of-order results while one request stays pending across both groups."""
    model = OpenRouterModelId.parse("example/shared:free")
    other_model = OpenRouterModelId.parse("example/independent:free")
    started: Queue[tuple[str, int]] = Queue()
    gates = {index: Event() for index in range(9)}
    lock = Lock()
    active: Counter[str] = Counter()
    peaks: Counter[str] = Counter()

    class Transport:
        def post_json(self, path: str, body: JsonObject) -> JsonObject:
            slug, index = str(body["model"]), int(body["index"])
            with lock:
                active[slug] += 1
                peaks[slug] = max(peaks[slug], active[slug])
            started.put((slug, index))
            try:
                assert gates[index].wait(5), f"request {index} was never released"
                return {"index": index, "model": slug, "choices": [{"message": {"content": "answer"}}]}
            finally:
                with lock:
                    active[slug] -= 1

        def get_json(self, path: str) -> JsonObject:
            raise AssertionError(path)

    client = OpenRouterClient(Transport(), sync_workers=2)
    groups = (
        CompletionGroup(ModelRoute(model, None), tuple({"index": i} for i in range(4))),
        CompletionGroup(ModelRoute(model, None), tuple({"index": i} for i in range(4, 8))),
        CompletionGroup(ModelRoute(other_model, None), ({"index": 8},)),
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(client.complete_many_grouped, groups, prefer_batch=False)
        try:
            first_wave = {started.get(timeout=5) for _ in range(3)}
            assert first_wave == {(str(model), 0), (str(model), 1), (str(other_model), 8)}
            gates[8].set()
            # Keep request 0 pending. Completion of each later request must immediately
            # admit the next, including the first request of the second same-model group.
            for index in range(1, 7):
                gates[index].set()
                assert started.get(timeout=5) == (str(model), index + 1)
                assert not result.done()
            gates[7].set()
            gates[0].set()
            responses = result.result(timeout=5)
        finally:
            for gate in gates.values():
                gate.set()

    assert peaks == {str(model): 2, str(other_model): 1}
    assert active == {str(model): 0, str(other_model): 0}
    assert responses == (
        tuple({"index": i, "model": str(model), "choices": [{"message": {"content": "answer"}}]} for i in range(4)),
        tuple({"index": i, "model": str(model), "choices": [{"message": {"content": "answer"}}]} for i in range(4, 8)),
        ({"index": 8, "model": str(other_model), "choices": [{"message": {"content": "answer"}}]},),
    )
