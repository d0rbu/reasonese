"""Completion-driven requests with independent adaptive limits for each model."""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import UTC
from email.utils import parsedate_to_datetime
from math import isfinite
from typing import cast

import requests
from beartype import beartype

_LOGGER = logging.getLogger(__name__)


def retry_after_seconds(value: str | None) -> float:
    """Read delta seconds or an HTTP date without shortening a provider cooldown."""
    if value is None:
        return 0.0
    try:
        delay = float(value)
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=UTC)
            delay = date.timestamp() - time.time()
        except (ValueError, TypeError, OverflowError):
            return 0.0
    return max(0.0, delay) if isfinite(delay) else 0.0


@dataclass(slots=True)
class ModelLimiter:
    """Additive concurrency growth and multiplicative backoff, owned by the scheduler."""

    ceiling: int
    window: float = field(init=False)
    interval: float = 0.0
    ready_at: float = 0.0
    generation: int = 0

    def __post_init__(self) -> None:
        self.window = float(min(2, self.ceiling))

    def succeeded(self, generation: int) -> None:
        # Requests already in flight when a 429 arrived cannot undo its backoff.
        if generation != self.generation:
            return
        self.window = min(float(self.ceiling), self.window + 1.0 / self.window)
        self.interval *= 0.8
        if self.interval < 0.01:
            self.interval = 0.0

    def limited(self, now: float, retry_after: float) -> None:
        self.generation += 1
        self.window = max(1.0, self.window / 2.0)
        self.interval = min(30.0, max(1.0, self.interval * 2.0))
        self.ready_at = max(self.ready_at, now + max(self.interval, retry_after))


@dataclass(frozen=True, slots=True)
class ScheduledRequest[Result]:
    """One attempt and a callback that may produce a priority continuation."""

    model: str
    send: Callable[[], Result]
    receive: Callable[[Result], ScheduledRequest[Result] | None]


@dataclass(frozen=True, slots=True)
class _Work[Result]:
    request: ScheduledRequest[Result]
    retries: int = 0


@beartype
@dataclass(slots=True)
class ModelScheduler:
    """One synchronous caller, asynchronous attempts and callbacks, per-model limits.

    Limits persist across sequential stages on this scheduler. Concurrent invocations
    should use separate clients; limits are not shared between processes or API keys.
    """

    max_concurrency: int = 8
    rate_limit_retries: int = 3
    sleep: Callable[[float], None] = field(default=time.sleep, repr=False)
    monotonic: Callable[[], float] = field(default=time.monotonic, repr=False)
    limiters: dict[str, ModelLimiter] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if isinstance(self.max_concurrency, bool) or self.max_concurrency < 1:
            raise ValueError("sync workers must be a positive integer")
        if isinstance(self.rate_limit_retries, bool) or self.rate_limit_retries < 0:
            raise ValueError("rate-limit retries must be a non-negative integer")

    def run[Result](self, requests_to_run: Iterable[ScheduledRequest[Result]]) -> None:
        """Run immutable requests; callbacks may run concurrently across requests.

        Each model's active count includes its response processing, so slow local
        tools cannot consume the worker capacity reserved for another model.
        """
        queues: dict[str, deque[_Work[Result]]] = {}
        for request in requests_to_run:
            queues.setdefault(request.model, deque()).append(_Work(request))
            if request.model not in self.limiters:
                self.limiters[request.model] = ModelLimiter(self.max_concurrency)
        if not queues:
            return

        active = dict.fromkeys(queues, 0)
        failed: dict[str, Exception] = {}
        failure: Exception | None = None

        def stop_admission(error: Exception) -> None:
            nonlocal failure
            if failure is None:
                failure = error
            for queue in queues.values():
                queue.clear()

        pending: dict[Future[Result], tuple[_Work[Result], int]] = {}
        processing: dict[Future[ScheduledRequest[Result] | None], _Work[Result]] = {}
        worker_count = sum(min(self.max_concurrency, len(queue)) for queue in queues.values())
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            while pending or processing or any(queues.values()):
                done = [future for future in pending if future.done()]
                processed = [future for future in processing if future.done()]
                if done or processed:
                    responses: list[tuple[_Work[Result], int, Result]] = []
                    # Apply every observed 429 before crediting successes or admitting work.
                    for future in done:
                        work, generation = pending.pop(future)
                        model = work.request.model
                        limiter = self.limiters[model]
                        try:
                            result = future.result()
                        except Exception as error:
                            active[model] -= 1
                            response = (
                                error.response if isinstance(error, requests.HTTPError) else None
                            )
                            if response is None or response.status_code != 429:
                                stop_admission(error)
                                continue
                            limiter.limited(
                                self.monotonic(),
                                retry_after_seconds(response.headers.get("Retry-After")),
                            )
                            _LOGGER.warning(
                                "HTTP 429 for %s: concurrency=%d, interval=%.2fs, cooldown=%.2fs",
                                model,
                                int(limiter.window),
                                limiter.interval,
                                max(0.0, limiter.ready_at - self.monotonic()),
                            )
                            if work.retries >= self.rate_limit_retries:
                                failed.setdefault(model, error)
                                queues[model].clear()
                            elif model not in failed and failure is None:
                                queues[model].appendleft(_Work(work.request, work.retries + 1))
                        else:
                            responses.append((work, generation, result))
                    for work, generation, result in responses:
                        self.limiters[work.request.model].succeeded(generation)
                        processing[executor.submit(work.request.receive, result)] = work
                    for future in processed:
                        work = processing.pop(future)
                        model = work.request.model
                        active[model] -= 1
                        # Callback failures are not failed HTTP attempts and must never retry.
                        try:
                            continuation = future.result()
                        except Exception as error:
                            stop_admission(error)
                            continue
                        if continuation is not None:
                            if not isinstance(continuation, ScheduledRequest):
                                stop_admission(
                                    TypeError("a callback must return a request or None")
                                )
                            elif continuation.model != model:
                                stop_admission(ValueError("a continuation must retain its model"))
                            elif model not in failed and failure is None:
                                queues[model].appendleft(_Work(continuation))
                    # Futures may have finished while responses were being processed.
                    continue

                for model, queue in queues.items():
                    limiter = self.limiters[model]
                    while queue and active[model] < int(limiter.window):
                        feedback = (
                            *(
                                (future, work.request.model)
                                for future, (work, _) in pending.items()
                            ),
                            *((future, work.request.model) for future, work in processing.items()),
                        )
                        if any(
                            future.done()
                            and (
                                owner == model
                                or future.cancelled()
                                or future.exception() is not None
                            )
                            for future, owner in feedback
                        ):
                            break
                        now = self.monotonic()
                        if now < limiter.ready_at:
                            break
                        work = queue.popleft()
                        pending[executor.submit(work.request.send)] = (work, limiter.generation)
                        active[model] += 1
                        limiter.ready_at = now + limiter.interval

                delays = [
                    max(0.0, self.limiters[model].ready_at - self.monotonic())
                    for model, queue in queues.items()
                    if queue and active[model] < int(self.limiters[model].window)
                ]
                # Chunk waits, not cooldown deadlines: huge valid headers must not overflow
                # platform sleep/condition timeouts or cause an early retry.
                timeout = min(60.0, *delays) if delays else None
                futures = (*pending, *processing)
                if futures:
                    wait(
                        cast(tuple[Future[object], ...], futures),
                        timeout=timeout,
                        return_when=FIRST_COMPLETED,
                    )
                else:
                    self.sleep(min(60.0, *delays))
        if failure is not None:
            raise failure
        if failed:
            raise next(iter(failed.values()))
