"""Completion-driven requests with independent adaptive limits for each model."""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from math import isfinite

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
            delay = parsedate_to_datetime(value).timestamp() - time.time()
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


@dataclass(slots=True)
class ScheduledRequest[Result]:
    """One attempt and a callback that may produce a priority continuation."""

    model: str
    send: Callable[[], Result]
    receive: Callable[[Result], ScheduledRequest[Result] | None]
    retries: int = 0


@beartype
@dataclass(slots=True)
class ModelScheduler:
    """One synchronous caller, asynchronous HTTP workers, no sleeping retry workers.

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
        queues: dict[str, deque[ScheduledRequest[Result]]] = {}
        for request in requests_to_run:
            queues.setdefault(request.model, deque()).append(request)
            if request.model not in self.limiters:
                self.limiters[request.model] = ModelLimiter(self.max_concurrency)
        if not queues:
            return

        active = dict.fromkeys(queues, 0)
        failed: dict[str, requests.HTTPError] = {}
        pending: dict[Future[Result], tuple[ScheduledRequest[Result], int]] = {}
        worker_count = sum(min(self.max_concurrency, len(queue)) for queue in queues.values())
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            while pending or any(queues.values()):
                for model, queue in queues.items():
                    limiter = self.limiters[model]
                    while queue and active[model] < int(limiter.window):
                        now = self.monotonic()
                        if now < limiter.ready_at:
                            break
                        request = queue.popleft()
                        pending[executor.submit(request.send)] = (request, limiter.generation)
                        active[model] += 1
                        limiter.ready_at = now + limiter.interval

                delays = [
                    max(0.0, self.limiters[model].ready_at - self.monotonic())
                    for model, queue in queues.items()
                    if queue and active[model] < int(self.limiters[model].window)
                ]
                timeout = min(delays) if delays else None
                if not pending:
                    if timeout is not None:
                        self.sleep(timeout)
                    continue
                finished, _ = wait(pending, timeout=timeout, return_when=FIRST_COMPLETED)
                # Handle all completed attempts before admitting fresh work.
                for future in tuple(pending):
                    if future not in finished:
                        continue
                    request, generation = pending.pop(future)
                    model = request.model
                    active[model] -= 1
                    limiter = self.limiters[model]
                    try:
                        result = future.result()
                    except requests.HTTPError as error:
                        response = error.response
                        if response is None or response.status_code != 429:
                            raise
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
                        if request.retries >= self.rate_limit_retries:
                            failed[model] = error
                            queues[model].clear()
                        elif model not in failed:
                            request.retries += 1
                            queues[model].appendleft(request)
                    else:
                        limiter.succeeded(generation)
                        if model not in failed:
                            continuation = request.receive(result)
                            if continuation is not None:
                                if continuation.model != model:
                                    raise ValueError("a continuation must retain its model")
                                queues[model].appendleft(continuation)
        if failed:
            # An exhausted model must not abort healthy models' queued work.
            raise next(iter(failed.values()))
