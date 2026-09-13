"""Small OpenRouter transport with synchronous and batch completion support."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from threading import get_ident, local
from typing import Any, Protocol, cast, runtime_checkable

import requests
from beartype import beartype
from phantom import Phantom

from reasonese.axes import Assistant, Author
from reasonese.scheduling import ModelScheduler, ScheduledRequest

JsonObject = dict[str, Any]
_TERMINAL_BATCH_STATUSES = frozenset({"completed", "failed", "cancelled", "expired"})


def _batch_compatible(body: JsonObject) -> bool:
    """Return whether a request avoids server tools rejected by OpenRouter batches."""
    tools = body.get("tools")
    if not isinstance(tools, list):
        return True
    return not any(
        isinstance(tool, dict)
        and isinstance(tool.get("type"), str)
        and tool["type"].startswith("openrouter:")
        for tool in tools
    )


def _is_model_id(value: str) -> bool:
    return bool(value) and value == value.strip() and "/" in value


class OpenRouterModelId(str, Phantom[str], predicate=_is_model_id, bound=str):
    """A non-empty OpenRouter model slug."""


@beartype
@dataclass(frozen=True, slots=True)
class ModelRoute:
    """Synchronous model slug and optional batch/free variants."""

    model_id: OpenRouterModelId
    batch_model_id: OpenRouterModelId | None
    free_model_id: OpenRouterModelId | None = None


class RoutePreference(StrEnum):
    """Collection routing preferences; never experimental coordinates."""

    FREE = "free"
    PAID = "paid"
    BATCH = "batch"


class CompletionTransport(StrEnum):
    SYNC = "sync"
    BATCH = "batch"


@beartype
@dataclass(frozen=True, slots=True)
class RouteProvenance:
    """The actual requested slug and endpoint kind, independently of response metadata."""

    requested_model_id: OpenRouterModelId
    transport: CompletionTransport

    def to_dict(self) -> dict[str, str]:
        return {
            "requested_model_id": str(self.requested_model_id),
            "transport": str(self.transport),
        }


def provenance_from_dict(raw: object) -> RouteProvenance | None:
    if raw is None:
        return None
    if not isinstance(raw, dict) or set(raw) != {"requested_model_id", "transport"}:
        raise ValueError("route provenance has invalid fields")
    data = cast(JsonObject, raw)
    return RouteProvenance(
        OpenRouterModelId.parse(data["requested_model_id"]), CompletionTransport(data["transport"])
    )


@beartype
def canonical_model_id(slug: str) -> str:
    """Remove exactly one recognized terminal route suffix; preserve everything else."""
    for suffix in (":free", ":batch"):
        if slug.endswith(suffix):
            return slug[: -len(suffix)]
    return slug


def fingerprint_response(response: JsonObject) -> JsonObject:
    """Project provider metadata for hashing without changing the stored payload."""
    model = response.get("model")
    return {**response, "model": canonical_model_id(model)} if isinstance(model, str) else response


def completion_provenance(
    route: ModelRoute, bodies: tuple[JsonObject, ...], *, prefer_batch: bool
) -> RouteProvenance:
    batch = (
        prefer_batch
        and route.batch_model_id is not None
        and all(_batch_compatible(body) for body in bodies)
    )
    return RouteProvenance(
        route.model_id, CompletionTransport.BATCH if batch else CompletionTransport.SYNC
    )


@beartype
@dataclass(frozen=True, slots=True)
class CompletionGroup:
    """One ordered group of completion requests sharing a model route."""

    route: ModelRoute
    bodies: tuple[JsonObject, ...]


@dataclass(slots=True)
class _PendingBatch:
    """A submitted batch plus the information needed to collect its results."""

    batch_id: str
    batch: JsonObject
    body_count: int
    deadline: float


_MODEL_ROUTES: dict[Author, ModelRoute] = {
    Author.QWEN3_8_FLASH: ModelRoute(OpenRouterModelId.parse("qwen/qwen3.8-flash"), None),
    Author.QWEN3_8_2_4T: ModelRoute(
        OpenRouterModelId.parse("qwen/qwen3.8-2.4t-a95b"),
        OpenRouterModelId.parse("qwen/qwen3.8-2.4t-a95b:batch"),
    ),
    Author.INKLING: ModelRoute(
        OpenRouterModelId.parse("thinkingmachines/inkling"),
        OpenRouterModelId.parse("thinkingmachines/inkling:batch"),
        OpenRouterModelId.parse("thinkingmachines/inkling:free"),
    ),
    Author.INKLING_SMALL: ModelRoute(
        OpenRouterModelId.parse("thinkingmachines/inkling-small"),
        OpenRouterModelId.parse("thinkingmachines/inkling-small:batch"),
        OpenRouterModelId.parse("thinkingmachines/inkling-small:free"),
    ),
    Author.NEMOTRON_3_5_LIGHTNING: ModelRoute(
        OpenRouterModelId.parse("nvidia/nemotron-3.5-lightning"),
        None,
        OpenRouterModelId.parse("nvidia/nemotron-3.5-lightning:free"),
    ),
    Author.GEMMA_4_31B_IT: ModelRoute(
        OpenRouterModelId.parse("google/gemma-4-31b-it"),
        OpenRouterModelId.parse("google/gemma-4-31b-it:batch"),
        OpenRouterModelId.parse("google/gemma-4-31b-it:free"),
    ),
}


@beartype
def model_route(model: Author | Assistant) -> ModelRoute:
    """Return OpenRouter slugs for a model-backed author or assistant."""
    author = Author(model.value)
    if author is Author.USER:
        raise ValueError("the user author does not have an OpenRouter model")
    return _MODEL_ROUTES[author]


@beartype
def select_route(model: Author | Assistant, preference: RoutePreference) -> ModelRoute:
    """Resolve a preference before execution; never retry a failed free call as paid."""
    route = model_route(model)
    if preference is RoutePreference.FREE and route.free_model_id is not None:
        return ModelRoute(route.free_model_id, None)
    if preference is RoutePreference.BATCH:
        return route
    return ModelRoute(route.model_id, None)


@runtime_checkable
class JsonTransport(Protocol):
    """The network boundary used by the OpenRouter client."""

    def post_json(self, path: str, body: JsonObject) -> JsonObject: ...

    def get_json(self, path: str) -> JsonObject: ...


class RequestsTransport:
    """Authenticated requests transport that never stores the key in artifacts."""

    @beartype
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://openrouter.ai",
        timeout_seconds: float = 120.0,
    ) -> None:
        if not api_key.strip():
            raise ValueError("OpenRouter API key must not be blank")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._session = requests.Session()
        self._owner_thread = get_ident()
        self._thread_sessions = local()

    def _active_session(self) -> requests.Session:
        if get_ident() == self._owner_thread:
            return self._session
        session = getattr(self._thread_sessions, "session", None)
        if session is None:
            session = requests.Session()
            self._thread_sessions.session = session
        return cast(requests.Session, session)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _json(response: requests.Response) -> JsonObject:
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("OpenRouter returned a non-object JSON response")
        return cast(JsonObject, payload)

    @beartype
    def post_json(self, path: str, body: JsonObject) -> JsonObject:
        """Send one attempt; the client schedules model-specific retries."""
        response = self._active_session().post(
            f"{self._base_url}{path}",
            headers=self._headers(),
            json=body,
            timeout=self._timeout_seconds,
        )
        return self._json(response)

    @beartype
    def get_json(self, path: str) -> JsonObject:
        response = self._active_session().get(
            f"{self._base_url}{path}",
            headers=self._headers(),
            timeout=self._timeout_seconds,
        )
        return self._json(response)


@beartype
@dataclass(slots=True)
class OpenRouterClient:
    """Chat-completion client that prefers batch variants when available."""

    transport: JsonTransport
    poll_interval_seconds: float = 10.0
    batch_timeout_seconds: float = 86_400.0
    sync_workers: int = 8
    rate_limit_retries: int = 3
    sleep: Callable[[float], None] = field(default=time.sleep, repr=False)
    monotonic: Callable[[], float] = field(default=time.monotonic, repr=False)

    scheduler: ModelScheduler = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.scheduler = ModelScheduler(
            self.sync_workers, self.rate_limit_retries, self.sleep, self.monotonic
        )

    def completion_request(
        self,
        model_id: OpenRouterModelId,
        body: JsonObject,
        receive: Callable[[JsonObject], ScheduledRequest[JsonObject] | None],
    ) -> ScheduledRequest[JsonObject]:
        """Prepare one completion attempt for the shared per-model scheduler."""
        return ScheduledRequest(
            str(model_id),
            lambda: self.transport.post_json(
                "/api/v1/chat/completions", {**body, "model": str(model_id)}
            ),
            receive,
        )

    def complete(self, model_id: OpenRouterModelId, body: JsonObject) -> JsonObject:
        """Run one completion with the same model limit and bounded retries as groups."""
        responses: list[JsonObject] = []
        self.scheduler.run((self.completion_request(model_id, body, responses.append),))
        return responses[0]

    def complete_many(
        self,
        route: ModelRoute,
        bodies: tuple[JsonObject, ...],
        *,
        prefer_batch: bool,
    ) -> tuple[JsonObject, ...]:
        """Complete requests in one model batch when possible, otherwise synchronously."""
        return self.complete_many_grouped(
            (CompletionGroup(route, bodies),),
            prefer_batch=prefer_batch,
        )[0]

    def complete_many_grouped(
        self,
        groups: tuple[CompletionGroup, ...],
        *,
        prefer_batch: bool,
    ) -> tuple[tuple[JsonObject, ...], ...]:
        """Complete model groups while overlapping independent batch jobs."""
        results: list[list[JsonObject | None]] = [[None] * len(group.bodies) for group in groups]
        pending: dict[int, _PendingBatch] = {}
        batch_requests: list[ScheduledRequest[JsonObject]] = []
        sync_requests: list[ScheduledRequest[JsonObject]] = []

        def collect_batch(index: int, batch: JsonObject) -> None:
            batch_id = batch.get("id")
            if not isinstance(batch_id, str) or not batch_id:
                raise ValueError("OpenRouter batch response is missing an id")
            pending[index] = _PendingBatch(
                batch_id,
                batch,
                len(groups[index].bodies),
                self.monotonic() + self.batch_timeout_seconds,
            )

        def collect_sync(index: int, body_index: int, response: JsonObject) -> None:
            results[index][body_index] = response

        for index, group in enumerate(groups):
            if not group.bodies:
                continue
            if (
                completion_provenance(
                    group.route, group.bodies, prefer_batch=prefer_batch
                ).transport
                is CompletionTransport.BATCH
            ):
                batch_requests.append(
                    ScheduledRequest(
                        str(group.route.model_id),
                        lambda group=group: self._submit_batch(group.route.model_id, group.bodies),
                        lambda batch, index=index: collect_batch(index, batch),
                    )
                )
            else:
                for body_index, body in enumerate(group.bodies):
                    sync_requests.append(
                        self.completion_request(
                            group.route.model_id,
                            body,
                            lambda response, index=index, body_index=body_index: collect_sync(
                                index, body_index, response
                            ),
                        )
                    )
        self.scheduler.run((*batch_requests, *sync_requests))

        if pending:
            indexes = sorted(pending)
            completed = self._wait_for_batches(tuple(pending[index] for index in indexes))
            for index, responses in zip(indexes, completed, strict=True):
                results[index] = list(responses)

        if any(response is None for group in results for response in group):
            raise RuntimeError("completion group was not executed")  # pragma: no cover
        return cast(tuple[tuple[JsonObject, ...], ...], tuple(tuple(group) for group in results))

    def _submit_batch(
        self,
        model_id: OpenRouterModelId,
        bodies: tuple[JsonObject, ...],
    ) -> JsonObject:
        requests_payload = [
            {
                "custom_id": f"request-{index}",
                "body": {**body, "model": str(model_id)},
            }
            for index, body in enumerate(bodies)
        ]
        return self.transport.post_json(
            "/api/beta/batches",
            {
                "endpoint": "/v1/chat/completions",
                "model": str(model_id),
                "requests": requests_payload,
            },
        )

    def _wait_for_batches(
        self,
        jobs: tuple[_PendingBatch, ...],
    ) -> tuple[tuple[JsonObject, ...], ...]:
        results: list[tuple[JsonObject, ...] | None] = [None] * len(jobs)
        pending = list(enumerate(jobs))
        while pending:
            next_pending: list[tuple[int, _PendingBatch]] = []
            for index, job in pending:
                if job.batch.get("status") in _TERMINAL_BATCH_STATUSES:
                    results[index] = self._batch_results(job)
                else:
                    next_pending.append((index, job))
            if not next_pending:
                break
            for _, job in next_pending:
                if self.monotonic() >= job.deadline:
                    raise TimeoutError(
                        f"OpenRouter batch {job.batch_id} did not finish before timeout"
                    )
            self.sleep(self.poll_interval_seconds)
            for _, job in next_pending:
                job.batch = self.transport.get_json(f"/api/beta/batches/{job.batch_id}")
            pending = next_pending

        if any(result is None for result in results):  # pragma: no cover - internal invariant
            raise RuntimeError("submitted batch was not collected")
        return cast(tuple[tuple[JsonObject, ...], ...], tuple(results))

    @staticmethod
    def _batch_results(job: _PendingBatch) -> tuple[JsonObject, ...]:
        batch = job.batch
        if batch.get("status") != "completed":
            raise RuntimeError(
                f"OpenRouter batch {job.batch_id} ended with status {batch.get('status')}"
            )
        raw_results = batch.get("results")
        if not isinstance(raw_results, list):
            raise ValueError("completed OpenRouter batch has no results")

        results: dict[str, JsonObject] = {}
        for raw_result in raw_results:
            if not isinstance(raw_result, dict):
                raise ValueError("OpenRouter batch result is not an object")
            custom_id = raw_result.get("custom_id")
            response = raw_result.get("response")
            error = raw_result.get("error")
            if error is not None:
                raise RuntimeError(f"OpenRouter batch item {custom_id} failed: {error}")
            if not isinstance(custom_id, str) or not isinstance(response, dict):
                raise ValueError("OpenRouter batch result is malformed")
            if response.get("status_code") != 200 or not isinstance(response.get("body"), dict):
                raise RuntimeError(f"OpenRouter batch item {custom_id} returned {response}")
            results[custom_id] = cast(JsonObject, response["body"])

        expected_ids = [f"request-{index}" for index in range(job.body_count)]
        if set(results) != set(expected_ids):
            raise ValueError("OpenRouter batch results do not match submitted requests")
        return tuple(results[custom_id] for custom_id in expected_ids)


def response_content(response: JsonObject) -> str:
    """Extract non-empty assistant content from a chat-completion response."""
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("OpenRouter response has no assistant content") from error
    if not isinstance(content, str) or not content.strip():
        raise ValueError("OpenRouter response assistant content is empty")
    return content.strip()
