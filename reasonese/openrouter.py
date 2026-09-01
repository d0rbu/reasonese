"""Small OpenRouter transport with synchronous and batch completion support."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, cast, runtime_checkable

import requests
from beartype import beartype
from phantom import Phantom

from reasonese.axes import Assistant, Author

JsonObject = dict[str, Any]
_TERMINAL_BATCH_STATUSES = frozenset({"completed", "failed", "cancelled", "expired"})


def _is_model_id(value: str) -> bool:
    return bool(value) and value == value.strip() and "/" in value


class OpenRouterModelId(str, Phantom[str], predicate=_is_model_id, bound=str):
    """A non-empty OpenRouter model slug."""


@beartype
@dataclass(frozen=True, slots=True)
class ModelRoute:
    """Synchronous model slug and its optional batch variant."""

    model_id: OpenRouterModelId
    batch_model_id: OpenRouterModelId | None


_MODEL_ROUTES: dict[Author, ModelRoute] = {
    Author.QWEN3_8_FLASH: ModelRoute(OpenRouterModelId.parse("qwen/qwen3.8-flash"), None),
    Author.QWEN3_8_2_4T: ModelRoute(
        OpenRouterModelId.parse("qwen/qwen3.8-2.4t-a95b"),
        OpenRouterModelId.parse("qwen/qwen3.8-2.4t-a95b:batch"),
    ),
    Author.INKLING: ModelRoute(
        OpenRouterModelId.parse("thinkingmachines/inkling"),
        OpenRouterModelId.parse("thinkingmachines/inkling:batch"),
    ),
    Author.INKLING_SMALL: ModelRoute(
        OpenRouterModelId.parse("thinkingmachines/inkling-small"),
        OpenRouterModelId.parse("thinkingmachines/inkling-small:batch"),
    ),
}


@beartype
def model_route(model: Author | Assistant) -> ModelRoute:
    """Return OpenRouter slugs for a model-backed author or assistant."""
    author = Author(model.value)
    if author is Author.USER:
        raise ValueError("the user author does not have an OpenRouter model")
    return _MODEL_ROUTES[author]


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
        response = self._session.post(
            f"{self._base_url}{path}",
            headers=self._headers(),
            json=body,
            timeout=self._timeout_seconds,
        )
        return self._json(response)

    @beartype
    def get_json(self, path: str) -> JsonObject:
        response = self._session.get(
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
    sleep: Callable[[float], None] = field(default=time.sleep, repr=False)
    monotonic: Callable[[], float] = field(default=time.monotonic, repr=False)

    def complete(self, model_id: OpenRouterModelId, body: JsonObject) -> JsonObject:
        """Run one synchronous chat completion."""
        return self.transport.post_json(
            "/api/v1/chat/completions",
            {**body, "model": str(model_id)},
        )

    def complete_many(
        self,
        route: ModelRoute,
        bodies: tuple[JsonObject, ...],
        *,
        prefer_batch: bool,
    ) -> tuple[JsonObject, ...]:
        """Complete requests in one model batch when possible, otherwise synchronously."""
        if not bodies:
            return ()
        if prefer_batch and route.batch_model_id is not None:
            return self._complete_batch(route.model_id, bodies)
        return tuple(self.complete(route.model_id, body) for body in bodies)

    def _complete_batch(
        self,
        model_id: OpenRouterModelId,
        bodies: tuple[JsonObject, ...],
    ) -> tuple[JsonObject, ...]:
        requests_payload = [
            {
                "custom_id": f"request-{index}",
                "body": {**body, "model": str(model_id)},
            }
            for index, body in enumerate(bodies)
        ]
        batch = self.transport.post_json(
            "/api/beta/batches",
            {
                "endpoint": "/v1/chat/completions",
                "model": str(model_id),
                "requests": requests_payload,
            },
        )
        batch_id = batch.get("id")
        if not isinstance(batch_id, str) or not batch_id:
            raise ValueError("OpenRouter batch response is missing an id")

        deadline = self.monotonic() + self.batch_timeout_seconds
        while batch.get("status") not in _TERMINAL_BATCH_STATUSES:
            if self.monotonic() >= deadline:
                raise TimeoutError(f"OpenRouter batch {batch_id} did not finish before timeout")
            self.sleep(self.poll_interval_seconds)
            batch = self.transport.get_json(f"/api/beta/batches/{batch_id}")

        if batch.get("status") != "completed":
            raise RuntimeError(
                f"OpenRouter batch {batch_id} ended with status {batch.get('status')}"
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

        expected_ids = [f"request-{index}" for index in range(len(bodies))]
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
