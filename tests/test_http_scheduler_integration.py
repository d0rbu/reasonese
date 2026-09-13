"""Exercise actual requests sessions against a local HTTP server, never a provider."""

from __future__ import annotations

import json
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Barrier, Lock, Thread
from time import monotonic
from typing import Any

from reasonese.openrouter import (
    CompletionGroup,
    JsonObject,
    ModelRoute,
    OpenRouterClient,
    OpenRouterModelId,
    RequestsTransport,
)


def test_http_429_isolates_models_preserves_payloads_and_allows_parallel_connections() -> None:
    first_healthy_pair = Barrier(2, timeout=5)
    lock = Lock()
    seen: list[tuple[str, JsonObject, float]] = []
    counts: Counter[str] = Counter()
    active: Counter[str] = Counter()
    peaks: Counter[str] = Counter()
    handler_errors: list[BaseException] = []
    limited = "example/limited:free"
    healthy = "example/healthy:free"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_POST(self) -> None:  # noqa: N802
            try:
                assert self.path == "/api/v1/chat/completions"
                assert self.headers["Authorization"] == "Bearer offline-test-key"
                assert self.headers["Content-Type"] == "application/json"
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                model = body["model"]
                with lock:
                    counts[model] += 1
                    attempt = counts[model]
                    active[model] += 1
                    peaks[model] = max(peaks[model], active[model])
                    seen.append((model, body, monotonic()))
                if model == healthy and attempt <= 2:
                    # Neither request can finish unless real HTTP connections overlap.
                    first_healthy_pair.wait()
                status = 429 if model == limited and attempt == 1 else 200
                payload = json.dumps({"id": f"{model}-{body['index']}", "choices": [{"message": {"content": "answer"}}]}).encode()
                self.send_response(status)
                if status == 429:
                    self.send_header("Retry-After", "1")
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                with lock:
                    active[model] -= 1
            except BaseException as error:
                handler_errors.append(error)
                raise

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = OpenRouterClient(
            RequestsTransport(
                "offline-test-key",
                base_url=f"http://127.0.0.1:{server.server_port}",
                timeout_seconds=5.0,
            ),
            sync_workers=2,
        )
        groups = tuple(
            CompletionGroup(
                ModelRoute(OpenRouterModelId.parse(model), None),
                tuple(
                    {"index": index, "messages": [{"role": "user", "content": "test"}]}
                    for index in range(size)
                ),
            )
            for model, size in ((limited, 1), (healthy, 6))
        )
        result = client.complete_many_grouped(groups, prefer_batch=False)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert not handler_errors
    assert not thread.is_alive()
    assert result == (({"id": f"{limited}-0", "choices": [{"message": {"content": "answer"}}]},), tuple({"id": f"{healthy}-{i}", "choices": [{"message": {"content": "answer"}}]} for i in range(6)))
    assert counts == {limited: 2, healthy: 6}
    assert peaks[healthy] == 2
    limited_calls = [(body, timestamp) for model, body, timestamp in seen if model == limited]
    assert limited_calls[0][0] == limited_calls[1][0]
    assert limited_calls[1][1] - limited_calls[0][1] >= 1.0
    assert max(timestamp for model, _, timestamp in seen if model == healthy) < limited_calls[1][1]
