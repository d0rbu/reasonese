"""Offline assistant fixtures."""

import pytest

from reasonese.openrouter import OpenRouterClient


@pytest.fixture(autouse=True)
def offline_assistant_services(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PARALLEL_API_KEY", "test-search-key")
    # Existing fixtures model synchronous assistant endpoints. Individual batch
    # integration tests can clear this snapshot to exercise catalog discovery.
    original = OpenRouterClient.__post_init__

    def post_init(client: OpenRouterClient) -> None:
        original(client)
        client._batch_models = frozenset()

    monkeypatch.setattr(OpenRouterClient, "__post_init__", post_init)
