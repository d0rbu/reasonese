"""Offline coverage for assistant batch routing and tool parity."""

import json
from pathlib import Path
from threading import Event

import pytest
import requests

from reasonese.axes import Assistant
from reasonese.cache import YamlMessageCache, YamlTraceCache
from reasonese.conversation import ConversationTrace
from reasonese.judging import judge_requests
from reasonese.message_qa_cache import YamlMessageQaCache
from reasonese.openrouter import JsonObject, OpenRouterClient, RoutePreference, model_route
from reasonese.routing import CollectionRouting
from reasonese.runner import AssistantRunGroup, run_assistant_groups, run_matchup
from reasonese.study import build_trials
from reasonese.tools import ParallelSearch
from tests.test_cache_runner_cli import _chat, _qa_batch, _tool_chat
from tests.test_judging import _trace
from tests.test_openrouter import FakeTransport
from tests.test_study_orchestration import _manual_library, _study

MODEL = "deepseek/deepseek-v4.1-flash-1017"


@pytest.mark.parametrize("parallel_key", [None, "router-key", "sk-or-v1-old-router-key"])
def test_parallel_search_requires_its_own_key(
    monkeypatch: pytest.MonkeyPatch, parallel_key: str | None
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")
    if parallel_key is None:
        monkeypatch.delenv("PARALLEL_API_KEY")
    else:
        monkeypatch.setenv("PARALLEL_API_KEY", parallel_key)
    with pytest.raises(ValueError, match="API key issued by Parallel"):
        ParallelSearch()


def _batch(response: JsonObject) -> JsonObject:
    return {
        "id": "assistant-batch",
        "status": "completed",
        "results": [{"custom_id": "request-0", "response": {"status_code": 200, "body": response}}],
    }


def test_cold_matchup_checks_search_key_before_provider_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = _study()
    manual = _manual_library(tmp_path, study)
    transport = FakeTransport()
    monkeypatch.delenv("PARALLEL_API_KEY")
    with pytest.raises(ValueError, match="PARALLEL_API_KEY"):
        run_matchup(
            build_trials(study)[0].matchup,
            OpenRouterClient(transport),
            YamlMessageCache(tmp_path / "messages.yaml"),
            YamlTraceCache(tmp_path / "traces.yaml"),
            YamlMessageQaCache(tmp_path / "qa.yaml"),
            manual,
            prefer_batch=True,
            routing=CollectionRouting(allow_paid=True),
        )
    assert transport.calls == []


def test_assistant_batch_and_sync_deliver_identical_search_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = _study(assistant=Assistant.DEEPSEEK_V4_1_FLASH_1017)
    matchup = build_trials(study)[0].matchup
    manual = _manual_library(tmp_path, study)
    tool = _tool_chat(
        "web_search", '{"objective":"Look up the capital","search_queries":["capital of France"]}'
    )
    overlong = _tool_chat(
        "web_search", json.dumps({"objective": "Look up the capital", "search_queries": ["x" * 201]})
    )
    final = _chat("Paris. https://example.org/paris")
    searches = []

    def search(url: str, **kwargs: object) -> requests.Response:
        assert url == "https://api.parallel.ai/v1/search"
        searches.append(kwargs["json"])
        assert kwargs["headers"] == {"x-api-key": "test-search-key"}
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps({"results": [
            {"title": "Paris", "url": "https://example.org/paris", "excerpts": ["Paris is the capital."]},
        ] * 6}).encode()
        return response

    monkeypatch.setattr("reasonese.tools.requests.post", search)
    transports = []
    traces = []
    for use_batch in (True, False):
        cache = tmp_path / str(use_batch)
        transport = FakeTransport(
            posts=[
                _qa_batch(2),
                *(_batch(turn) if use_batch else turn for turn in (overlong, tool, tool, tool, final)),
            ],
            gets=[{"data": [{"id": MODEL}, {"id": MODEL + ":batch"}]}],
        )
        client = OpenRouterClient(transport)
        client._batch_models = None
        result = run_matchup(
            matchup, client, YamlMessageCache(cache / "messages.yaml"),
            YamlTraceCache(cache / "traces.yaml"), YamlMessageQaCache(cache / "qa.yaml"), manual,
            prefer_batch=True,
            routing=CollectionRouting(RoutePreference.AUTO if use_batch else RoutePreference.PAID, True),
        )
        assert result.trace.provenance is not None
        assert result.trace.provenance.transport == ("batch" if use_batch else "sync")
        assert result.trace.web_search == "parallel-fast"
        assert YamlTraceCache(cache / "traces.yaml").get(matchup) == result.trace
        assert transport.get_calls == (["/api/v1/models"] if use_batch else [])
        traces.append(result.trace)
        transports.append(transport)
    batched, synchronous = (transport.post_calls[1:] for transport in transports)
    assert [path for path, _ in batched] == ["/api/v1/batches"] * 5
    assert [payload["model"] for _, payload in batched] == [MODEL] * 5
    assert [payload["requests"][0]["body"] for _, payload in batched] == [
        payload for _, payload in synchronous
    ]
    assert searches == [{
        "objective": "Look up the capital", "search_queries": ["capital of France"],
        "mode": "fast", "advanced_settings": {"max_results": 5},
    }] * 4
    invalid, first, second, exhausted = traces[0].tool_steps
    assert "at most 200 characters" in invalid.results[0].content
    assert len(json.loads(first.results[0].content)) == len(json.loads(second.results[0].content)) == 5
    assert "search limit reached" in exhausted.results[0].content
    assert judge_requests(traces[0]) == judge_requests(traces[1])


def test_batch_splits_confirmed_oversize_submission_and_aligns_results() -> None:
    class SizeLimitedTransport(FakeTransport):
        def post_json(self, path: str, body: JsonObject) -> JsonObject:
            assert path == "/api/v1/batches"
            requests_payload = body["requests"]
            if len(requests_payload) > 1:
                rejection = requests.Response()
                rejection.status_code = 413
                raise requests.HTTPError(response=rejection)
            return _batch(_chat(requests_payload[0]["body"]["label"]))

    client = OpenRouterClient(SizeLimitedTransport(gets=[{"data": [{"id": MODEL + ":batch"}]}]))
    client._batch_models = None
    route = client.assistant_route(
        Assistant.DEEPSEEK_V4_1_FLASH_1017, RoutePreference.AUTO, prefer_batch=True,
    )

    assert client.complete_many(route, ({"label": "a"}, {"label": "b"}), prefer_batch=True) == (
        _chat("a"), _chat("b"),
    )


def test_assistant_catalog_is_lazy_and_exact() -> None:
    transport = FakeTransport(gets=[{"data": [
        {"id": MODEL + ":batch"},
        {"id": "thinkingmachines/inkling-small:batch"},
    ]}])
    client = OpenRouterClient(transport)
    client._batch_models = None
    deepseek = Assistant.DEEPSEEK_V4_1_FLASH_1017

    for preference, prefer_batch in (
        (RoutePreference.PAID, True), (RoutePreference.FREE, True), (RoutePreference.AUTO, False),
    ):
        route = client.assistant_route(deepseek, preference, prefer_batch=prefer_batch)
        assert str(route.model_id) == MODEL
        assert route.batch_model_id is None
    assert transport.get_calls == []

    route = client.assistant_route(deepseek, RoutePreference.AUTO, prefer_batch=True)
    assert route.batch_model_id == MODEL + ":batch"
    route = client.assistant_route(Assistant.INKLING, RoutePreference.AUTO, prefer_batch=True)
    assert route.batch_model_id is None
    assert route.model_id == "thinkingmachines/inkling:free"
    assert transport.get_calls == ["/api/v1/models"]


def test_sync_peer_finishes_during_batch_poll_and_valid_items_survive_failure() -> None:
    sync_finished = Event()
    completed = []

    class MixedTransport(FakeTransport):
        def post_json(self, path: str, body: JsonObject) -> JsonObject:
            if path == "/api/v1/batches":
                if len(body["requests"]) == 1:
                    messages = body["requests"][0]["body"]["messages"]
                    assert messages[-1]["role"] == "tool"
                    assert messages[-1]["tool_call_id"] == "live-call"
                    return _batch(_chat("batch answer"))
                assert len(body["requests"]) == 2
                return {"id": "mixed", "status": "in_progress"}
            assert path == "/api/v1/chat/completions"
            return _chat("sync answer")

        def get_json(self, path: str) -> JsonObject:
            assert path == "/api/v1/batches/mixed"
            assert sync_finished.wait(timeout=5)
            return {
                "status": "completed",
                "results": [
                    {"custom_id": "request-1", "error": {"message": "failed request"}},
                    {"custom_id": "request-0", "response": {
                        "status_code": 200, "body": _tool_chat(),
                    }},
                ],
            }

    def on_trace(group: int, position: int, trace: ConversationTrace) -> None:
        completed.append((group, position, trace.response))
        if group == 1:
            sync_finished.set()

    setup = _trace().setup
    client = OpenRouterClient(MixedTransport(), sleep=lambda _: None)
    with pytest.raises(ValueError, match="failed request"):
        run_assistant_groups(
            (
                AssistantRunGroup(model_route(Assistant.INKLING), (setup, setup)),
                AssistantRunGroup(model_route(Assistant.QWEN3_8_FLASH), (setup,)),
            ),
            client,
            on_trace=on_trace,
        )
    assert completed == [(1, 0, _chat("sync answer")), (0, 0, _chat("batch answer"))]
