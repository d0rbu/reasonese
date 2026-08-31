from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from reasonese.axes import Assistant, Author, Channel, Framing, Instruction
from reasonese.cache import YamlMessageCache, YamlTraceCache
from reasonese.conversation import (
    ConversationTrace,
    GeneratedMessage,
    GeneratedText,
    construct_conversation,
)
from reasonese.matchup import Matchup, make_matchup, matchup_to_dict
from reasonese.openrouter import JsonObject, OpenRouterClient
from reasonese.planning import PromptSpec
from reasonese.run_conversation import main as run_conversation
from reasonese.runner import materialize_messages, run_matchup


def _spec(
    text: str,
    channel: Channel,
    author: Author = Author.USER,
) -> PromptSpec:
    return PromptSpec(Instruction.parse(text), Framing.NORMAL, channel, author)


def _matchup(*specs: PromptSpec) -> Matchup:
    return make_matchup(specs, Assistant.INKLING_SMALL)


def _chat(content: str, response_id: str = "response-1") -> JsonObject:
    return {
        "id": response_id,
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "reasoning": "preserved reasoning",
        "reasoning_details": [{"type": "reasoning.text", "text": "details"}],
    }


class FakeTransport:
    def __init__(self, posts: list[JsonObject]) -> None:
        self.posts = posts
        self.post_calls: list[tuple[str, JsonObject]] = []

    def post_json(self, path: str, body: JsonObject) -> JsonObject:
        self.post_calls.append((path, body))
        return self.posts.pop(0)

    def get_json(self, path: str) -> JsonObject:
        raise AssertionError(f"unexpected GET {path}")


def test_message_cache_round_trips_raw_responses_and_replaces_by_spec(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "messages.yaml"
    cache = YamlMessageCache(path)
    user_spec = _spec("Original user text.", Channel.SYSTEM)
    model_spec = _spec("Base task.", Channel.USER, Author.INKLING)
    original = GeneratedMessage(model_spec, GeneratedText.parse("first rendering"), _chat("first"))
    replacement = GeneratedMessage(
        model_spec,
        GeneratedText.parse("replacement rendering"),
        _chat("replacement", "response-2"),
    )
    user_message = GeneratedMessage(
        user_spec,
        GeneratedText.parse("Original user text."),
        None,
    )

    assert cache.load() == ()
    cache.put_many((original, user_message))
    cache.put_many((replacement,))

    assert cache.load() == (replacement, user_message)
    assert cache.get(model_spec) == replacement
    assert cache.get(_spec("Missing.", Channel.USER)) is None
    assert cache.load()[0].response == _chat("replacement", "response-2")


def test_trace_cache_round_trips_conversation_and_complete_response(tmp_path: Path) -> None:
    specs = (_spec("System task.", Channel.SYSTEM), _spec("User task.", Channel.USER))
    matchup = _matchup(*specs)
    generated = tuple(
        GeneratedMessage(spec, GeneratedText.parse(str(spec.instruction)), None) for spec in specs
    )
    first = ConversationTrace(construct_conversation(matchup, generated), _chat("first"))
    replacement = ConversationTrace(
        construct_conversation(matchup, generated), _chat("replacement", "response-2")
    )
    cache = YamlTraceCache(tmp_path / "traces.yaml")

    assert cache.get(matchup) is None
    cache.put(first)
    cache.put(replacement)

    assert cache.load() == (replacement,)
    assert cache.get(matchup) == replacement
    assert cache.load()[0].response["reasoning_details"] == [
        {"type": "reasoning.text", "text": "details"}
    ]


@pytest.mark.parametrize(
    ("key", "value", "error"),
    [
        ("messages", "bad", "one 'messages' list"),
        ("messages", ["bad"], "generated message must be a mapping"),
        ("messages", [{"input": {}, "content": "x"}], "invalid fields"),
        (
            "messages",
            [
                {
                    "input": {
                        "instruction": "Task.",
                        "framing": "normal",
                        "channel": "user message",
                        "author": "user",
                    },
                    "content": "Task.",
                    "response": [],
                }
            ],
            "mapping or null",
        ),
        ("traces", ["bad"], "trace must be a mapping"),
        ("traces", [{"matchup": {}, "conversation": []}], "invalid fields"),
    ],
)
def test_caches_reject_malformed_yaml(tmp_path: Path, key: str, value: object, error: str) -> None:
    path = tmp_path / "cache.yaml"
    path.write_text(yaml.safe_dump({key: value}))
    cache: YamlMessageCache | YamlTraceCache
    cache = YamlMessageCache(path) if key == "messages" else YamlTraceCache(path)
    with pytest.raises(ValueError, match=error):
        cache.load()


@pytest.mark.parametrize(
    ("conversation", "response", "error"),
    [
        ("bad", _chat("answer"), "conversation must be a list"),
        (["bad"], _chat("answer"), "conversation message has invalid fields"),
        ([{"role": "user", "content": "Task."}], None, "must not be null"),
    ],
)
def test_trace_cache_rejects_malformed_trace_fields(
    tmp_path: Path, conversation: object, response: object, error: str
) -> None:
    matchup = _matchup(_spec("System.", Channel.SYSTEM), _spec("User.", Channel.USER))
    path = tmp_path / "traces.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "traces": [
                    {
                        "matchup": matchup_to_dict(matchup),
                        "conversation": conversation,
                        "response": response,
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match=error):
        YamlTraceCache(path).load()


def test_empty_yaml_cache_is_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("null\n")
    assert YamlMessageCache(path).load() == ()


def test_materialize_messages_deduplicates_and_groups_missing_model_authors(
    tmp_path: Path,
) -> None:
    user = _spec("Pre-authored system text.", Channel.SYSTEM)
    inkling = _spec("First base task.", Channel.USER, Author.INKLING)
    inkling_small = _spec("Second base task.", Channel.USER, Author.INKLING_SMALL)
    matchup = _matchup(user, inkling, inkling, inkling_small)
    transport = FakeTransport([_chat("Inkling rewrite"), _chat("Inkling Small rewrite")])
    cache = YamlMessageCache(tmp_path / "messages.yaml")

    messages = materialize_messages(
        matchup,
        OpenRouterClient(transport),
        cache,
        prefer_batch=False,
    )

    assert [str(message.content) for message in messages] == [
        "Pre-authored system text.",
        "Inkling rewrite",
        "Inkling rewrite",
        "Inkling Small rewrite",
    ]
    assert len(transport.post_calls) == 2
    assert len(cache.load()) == 3
    cached_user = cache.get(user)
    assert cached_user is not None
    assert cached_user.response is None


def test_materialize_messages_uses_a_warm_cache_without_requests(tmp_path: Path) -> None:
    specs = (_spec("System.", Channel.SYSTEM), _spec("User.", Channel.USER))
    matchup = _matchup(*specs)
    cached = tuple(
        GeneratedMessage(spec, GeneratedText.parse(str(spec.instruction)), None) for spec in specs
    )
    cache = YamlMessageCache(tmp_path / "messages.yaml")
    cache.put_many(cached)

    assert (
        materialize_messages(
            matchup,
            OpenRouterClient(FakeTransport([])),
            cache,
            prefer_batch=True,
        )
        == cached
    )


def test_run_matchup_executes_once_preserves_reasoning_and_then_hits_cache(
    tmp_path: Path,
) -> None:
    matchup = _matchup(
        _spec("System.", Channel.SYSTEM),
        _spec("User.", Channel.USER),
    )
    transport = FakeTransport([_chat("assistant answer")])
    client = OpenRouterClient(transport)
    message_cache = YamlMessageCache(tmp_path / "messages.yaml")
    trace_cache = YamlTraceCache(tmp_path / "traces.yaml")

    first = run_matchup(
        matchup,
        client,
        message_cache,
        trace_cache,
        prefer_batch=True,
    )
    second = run_matchup(
        matchup,
        client,
        message_cache,
        trace_cache,
        prefer_batch=True,
    )

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.trace == second.trace
    assert first.trace.response["reasoning"] == "preserved reasoning"
    assert len(transport.post_calls) == 1
    assert transport.post_calls[0][1]["reasoning"] == {
        "enabled": True,
        "exclude": False,
    }


def _write_matchup(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            matchup_to_dict(
                _matchup(
                    _spec("System task.", Channel.SYSTEM),
                    _spec("User task.", Channel.USER),
                )
            ),
            sort_keys=False,
        )
    )


def test_run_conversation_cli_executes_and_warm_cache_needs_no_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    matchup_path = tmp_path / "matchup.yaml"
    message_cache = tmp_path / "messages.yaml"
    trace_cache = tmp_path / "traces.yaml"
    _write_matchup(matchup_path)
    transport = FakeTransport([_chat("assistant answer", "live-id")])
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("reasonese.run_conversation.RequestsTransport", lambda key: transport)
    args = [
        "--matchup",
        str(matchup_path),
        "--message-cache",
        str(message_cache),
        "--trace-cache",
        str(trace_cache),
        "--no-batch",
    ]

    assert run_conversation(args) == 0
    cold_summary = json.loads(capsys.readouterr().out)
    monkeypatch.delenv("OPENROUTER_API_KEY")
    assert run_conversation(args) == 0
    warm_summary = json.loads(capsys.readouterr().out)

    assert cold_summary["cache_hit"] is False
    assert cold_summary["response_id"] == "live-id"
    assert warm_summary["cache_hit"] is True
    assert warm_summary["messages"] == 2
    assert len(transport.post_calls) == 1


def test_run_conversation_cli_requires_key_for_uncached_matchup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matchup_path = tmp_path / "matchup.yaml"
    _write_matchup(matchup_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(SystemExit, match="2"):
        run_conversation(
            [
                "--matchup",
                str(matchup_path),
                "--message-cache",
                str(tmp_path / "messages.yaml"),
                "--trace-cache",
                str(tmp_path / "traces.yaml"),
            ]
        )
