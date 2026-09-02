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
    ToolCallId,
    ToolResult,
    ToolStep,
    construct_conversation,
)
from reasonese.manual_messages import ManualMessageLibrary
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


def _manual_library(tmp_path: Path, specs: tuple[PromptSpec, ...]) -> ManualMessageLibrary:
    root = tmp_path / "manual"
    for index, instruction in enumerate(dict.fromkeys(spec.instruction for spec in specs)):
        directory = root / f"instruction-{index}"
        directory.mkdir(parents=True)
        (directory / "instruction.txt").write_text(str(instruction))
        for framing in Framing:
            (directory / f"{framing}.txt").write_text(str(instruction))
    return ManualMessageLibrary(root)


def _chat(content: str, response_id: str = "response-1") -> JsonObject:
    return {
        "id": response_id,
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "reasoning": "preserved reasoning",
        "reasoning_details": [{"type": "reasoning.text", "text": "details"}],
    }


def _tool_chat(name: str = "read_file", arguments: str = '{"path":"README.md"}') -> JsonObject:
    return {
        "id": "tool-response",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning": "preserve this intermediate reasoning",
                    "tool_calls": [
                        {
                            "id": "live-call",
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                    ],
                }
            }
        ],
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
    specs = (_spec("File task.", Channel.README), _spec("User task.", Channel.USER))
    matchup = _matchup(*specs)
    generated = tuple(
        GeneratedMessage(spec, GeneratedText.parse(str(spec.instruction)), None) for spec in specs
    )
    first = ConversationTrace(construct_conversation(matchup, generated), _chat("first"))
    tool_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "live-call",
                            "type": "function",
                            "function": {"name": "python", "arguments": '{"code":"print(1)"}'},
                        }
                    ],
                }
            }
        ]
    }
    replacement = ConversationTrace(
        construct_conversation(matchup, generated),
        _chat("replacement", "response-2"),
        (
            ToolStep(
                tool_response,
                (
                    ToolResult(
                        ToolCallId.parse("live-call"),
                        GeneratedText.parse("exit_code: 0\noutput:\n1"),
                    ),
                ),
            ),
        ),
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
    assert cache.load()[0].tool_steps == replacement.tool_steps
    assert cache.load()[0].setup.content_for_input(0) == "File task."


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
    ("conversation", "tool_steps", "response", "error"),
    [
        ("bad", [], _chat("answer"), "conversation must be a list"),
        (["bad"], [], _chat("answer"), "conversation message has invalid fields"),
        (
            [{"role": "system", "content": "System."}, {"role": "user", "content": "User."}],
            "bad",
            _chat("answer"),
            "tool_steps must be a list",
        ),
        ([{"role": "user", "content": "Task."}], [], None, "must not be null"),
        ([{"role": "user", "content": "x", "extra": 1}], [], _chat("answer"), "text message"),
        (
            [{"role": "assistant", "content": None}],
            [],
            _chat("answer"),
            "assistant message",
        ),
        (
            [{"role": "assistant", "content": None, "tool_calls": "bad"}],
            [],
            _chat("answer"),
            "assistant tool_calls",
        ),
        (
            [{"role": "assistant", "content": None, "tool_calls": ["bad"]}],
            [],
            _chat("answer"),
            "tool call has invalid fields",
        ),
        (
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "x", "type": "other", "function": {}}],
                }
            ],
            [],
            _chat("answer"),
            "must be a function",
        ),
        (
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "x", "type": "function", "function": []}],
                }
            ],
            [],
            _chat("answer"),
            "function has invalid fields",
        ),
        (
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "x",
                            "type": "function",
                            "function": {"name": "bash", "arguments": {}},
                        }
                    ],
                }
            ],
            [],
            _chat("answer"),
            "arguments must be text",
        ),
        (
            [{"role": "tool", "content": "x"}],
            [],
            _chat("answer"),
            "tool message has invalid fields",
        ),
        (
            [{"role": "system", "content": "System."}, {"role": "user", "content": "User."}],
            ["bad"],
            _chat("answer"),
            "tool step has invalid fields",
        ),
        (
            [{"role": "system", "content": "System."}, {"role": "user", "content": "User."}],
            [{"response": None, "results": []}],
            _chat("answer"),
            "tool-step response must not be null",
        ),
        (
            [{"role": "system", "content": "System."}, {"role": "user", "content": "User."}],
            [{"response": _chat("tools"), "results": "bad"}],
            _chat("answer"),
            "tool-step results must be a list",
        ),
        (
            [{"role": "system", "content": "System."}, {"role": "user", "content": "User."}],
            [{"response": _chat("tools"), "results": [{"role": "user"}]}],
            _chat("answer"),
            "tool result has invalid fields",
        ),
    ],
)
def test_trace_cache_rejects_malformed_trace_fields(
    tmp_path: Path,
    conversation: object,
    tool_steps: object,
    response: object,
    error: str,
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
                        "tool_steps": tool_steps,
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
        _manual_library(tmp_path, matchup.inputs),
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
            _manual_library(tmp_path, matchup.inputs),
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
    manual = _manual_library(tmp_path, matchup.inputs)

    first = run_matchup(
        matchup,
        client,
        message_cache,
        trace_cache,
        manual,
        prefer_batch=True,
    )
    second = run_matchup(
        matchup,
        client,
        message_cache,
        trace_cache,
        manual,
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
    assert transport.post_calls[0][1]["temperature"] == 0.7
    assert transport.post_calls[0][1]["parallel_tool_calls"] is False
    assert {tool["type"] for tool in transport.post_calls[0][1]["tools"]} == {
        "function",
        "openrouter:web_search",
    }


def test_editing_a_manual_variant_invalidates_message_and_trace_caches(tmp_path: Path) -> None:
    specs = (_spec("System.", Channel.SYSTEM), _spec("User.", Channel.USER))
    matchup = _matchup(*specs)
    manual = _manual_library(tmp_path, matchup.inputs)
    transport = FakeTransport([_chat("first answer"), _chat("second answer")])
    message_cache = YamlMessageCache(tmp_path / "messages.yaml")
    trace_cache = YamlTraceCache(tmp_path / "traces.yaml")

    first = run_matchup(
        matchup,
        OpenRouterClient(transport),
        message_cache,
        trace_cache,
        manual,
        prefer_batch=False,
    )
    system_directory = next(
        directory
        for directory in manual.root.iterdir()
        if (directory / "instruction.txt").read_text() == "System."
    )
    (system_directory / "normal.txt").write_text("Changed manual system message.")
    second = run_matchup(
        matchup,
        OpenRouterClient(transport),
        message_cache,
        trace_cache,
        manual,
        prefer_batch=False,
    )

    assert first.cache_hit is False
    assert second.cache_hit is False
    assert second.trace.setup.content_for_input(0) == "Changed manual system message."
    assert second.trace.response == _chat("second answer")
    cached_system = message_cache.get(specs[0])
    assert cached_system is not None
    assert cached_system.content == "Changed manual system message."
    assert len(transport.post_calls) == 2


def test_run_matchup_executes_local_tool_calls_and_preserves_every_step(tmp_path: Path) -> None:
    matchup = _matchup(
        _spec("Repository instruction.", Channel.README),
        _spec("User request.", Channel.USER),
    )
    transport = FakeTransport([_tool_chat(), _chat("final answer")])

    result = run_matchup(
        matchup,
        OpenRouterClient(transport),
        YamlMessageCache(tmp_path / "messages.yaml"),
        YamlTraceCache(tmp_path / "traces.yaml"),
        _manual_library(tmp_path, matchup.inputs),
        prefer_batch=False,
    )

    assert result.trace.response == _chat("final answer")
    assert result.trace.tool_steps[0].response == _tool_chat()
    assert result.trace.tool_steps[0].results[0].content == "Repository instruction."
    second_messages = transport.post_calls[1][1]["messages"]
    assert second_messages[-2]["reasoning"] == "preserve this intermediate reasoning"
    assert second_messages[-1] == {
        "role": "tool",
        "tool_call_id": "live-call",
        "content": "Repository instruction.",
    }


def test_run_matchup_fails_when_assistant_exceeds_local_tool_step_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matchup = _matchup(_spec("System.", Channel.SYSTEM), _spec("User.", Channel.USER))
    monkeypatch.setattr("reasonese.runner._MAX_LOCAL_TOOL_STEPS", 0)
    with pytest.raises(RuntimeError, match="exceeded 0"):
        run_matchup(
            matchup,
            OpenRouterClient(FakeTransport([_tool_chat()])),
            YamlMessageCache(tmp_path / "messages.yaml"),
            YamlTraceCache(tmp_path / "traces.yaml"),
            _manual_library(tmp_path, matchup.inputs),
            prefer_batch=False,
        )


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
    manual = _manual_library(
        tmp_path,
        (_spec("System task.", Channel.SYSTEM), _spec("User task.", Channel.USER)),
    )
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
        "--user-messages",
        str(manual.root),
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
    manual = _manual_library(
        tmp_path,
        (_spec("System task.", Channel.SYSTEM), _spec("User task.", Channel.USER)),
    )
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
                "--user-messages",
                str(manual.root),
            ]
        )
