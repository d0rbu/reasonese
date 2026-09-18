from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from reasonese.axes import Assistant, Author, Channel, Framing, Instruction
from reasonese.cache import YamlTraceCache, trace_from_dict, trace_to_dict
from reasonese.conversation import (
    ConversationSetup,
    ConversationTrace,
    GeneratedMessage,
    GeneratedText,
    ToolCallId,
    ToolResult,
    ToolStep,
    construct_conversation,
)
from reasonese.judging import fingerprint_traces, judge_request, trace_fingerprint
from reasonese.openrouter import JsonObject, ModelRoute, OpenRouterClient, OpenRouterModelId
from reasonese.planning import PromptSpec
from reasonese.runner import AssistantRunGroup, run_assistant_groups
from reasonese.study import Trial, build_trials, make_study
from reasonese.study_cache import SqliteStudyCache

TOOL_OUTPUTS = (
    "",
    "  \t  ",
    "  indented\n    lines\n",
    "line one\r\nline two\r\n",
    "naïve café — 東京 🧪\n",
)


def _spec(text: str, channel: Channel) -> PromptSpec:
    return PromptSpec(Instruction.parse(text), Framing.NORMAL, channel, Author.USER)


def _tool_response() -> JsonObject:
    return {
        "id": "tool-response",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "fixture-call",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path":"README.md"}',
                            },
                        }
                    ],
                }
            }
        ],
    }


def _final_response() -> JsonObject:
    return {
        "id": "final-response",
        "choices": [{"message": {"role": "assistant", "content": "final answer"}}],
    }


class _FakeTransport:
    def __init__(self, responses: list[JsonObject]) -> None:
        self.responses = responses
        self.posts: list[JsonObject] = []

    def post_json(self, path: str, body: JsonObject) -> JsonObject:
        assert path == "/api/v1/chat/completions"
        self.posts.append(body)
        return self.responses.pop(0)

    def get_json(self, path: str) -> JsonObject:
        raise AssertionError(f"unexpected GET {path}")


def _trial_and_setup() -> tuple[Trial, ConversationSetup]:
    study = make_study(
        (_spec("Seed README content.", Channel.README), _spec("User request.", Channel.USER)),
        Assistant.INKLING_SMALL,
        1,
    )
    trial = build_trials(study)[0]
    generated = tuple(
        GeneratedMessage(spec, GeneratedText.parse(str(spec.instruction)), None)
        for spec in trial.matchup.inputs
    )
    return trial, construct_conversation(trial.matchup, generated)


def _patch_runtime_with_output(monkeypatch: pytest.MonkeyPatch, output: str) -> None:
    from reasonese.tools import ToolRuntime

    class FixtureRuntime(ToolRuntime):
        def _write_readme(self) -> None:
            self.root.joinpath("README.md").write_bytes(output.encode("utf-8"))

    monkeypatch.setattr("reasonese.runner.ToolRuntime", FixtureRuntime)


def _run_with_output(
    monkeypatch: pytest.MonkeyPatch, output: str
) -> tuple[Trial, ConversationTrace, _FakeTransport]:
    trial, setup = _trial_and_setup()
    _patch_runtime_with_output(monkeypatch, output)
    transport = _FakeTransport([_tool_response(), _final_response()])
    route = ModelRoute(OpenRouterModelId.parse("example/model"), None)
    traces = run_assistant_groups(
        (AssistantRunGroup(route, (setup,)),), OpenRouterClient(transport)
    )
    return trial, traces[0][0], transport


@pytest.mark.parametrize("output", TOOL_OUTPUTS)
def test_tool_output_survives_runner_and_trace_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output: str
) -> None:
    trial, trace, transport = _run_with_output(monkeypatch, output)
    result = trace.tool_steps[0].results[0]

    assert type(result.content) is str
    assert result.content == output
    assert transport.posts[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "fixture-call",
        "content": output,
    }

    yaml_cache = YamlTraceCache(tmp_path / "traces.yaml")
    yaml_cache.put(trace)
    yaml_trace = yaml_cache.load()[0]

    sqlite_cache = SqliteStudyCache(tmp_path / "collection.sqlite3")
    sqlite_cache.put_traces(((trial.trial_id, trace),))
    sqlite_trace = sqlite_cache.load_traces((trial,))[trial.trial_id]

    expected_evidence_fragment = html.escape(
        json.dumps(result.openrouter_dict(), ensure_ascii=False, sort_keys=True)
    )
    original_request = judge_request(trace, 0)
    assert expected_evidence_fragment in original_request["messages"][1]["content"]
    original_fingerprint = trace_fingerprint(trace)
    original_batch_fingerprint = fingerprint_traces((trace,))[0].fingerprint
    for loaded in (yaml_trace, sqlite_trace):
        assert loaded.tool_steps[0].results[0].content == output
        assert trace_fingerprint(loaded) == original_fingerprint
        assert fingerprint_traces((loaded,))[0].fingerprint == original_batch_fingerprint
        assert judge_request(loaded, 0) == original_request


@pytest.mark.parametrize("invalid", [None, 123, [], {}])
def test_yaml_trace_cache_rejects_non_text_tool_results(tmp_path: Path, invalid: object) -> None:
    trial, setup = _trial_and_setup()
    trace = ConversationTrace(
        setup,
        _final_response(),
        (ToolStep(_tool_response(), (ToolResult(ToolCallId.parse("fixture-call"), "text"),)),),
    )
    raw = trace_to_dict(trace)
    tool_steps = cast(list[dict[str, Any]], raw["tool_steps"])
    results = cast(list[dict[str, Any]], tool_steps[0]["results"])
    results[0]["content"] = invalid
    path = tmp_path / "traces.yaml"
    path.write_text(yaml.safe_dump({"traces": [raw]}), encoding="utf-8")

    with pytest.raises(ValueError, match="content must be text"):
        YamlTraceCache(path).load()
    with pytest.raises(ValueError, match="content must be text"):
        trace_from_dict(raw, expected_matchup=trial.matchup)


@pytest.mark.parametrize("content", ["", " ", "\t", "\ntext", "text\n"])
def test_authored_generated_text_keeps_its_trimmed_nonempty_contract(content: str) -> None:
    with pytest.raises(TypeError):
        GeneratedText.parse(content)
