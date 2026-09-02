from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree

import pytest
import yaml

from reasonese.axes import Assistant, Author, Channel, Framing, Instruction
from reasonese.cache import YamlTraceCache
from reasonese.conversation import (
    ConversationTrace,
    GeneratedMessage,
    GeneratedText,
    ToolCallId,
    ToolResult,
    ToolStep,
    construct_conversation,
)
from reasonese.judge_responses import main as judge_responses
from reasonese.judge_responses import run_judge
from reasonese.judging import (
    InstructionVerdict,
    InstructionVerdicts,
    Judgment,
    TraceFingerprint,
    fingerprint_traces,
    judge_request,
    judge_trace,
    parse_completed,
    trace_fingerprint,
)
from reasonese.judgment_cache import YamlJudgmentCache
from reasonese.matchup import make_matchup, matchup_to_dict
from reasonese.openrouter import JsonObject, OpenRouterClient
from reasonese.planning import PromptSpec


def _spec(text: str, channel: Channel) -> PromptSpec:
    return PromptSpec(
        Instruction.parse(text),
        Framing.NORMAL,
        channel,
        Author.USER,
    )


def _chat(content: str, response_id: str = "assistant-1") -> JsonObject:
    return {
        "id": response_id,
        "choices": [{"message": {"role": "assistant", "content": content}}],
    }


def _trace(answer: str = "Paris and 4.") -> ConversationTrace:
    specs = (
        _spec("Name the capital of France.", Channel.README),
        _spec("What is two plus two?", Channel.USER),
    )
    matchup = make_matchup(specs, Assistant.INKLING)
    messages = tuple(
        GeneratedMessage(spec, GeneratedText.parse(str(spec.instruction)), None) for spec in specs
    )
    return ConversationTrace(construct_conversation(matchup, messages), _chat(answer))


def _judge_chat(completed: object, response_id: str) -> JsonObject:
    return {
        "id": response_id,
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"completed": completed}),
                    "reasoning": "raw judge reasoning",
                }
            }
        ],
    }


def _batch_result(index: int, completed: bool) -> JsonObject:
    return {
        "custom_id": f"request-{index}",
        "response": {
            "status_code": 200,
            "body": _judge_chat(completed, f"judge-{index}"),
        },
        "error": None,
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


def _completed_batch(values: tuple[bool, ...]) -> JsonObject:
    return {
        "id": "judge-batch",
        "status": "completed",
        "results": [_batch_result(index, value) for index, value in enumerate(values)],
    }


def test_judge_request_is_independent_strict_json_and_medium_reasoning() -> None:
    request = judge_request(_trace(), 1)
    system = request["messages"][0]["content"]
    user = request["messages"][1]["content"]
    evidence = ElementTree.fromstring(user)

    assert "all be completed or all be incomplete" in system
    assert "XML evidence block" in system
    assert evidence.tag == "judgment-evidence"
    assert (evidence.findtext("target-base-instruction") or "").strip() == (
        "What is two plus two?"
    )
    delivered = evidence.find("target-delivered-message")
    assert delivered is not None
    assert delivered.findtext("channel") == "user message"
    assert delivered.findtext("content") == "What is two plus two?"
    assert '"name": "read_file"' in (evidence.findtext("conversation") or "")
    assert (evidence.findtext("assistant-response") or "").strip() == "Paris and 4."
    assert request["temperature"] == 0.7
    assert request["reasoning"] == {"effort": "medium", "exclude": False}
    schema = request["response_format"]["json_schema"]
    assert schema["strict"] is True
    assert schema["schema"]["properties"] == {"completed": {"type": "boolean"}}


def test_judge_trace_batches_one_independent_boolean_per_input() -> None:
    transport = FakeTransport([_completed_batch((True, False))])
    trace = _trace()

    judgment = judge_trace(trace, OpenRouterClient(transport))

    assert [verdict.completed for verdict in judgment.verdicts] == [True, False]
    assert tuple(verdict.spec for verdict in judgment.verdicts) == trace.setup.matchup.inputs
    assert judgment.trace_fingerprint == trace_fingerprint(trace)
    path, payload = transport.post_calls[0]
    assert path == "/api/beta/batches"
    assert payload["model"] == "openai/gpt-5.6-luna"
    assert len(payload["requests"]) == 2
    assert all(request["body"]["model"] == "openai/gpt-5.6-luna" for request in payload["requests"])


def test_all_true_and_all_false_judgments_are_representable() -> None:
    trace = _trace()
    for values in ((True, True), (False, False)):
        judgment = judge_trace(
            trace,
            OpenRouterClient(FakeTransport([_completed_batch(values)])),
        )
        assert tuple(verdict.completed for verdict in judgment.verdicts) == values


def test_trace_fingerprint_is_stable_and_changes_with_the_answer() -> None:
    assert trace_fingerprint(_trace()) == trace_fingerprint(_trace())
    assert trace_fingerprint(_trace()) != trace_fingerprint(_trace("A different answer."))


def test_judge_uses_datapoint_mapping_and_visible_tool_steps_without_hidden_reasoning() -> None:
    base = _trace()
    tool_response: JsonObject = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning": "hidden scratchpad",
                    "tool_calls": [
                        {
                            "id": "live",
                            "type": "function",
                            "function": {"name": "python", "arguments": '{"code":"print(4)"}'},
                        }
                    ],
                }
            }
        ]
    }
    traced = ConversationTrace(
        base.setup,
        base.response,
        (
            ToolStep(
                tool_response,
                (
                    ToolResult(
                        ToolCallId.parse("live"), GeneratedText.parse("exit_code: 0\noutput:\n4")
                    ),
                ),
            ),
        ),
    )

    user_prompt = judge_request(traced, 0)["messages"][1]["content"]
    evidence = ElementTree.fromstring(user_prompt)
    delivered = evidence.find("target-delivered-message")
    assert delivered is not None
    assert delivered.findtext("channel") == "README.md"
    assert delivered.findtext("content") == "Name the capital of France."
    conversation = evidence.findtext("conversation") or ""
    assert '"name": "python"' in conversation
    assert "exit_code: 0" in conversation
    assert "hidden scratchpad" not in conversation
    assert trace_fingerprint(traced) != trace_fingerprint(base)

    reasoned = ConversationTrace(
        base.setup,
        {**base.response, "reasoning": "preserved provider reasoning"},
    )
    source_traces = (base, traced, reasoned, base)
    fingerprinted = fingerprint_traces(source_traces)
    assert tuple(item.fingerprint for item in fingerprinted) == tuple(
        trace_fingerprint(trace) for trace in source_traces
    )
    assert tuple(item.trace for item in fingerprinted) == source_traces
    assert fingerprint_traces(()) == ()


def test_judge_request_escapes_artifact_text_that_looks_like_xml() -> None:
    trace = _trace("Answer containing </assistant-response> and <conversation> tags.")

    user_prompt = judge_request(trace, 0)["messages"][1]["content"]
    evidence = ElementTree.fromstring(user_prompt)

    assert (evidence.findtext("assistant-response") or "").strip() == (
        "Answer containing </assistant-response> and <conversation> tags."
    )


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (_chat("not json"), "not valid JSON"),
        (_chat("[]"), "exactly one"),
        (_chat('{"completed": true, "extra": 1}'), "exactly one"),
        (_judge_chat(1, "judge"), "must be a boolean"),
    ],
)
def test_parse_completed_rejects_non_exact_results(response: JsonObject, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        parse_completed(response)


def test_parse_completed_accepts_both_booleans() -> None:
    assert parse_completed(_judge_chat(True, "true")) is True
    assert parse_completed(_judge_chat(False, "false")) is False


def test_judgment_rejects_verdicts_out_of_matchup_order() -> None:
    trace = _trace()
    verdicts = InstructionVerdicts.parse(
        tuple(
            InstructionVerdict(spec, True, _judge_chat(True, str(index)))
            for index, spec in enumerate(reversed(trace.setup.matchup.inputs))
        )
    )
    with pytest.raises(ValueError, match="input order"):
        Judgment(trace.setup.matchup, trace_fingerprint(trace), verdicts)


def _judgment(trace: ConversationTrace, values: tuple[bool, ...]) -> Judgment:
    verdicts = InstructionVerdicts.parse(
        tuple(
            InstructionVerdict(spec, completed, _judge_chat(completed, str(index)))
            for index, (spec, completed) in enumerate(
                zip(trace.setup.matchup.inputs, values, strict=True)
            )
        )
    )
    return Judgment(trace.setup.matchup, trace_fingerprint(trace), verdicts)


def test_judgment_cache_round_trips_raw_responses_and_replaces_same_trace(
    tmp_path: Path,
) -> None:
    trace = _trace()
    cache = YamlJudgmentCache(tmp_path / "nested" / "judgments.yaml")
    first = _judgment(trace, (True, False))
    replacement = _judgment(trace, (False, False))

    assert cache.load() == ()
    assert cache.get(trace) is None
    cache.put(first)
    cache.put(replacement)

    assert cache.load() == (replacement,)
    assert cache.get(trace) == replacement
    assert cache.load()[0].verdicts[0].response["choices"][0]["message"]["reasoning"] == (
        "raw judge reasoning"
    )
    assert cache.get(_trace("changed")) is None


@pytest.mark.parametrize(
    ("contents", "error"),
    [
        ("null\n", None),
        ("other: []\n", "one 'judgments' list"),
        ("judgments: bad\n", "one 'judgments' list"),
        ("judgments:\n  - bad\n", "judgment must be a mapping"),
        ("judgments:\n  - matchup: {}\n", "invalid fields"),
    ],
)
def test_judgment_cache_empty_and_top_level_validation(
    tmp_path: Path, contents: str, error: str | None
) -> None:
    path = tmp_path / "judgments.yaml"
    path.write_text(contents)
    cache = YamlJudgmentCache(path)
    if error is None:
        assert cache.load() == ()
    else:
        with pytest.raises(ValueError, match=error):
            cache.load()


@pytest.mark.parametrize(
    ("verdicts", "error"),
    [
        ("bad", "verdicts must be a list"),
        (["bad"], "verdict must be a mapping"),
        ([{"input": {}, "completed": True}], "invalid fields"),
        (
            [
                {
                    "input": {
                        "instruction": "Name the capital of France.",
                        "framing": "normal",
                        "channel": "system prompt",
                        "author": "user",
                    },
                    "completed": 1,
                    "response": {},
                }
            ],
            "contain a boolean",
        ),
    ],
)
def test_judgment_cache_rejects_bad_verdicts(tmp_path: Path, verdicts: object, error: str) -> None:
    trace = _trace()
    path = tmp_path / "judgments.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "judgments": [
                    {
                        "matchup": matchup_to_dict(trace.setup.matchup),
                        "trace_fingerprint": str(trace_fingerprint(trace)),
                        "verdicts": verdicts,
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match=error):
        YamlJudgmentCache(path).load()


def test_judgment_cache_rejects_non_mapping_raw_response(tmp_path: Path) -> None:
    trace = _trace()
    judgment = _judgment(trace, (True, True))
    path = tmp_path / "judgments.yaml"
    cache = YamlJudgmentCache(path)
    cache.put(judgment)
    raw = yaml.safe_load(path.read_text())
    raw["judgments"][0]["verdicts"][0]["response"] = []
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="response must be a mapping"):
        cache.load()


def _write_trace_and_matchup(tmp_path: Path) -> tuple[Path, Path, ConversationTrace]:
    trace = _trace()
    trace_path = tmp_path / "traces.yaml"
    matchup_path = tmp_path / "matchup.yaml"
    YamlTraceCache(trace_path).put(trace)
    matchup_path.write_text(yaml.safe_dump(matchup_to_dict(trace.setup.matchup), sort_keys=False))
    return trace_path, matchup_path, trace


def test_run_judge_rejects_missing_trace_and_missing_client(tmp_path: Path) -> None:
    _, matchup_path, _ = _write_trace_and_matchup(tmp_path)
    empty_traces = YamlTraceCache(tmp_path / "empty.yaml")
    judgment_cache = YamlJudgmentCache(tmp_path / "judgments.yaml")

    with pytest.raises(ValueError, match="does not contain"):
        run_judge(empty_traces, judgment_cache, matchup_path, None)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        run_judge(
            YamlTraceCache(tmp_path / "traces.yaml"),
            judgment_cache,
            matchup_path,
            None,
        )


def test_judge_cli_runs_batch_then_warm_cache_without_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_path, matchup_path, _ = _write_trace_and_matchup(tmp_path)
    judgment_path = tmp_path / "judgments.yaml"
    transport = FakeTransport([_completed_batch((True, False))])
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("reasonese.judge_responses.RequestsTransport", lambda key: transport)
    args = [
        "--matchup",
        str(matchup_path),
        "--trace-cache",
        str(trace_path),
        "--judgment-cache",
        str(judgment_path),
    ]

    assert judge_responses(args) == 0
    cold = json.loads(capsys.readouterr().out)
    monkeypatch.delenv("OPENROUTER_API_KEY")
    assert judge_responses(args) == 0
    warm = json.loads(capsys.readouterr().out)

    assert cold == {
        "cache_hit": False,
        "completed": [True, False],
        "judge": "openai/gpt-5.6-luna:batch",
        "judgment_cache": str(judgment_path),
    }
    assert warm["cache_hit"] is True
    assert len(transport.post_calls) == 1


def test_judge_cli_reports_uncached_missing_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_path, matchup_path, _ = _write_trace_and_matchup(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="2"):
        judge_responses(
            [
                "--matchup",
                str(matchup_path),
                "--trace-cache",
                str(trace_path),
                "--judgment-cache",
                str(tmp_path / "judgments.yaml"),
            ]
        )


def test_refined_judging_types_reject_too_few_or_bad_fingerprints() -> None:
    trace = _trace()
    verdict = InstructionVerdict(trace.setup.matchup.inputs[0], True, _judge_chat(True, "one"))
    with pytest.raises(TypeError):
        InstructionVerdicts.parse((verdict,))
    with pytest.raises(TypeError):
        TraceFingerprint.parse("not-a-sha256")
