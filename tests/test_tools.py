from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, cast

import pytest

from reasonese.conversation import (
    GeneratedText,
    ToolCall,
    ToolCallId,
    ToolName,
)
from reasonese.openrouter import JsonObject
from reasonese.tools import (
    ASSISTANT_TOOLS,
    ToolRuntime,
    assistant_message_from_response,
    tool_calls_from_response,
)


def _call(name: str, arguments: object, call_id: str = "call-1") -> ToolCall:
    encoded = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return ToolCall(ToolCallId.parse(call_id), ToolName.parse(name), encoded)


def _response(message: object) -> JsonObject:
    return {"choices": [{"message": message}]}


def test_tool_definitions_offer_local_tools_and_bounded_web_search() -> None:
    assert [tool["function"]["name"] for tool in ASSISTANT_TOOLS[:3]] == [
        "read_file",
        "bash",
        "python",
    ]
    assert ASSISTANT_TOOLS[3] == {
        "type": "openrouter:web_search",
        "parameters": {"engine": "auto", "max_results": 5, "max_total_results": 10},
    }


def test_runtime_reads_seeded_readme_and_rejects_paths_outside_workspace() -> None:
    first = GeneratedText.parse("First instruction.")
    second = GeneratedText.parse("Second instruction.")
    with ToolRuntime((first, second)) as runtime:
        result = runtime.execute(_call("read_file", {"path": "README.md"}))
        absolute = runtime.execute(_call("read_file", {"path": "/etc/passwd"}, "absolute"))
        traversal = runtime.execute(_call("read_file", {"path": "../outside"}, "traversal"))
        missing = runtime.execute(_call("read_file", {"path": "missing.txt"}, "missing"))

    assert result.content == "First instruction.\n\nSecond instruction."
    assert "path must be relative" in absolute.content
    assert "remain inside" in traversal.content
    assert "file does not exist" in missing.content


def test_runtime_reports_invalid_arguments_unknown_tools_and_lifecycle_errors() -> None:
    runtime = ToolRuntime(())
    with pytest.raises(RuntimeError, match="context manager"):
        _ = runtime.root

    with runtime:
        invalid_json = runtime.execute(_call("read_file", "{"))
        wrong_key = runtime.execute(_call("read_file", {"wrong": "README.md"}, "wrong"))
        blank = runtime.execute(_call("bash", {"command": " "}, "blank"))
        unknown = runtime.execute(_call("other", {}, "unknown"))

    assert "invalid JSON" in invalid_json.content
    assert "only 'path'" in wrong_key.content
    assert "non-empty string" in blank.content
    assert unknown.content == "error: unsupported tool other"


def test_runtime_rejects_non_utf8_and_oversized_files(monkeypatch: pytest.MonkeyPatch) -> None:
    with ToolRuntime(()) as runtime:
        (runtime.root / "binary").write_bytes(b"\xff")
        binary = runtime.execute(_call("read_file", {"path": "binary"}))
        (runtime.root / "large").write_bytes(b"x" * 5)
        monkeypatch.setattr("reasonese.tools._MAX_FILE_BYTES", 4)
        large = runtime.execute(_call("read_file", {"path": "large"}, "large"))

    assert "not valid UTF-8" in binary.content
    assert "exceeds 4 bytes" in large.content


def test_bash_and_python_execute_in_the_disposable_workspace() -> None:
    if shutil.which("bwrap") is None:
        pytest.skip("bubblewrap is not installed")
    with ToolRuntime(()) as runtime:
        bash = runtime.execute(
            _call("bash", {"command": "printf shell-ok; printf saved > artifact.txt"})
        )
        artifact = runtime.execute(_call("read_file", {"path": "artifact.txt"}, "artifact"))
        python = runtime.execute(_call("python", {"code": "print(6 * 7)"}, "python"))

    assert bash.content == "exit_code: 0\noutput:\nshell-ok"
    assert artifact.content == "saved"
    assert python.content == "exit_code: 0\noutput:\n42"


def test_sandbox_reports_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    if shutil.which("bwrap") is None:
        pytest.skip("bubblewrap is not installed")
    with ToolRuntime(()) as runtime:
        monkeypatch.setattr("reasonese.tools._TIMEOUT_SECONDS", 0.01)
        timed_out = runtime.execute(_call("bash", {"command": "sleep 1"}))

    assert "exceeded 0.01 seconds" in timed_out.content


def test_sandbox_reports_missing_bubblewrap(monkeypatch: pytest.MonkeyPatch) -> None:
    with ToolRuntime(()) as runtime:
        monkeypatch.setattr("reasonese.tools.shutil.which", lambda _: None)
        unavailable = runtime.execute(_call("python", {"code": "print(1)"}, "python"))

    assert "bubblewrap is required" in unavailable.content


def test_tool_call_parser_and_assistant_message_preserve_provider_fields() -> None:
    message: JsonObject = {
        "role": "assistant",
        "content": None,
        "reasoning": "raw reasoning",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "python", "arguments": '{"code":"print(1)"}'},
            }
        ],
    }
    response = _response(message)

    assert tool_calls_from_response(response) == (_call("python", '{"code":"print(1)"}'),)
    assert assistant_message_from_response(response) is message
    assert tool_calls_from_response(_response({"role": "assistant", "content": "done"})) == ()


@pytest.mark.parametrize(
    ("response", "error"),
    [
        ({}, "no first choice"),
        ({"choices": []}, "no first choice"),
        ({"choices": ["bad"]}, "no first choice"),
        ({"choices": [{}]}, "no assistant message"),
        (_response({"tool_calls": "bad"}), "must be a list"),
        (_response({"tool_calls": ["bad"]}), "malformed function call"),
        (_response({"tool_calls": [{"type": "other"}]}), "malformed function call"),
        (
            _response({"tool_calls": [{"id": "x", "type": "function"}]}),
            "no function object",
        ),
        (
            _response(
                {
                    "tool_calls": [
                        {
                            "id": "x",
                            "type": "function",
                            "function": {"name": "bash", "arguments": {}},
                        }
                    ]
                }
            ),
            "arguments must be JSON text",
        ),
    ],
)
def test_tool_call_parser_rejects_malformed_responses(response: JsonObject, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        tool_calls_from_response(response)


@pytest.mark.parametrize("response", [{}, {"choices": []}, {"choices": [{}]}])
def test_assistant_message_parser_rejects_malformed_responses(response: JsonObject) -> None:
    with pytest.raises(ValueError):
        assistant_message_from_response(response)


def test_runtime_turns_os_errors_into_tool_results(monkeypatch: pytest.MonkeyPatch) -> None:
    with ToolRuntime(()) as runtime:
        (runtime.root / "README.md").write_text("x")
        monkeypatch.setattr(
            Path, "read_bytes", cast(Any, lambda _: (_ for _ in ()).throw(OSError("x")))
        )
        result = runtime.execute(_call("read_file", {"path": "README.md"}))
    assert result.content.startswith("error:")
