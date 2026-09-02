"""Bounded tools exposed to assistant models during matchup execution."""

from __future__ import annotations

import json
import os
import resource
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import cast

from beartype import beartype

from reasonese.conversation import (
    GeneratedText,
    ToolCall,
    ToolCallId,
    ToolName,
    ToolResult,
)
from reasonese.openrouter import JsonObject

_MAX_OUTPUT_BYTES = 64 * 1024
_MAX_FILE_BYTES = 256 * 1024
_TIMEOUT_SECONDS = 10.0


ASSISTANT_TOOLS: tuple[JsonObject, ...] = (
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file from the temporary task workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative file path, such as README.md.",
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run a bash command in a temporary, network-isolated Linux workspace. "
                "Execution time and output are limited."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command to run."}
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python",
            "description": (
                "Run Python code in an isolated interpreter in the temporary task workspace. "
                "Execution time and output are limited."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python source code to execute."}
                },
                "required": ["code"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "openrouter:web_search",
        "parameters": {"engine": "auto", "max_results": 5, "max_total_results": 10},
    },
)


def _resource_limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (_MAX_OUTPUT_BYTES, _MAX_OUTPUT_BYTES))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))


class ToolRuntime:
    """A disposable workspace and executor for local function tools."""

    @beartype
    def __init__(self, readme_contents: tuple[GeneratedText, ...]) -> None:
        self._readme_contents = readme_contents
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._root: Path | None = None

    def __enter__(self) -> ToolRuntime:
        self._temporary = tempfile.TemporaryDirectory(prefix="reasonese-tools-")
        self._root = Path(self._temporary.name)
        if self._readme_contents:
            (self._root / "README.md").write_text(
                "\n\n".join(str(content) for content in self._readme_contents),
                encoding="utf-8",
            )
        return self

    def __exit__(self, *_: object) -> None:
        assert self._temporary is not None
        self._temporary.cleanup()
        self._temporary = None
        self._root = None

    @property
    def root(self) -> Path:
        """Return the active temporary workspace."""
        if self._root is None:
            raise RuntimeError("tool runtime must be used as a context manager")
        return self._root

    def _arguments(self, call: ToolCall, expected_key: str) -> str:
        try:
            raw = json.loads(call.arguments)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON arguments: {error.msg}") from error
        if not isinstance(raw, dict) or set(raw) != {expected_key}:
            raise ValueError(f"arguments must contain only {expected_key!r}")
        value = raw[expected_key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{expected_key} must be a non-empty string")
        return value

    def _read_file(self, call: ToolCall) -> str:
        relative = Path(self._arguments(call, "path"))
        if relative.is_absolute():
            raise ValueError("path must be relative to the task workspace")
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("path must remain inside the task workspace")
        if not path.is_file():
            raise ValueError(f"file does not exist: {relative}")
        data = path.read_bytes()
        if len(data) > _MAX_FILE_BYTES:
            raise ValueError(f"file exceeds {_MAX_FILE_BYTES} bytes")
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("file is not valid UTF-8 text") from error

    def _sandbox_prefix(self) -> list[str]:
        executable = shutil.which("bwrap")
        if executable is None:
            raise RuntimeError("bubblewrap is required for bash and python tools")
        prefix = [
            executable,
            "--unshare-all",
            "--die-with-parent",
            "--new-session",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
        ]
        for source in ("/usr", "/bin", "/lib", "/lib64"):
            if Path(source).exists():
                prefix.extend(("--ro-bind", source, source))
        prefix.extend(
            (
                "--dir",
                "/workspace",
                "--bind",
                str(self.root),
                "/workspace",
                "--chdir",
                "/workspace",
                "--clearenv",
                "--setenv",
                "PATH",
                "/usr/local/bin:/usr/bin:/bin",
            )
        )
        return prefix

    def _run(self, command: list[str]) -> str:
        with tempfile.TemporaryFile() as output:
            process = subprocess.Popen(
                [*self._sandbox_prefix(), *command],
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                preexec_fn=_resource_limits,
                start_new_session=True,
            )
            try:
                return_code = process.wait(timeout=_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                return f"error: tool exceeded {_TIMEOUT_SECONDS:g} seconds"
            output.seek(0)
            captured = output.read(_MAX_OUTPUT_BYTES).decode("utf-8", errors="replace")
        body = captured.rstrip() or "(no output)"
        return f"exit_code: {return_code}\noutput:\n{body}"

    @beartype
    def execute(self, call: ToolCall) -> ToolResult:
        """Execute one supported function call and return errors as tool-visible text."""
        try:
            match str(call.name):
                case "read_file":
                    content = self._read_file(call)
                case "bash":
                    content = self._run(["/bin/bash", "-lc", self._arguments(call, "command")])
                case "python":
                    content = self._run(
                        ["/usr/bin/python3", "-I", "-c", self._arguments(call, "code")]
                    )
                case _:
                    content = f"error: unsupported tool {call.name!s}"
        except (OSError, RuntimeError, ValueError) as error:
            content = f"error: {error}"
        return ToolResult(call.call_id, GeneratedText.parse(content))


@beartype
def tool_calls_from_response(response: JsonObject) -> tuple[ToolCall, ...]:
    """Parse function calls from one OpenRouter chat-completion response."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("OpenRouter response has no first choice")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("OpenRouter response has no assistant message")
    raw_calls = message.get("tool_calls")
    if raw_calls is None:
        return ()
    if not isinstance(raw_calls, list):
        raise ValueError("assistant tool_calls must be a list")
    calls: list[ToolCall] = []
    for raw_call in raw_calls:
        if not isinstance(raw_call, dict) or raw_call.get("type") != "function":
            raise ValueError("assistant returned a malformed function call")
        function = raw_call.get("function")
        if not isinstance(function, dict):
            raise ValueError("assistant function call has no function object")
        call_id = raw_call.get("id")
        name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            raise ValueError("assistant function-call arguments must be JSON text")
        calls.append(
            ToolCall(
                ToolCallId.parse(call_id),
                ToolName.parse(name),
                arguments,
            )
        )
    return tuple(calls)


@beartype
def assistant_message_from_response(response: JsonObject) -> JsonObject:
    """Extract the raw assistant message for the next tool-loop request."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("OpenRouter response has no first choice")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("OpenRouter response has no assistant message")
    return cast(JsonObject, message)
