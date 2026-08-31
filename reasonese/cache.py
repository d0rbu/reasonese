"""Readable YAML caches for generated messages and conversation traces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml
from beartype import beartype

from reasonese.conversation import (
    ChatMessage,
    ChatRole,
    ConversationSetup,
    ConversationTrace,
    GeneratedMessage,
    GeneratedText,
)
from reasonese.matchup import (
    Matchup,
    matchup_from_dict,
    matchup_to_dict,
    prompt_spec_from_dict,
    prompt_spec_to_dict,
)
from reasonese.openrouter import JsonObject
from reasonese.planning import PromptSpec


def _load_list(path: Path, key: str) -> list[object]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if raw is None:
        return []
    if not isinstance(raw, dict) or set(raw) != {key} or not isinstance(raw[key], list):
        raise ValueError(f"{path} must contain one {key!r} list")
    return cast(list[object], raw[key])


def _write_list(path: Path, key: str, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump({key: values}, handle, sort_keys=False, allow_unicode=True, width=100)


def _response(raw: object) -> JsonObject | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("cached OpenRouter response must be a mapping or null")
    return cast(JsonObject, raw)


def _message_from_dict(raw: object) -> GeneratedMessage:
    if not isinstance(raw, dict):
        raise ValueError("cached generated message must be a mapping")
    data = cast(dict[str, Any], raw)
    if set(data) != {"input", "content", "response"}:
        raise ValueError("cached generated message has invalid fields")
    return GeneratedMessage(
        prompt_spec_from_dict(data["input"]),
        GeneratedText.parse(data["content"]),
        _response(data["response"]),
    )


@beartype
def _message_to_dict(message: GeneratedMessage) -> dict[str, object]:
    return {
        "input": prompt_spec_to_dict(message.spec),
        "content": str(message.content),
        "response": message.response,
    }


@beartype
@dataclass(frozen=True, slots=True)
class YamlMessageCache:
    """Generated messages keyed directly by their four entry coordinates."""

    path: Path

    def load(self) -> tuple[GeneratedMessage, ...]:
        return tuple(_message_from_dict(raw) for raw in _load_list(self.path, "messages"))

    def get(self, spec: PromptSpec) -> GeneratedMessage | None:
        return next((message for message in self.load() if message.spec == spec), None)

    def put_many(self, messages: tuple[GeneratedMessage, ...]) -> None:
        by_spec = {message.spec: message for message in self.load()}
        by_spec.update({message.spec: message for message in messages})
        _write_list(
            self.path,
            "messages",
            [_message_to_dict(message) for message in by_spec.values()],
        )


def _trace_from_dict(raw: object) -> ConversationTrace:
    if not isinstance(raw, dict):
        raise ValueError("cached trace must be a mapping")
    data = cast(dict[str, Any], raw)
    if set(data) != {"matchup", "conversation", "response"}:
        raise ValueError("cached trace has invalid fields")
    matchup = matchup_from_dict(data["matchup"])
    raw_messages = data["conversation"]
    if not isinstance(raw_messages, list):
        raise ValueError("cached conversation must be a list")
    messages: list[ChatMessage] = []
    for raw_message in raw_messages:
        if not isinstance(raw_message, dict) or set(raw_message) != {"role", "content"}:
            raise ValueError("cached conversation message has invalid fields")
        messages.append(
            ChatMessage(
                ChatRole(raw_message["role"]),
                GeneratedText.parse(raw_message["content"]),
            )
        )
    response = _response(data["response"])
    if response is None:
        raise ValueError("cached trace response must not be null")
    return ConversationTrace(ConversationSetup(matchup, tuple(messages)), response)


@beartype
def _trace_to_dict(trace: ConversationTrace) -> dict[str, object]:
    return {
        "matchup": matchup_to_dict(trace.setup.matchup),
        "conversation": [
            {"role": str(message.role), "content": str(message.content)}
            for message in trace.setup.messages
        ],
        "response": trace.response,
    }


@beartype
@dataclass(frozen=True, slots=True)
class YamlTraceCache:
    """Assistant responses keyed by the complete matchup."""

    path: Path

    def load(self) -> tuple[ConversationTrace, ...]:
        return tuple(_trace_from_dict(raw) for raw in _load_list(self.path, "traces"))

    def get(self, matchup: Matchup) -> ConversationTrace | None:
        return next((trace for trace in self.load() if trace.setup.matchup == matchup), None)

    def put(self, trace: ConversationTrace) -> None:
        by_matchup = {cached.setup.matchup: cached for cached in self.load()}
        by_matchup[trace.setup.matchup] = trace
        _write_list(
            self.path,
            "traces",
            [_trace_to_dict(cached) for cached in by_matchup.values()],
        )
