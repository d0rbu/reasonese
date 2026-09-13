"""Exact native rendering and target-span recovery for probe scoring."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from reasonese.probe_rendering import (
    FUNCTION_TOOLS,
    render_collector_probe_context,
    render_native_dialogue_context,
)
from reasonese.role_probe_extraction import NEMOTRON_ADAPTER
from tests.test_probe_qa import _requests


class CharacterTokenizer:
    def __init__(self, template: str) -> None:
        self.chat_template = template
        self.calls: list[tuple[bool, dict[str, Any]]] = []

    def apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs: Any,
    ) -> str:
        assert not tokenize
        self.calls.append((add_generation_prompt, kwargs))
        rendered = ""
        for message in conversation:
            role = message["role"]
            if role == "assistant" and "reasoning_content" in message:
                rendered += (
                    f"<|im_start|>assistant\n<think>{message['reasoning_content']}</think>"
                    f"{message['content']}<|im_end|>\n"
                )
            elif role == "assistant":
                rendered += "<|im_start|>assistant\n<tool_call/> <|im_end|>\n"
            else:
                rendered += f"<|im_start|>{role}\n{message['content']}<|im_end|>\n"
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n<think>"
        return rendered

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        result: dict[str, Any] = {"input_ids": [ord(character) for character in text]}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result

    def decode(self, token_ids: Sequence[int], **kwargs: Any) -> str:
        return "".join(chr(value) for value in token_ids)


def _tokenizer_and_adapter():
    template = "collector-native-test-template"
    tokenizer = CharacterTokenizer(template)
    adapter = replace(
        NEMOTRON_ADAPTER,
        chat_template_sha256=hashlib.sha256(template.encode()).hexdigest(),
    )
    return tokenizer, adapter


def test_collector_render_uses_generation_thinking_tools_and_exact_target_span() -> None:
    _, requests = _requests()
    request = requests[3]
    tokenizer, adapter = _tokenizer_and_adapter()
    rendered = render_collector_probe_context(tokenizer, adapter, request.setup, request.position)

    expected = str(request.setup.content_for_input(request.position - 1))
    assert "".join(chr(value) for value in rendered.content_token_ids[0]) == expected
    assert all(generation for generation, _ in tokenizer.calls)
    assert all(call["enable_thinking"] is True for _, call in tokenizer.calls)
    assert all(call["tools"] == list(FUNCTION_TOOLS) for _, call in tokenizer.calls)
    assert len(rendered.render_config_sha256) == 64


def test_native_dialogue_replays_full_context_and_returns_reasoning_then_final() -> None:
    tokenizer, adapter = _tokenizer_and_adapter()
    rendered = render_native_dialogue_context(
        tokenizer,
        adapter,
        prompt="What is two plus two?",
        reasoning="I should compute the sum carefully.",
        final="The answer is four.",
    )

    measured = tuple(
        "".join(chr(value) for value in token_ids) for token_ids in rendered.content_token_ids
    )
    assert measured == ("I should compute the sum carefully.", "The answer is four.")
    assert all(not generation for generation, _ in tokenizer.calls)
    assert all("tools" not in call for _, call in tokenizer.calls)
