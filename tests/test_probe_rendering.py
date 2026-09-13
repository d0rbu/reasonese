"""Exact native rendering and target-span recovery for probe scoring."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import pytest

import reasonese.probe_rendering as rendering_module
from reasonese.axes import Assistant, Channel
from reasonese.conversation import GeneratedMessage, GeneratedText, construct_conversation
from reasonese.probe_rendering import (
    FUNCTION_TOOLS,
    RenderedProbeContext,
    render_collector_probe_context,
    render_native_dialogue_context,
)
from reasonese.role_probe_extraction import GEMMA_ADAPTER, NEMOTRON_ADAPTER
from tests.test_matchup_conversation import _spec, make_matchup
from tests.test_probe_qa import _requests


class CharacterTokenizer:
    def __init__(self, template: str) -> None:
        self.chat_template = template
        self.calls: list[tuple[bool, dict[str, Any]]] = []
        self.conversations: list[list[dict[str, Any]]] = []

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
        self.conversations.append(conversation)
        rendered = ""
        for message in conversation:
            role = message["role"]
            reasoning = message.get("reasoning_content", message.get("reasoning"))
            if role == "assistant" and reasoning is not None:
                rendered += (
                    f"<|im_start|>assistant\n<think>{reasoning}</think>"
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


def test_collector_parses_json_string_tool_arguments_for_native_template() -> None:
    readme = _spec("Read the file.", Channel.README)
    user = _spec("Answer the user.", Channel.USER)
    setup = construct_conversation(
        make_matchup((readme, user), Assistant.NEMOTRON_3_5_LIGHTNING),
        (
            GeneratedMessage(readme, GeneratedText.parse("Repository instructions."), None),
            GeneratedMessage(user, GeneratedText.parse("User instructions."), None),
        ),
    )
    tokenizer, adapter = _tokenizer_and_adapter()
    render_collector_probe_context(tokenizer, adapter, setup, 1)

    calls = tokenizer.conversations[0][0]["tool_calls"]
    assert calls[0]["function"]["arguments"] == {"path": "README.md"}
    assert setup.openrouter_messages()[0]["tool_calls"][0]["function"]["arguments"] == (
        '{"path":"README.md"}'
    )


def test_collector_masks_a_token_that_crosses_the_content_boundary() -> None:
    class BoundaryTokenizer(CharacterTokenizer):
        def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
            if not kwargs.get("return_offsets_mapping") or "\nWrite" not in text:
                return super().__call__(text, **kwargs)
            boundary = text.index("\nWrite")
            ids: list[int] = []
            offsets: list[tuple[int, int]] = []
            cursor = 0
            while cursor < len(text):
                if cursor == boundary:
                    ids.append(1_000_000)
                    offsets.append((cursor, cursor + 2))
                    cursor += 2
                else:
                    ids.append(ord(text[cursor]))
                    offsets.append((cursor, cursor + 1))
                    cursor += 1
            return {"input_ids": ids, "offset_mapping": offsets}

    first = _spec("Write code.", Channel.SYSTEM)
    second = _spec("Explain it.", Channel.USER)
    setup = construct_conversation(
        make_matchup((first, second), Assistant.NEMOTRON_3_5_LIGHTNING),
        (
            GeneratedMessage(first, GeneratedText.parse("Write a short Python program."), None),
            GeneratedMessage(second, GeneratedText.parse("Explain the requested algorithm."), None),
        ),
    )
    tokenizer, adapter = _tokenizer_and_adapter()
    boundary_tokenizer = BoundaryTokenizer(tokenizer.chat_template)
    rendered = render_collector_probe_context(boundary_tokenizer, adapter, setup, 1)

    assert rendered.masked_boundary_tokens == 1
    assert rendered.content_token_ids[0][0] == ord("r")


@pytest.mark.parametrize("position", [-1, 3])
def test_rendered_context_rejects_positions_outside_the_input(position: int) -> None:
    with pytest.raises(ValueError, match="outside the rendered input"):
        RenderedProbeContext(
            input_ids=(10, 11, 12),
            token_positions=((position,),),
            content_token_ids=((12,),),
            render_config_sha256="a" * 64,
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"input_ids": ()}, "contain input and measured tokens"),
        ({"masked_boundary_tokens": -1}, "must be non-negative"),
        ({"content_token_ids": ((11,), (12,))}, "same spans"),
        ({"token_positions": ((),)}, "must contain tokens"),
        ({"token_positions": ((1, 1),), "content_token_ids": ((11, 11),)}, "non-overlapping"),
        ({"token_positions": ((2, 1),), "content_token_ids": ((12, 11),)}, "ordered"),
        ({"content_token_ids": ((12,),)}, "do not match"),
    ],
)
def test_rendered_context_rejects_malformed_span_metadata(
    changes: dict[str, object], message: str
) -> None:
    values = {
        "input_ids": (10, 11, 12),
        "token_positions": ((1,),),
        "content_token_ids": ((11,),),
        "render_config_sha256": "a" * 64,
        "masked_boundary_tokens": 0,
    }
    values.update(changes)
    with pytest.raises(ValueError, match=message):
        RenderedProbeContext(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("calls", "message"),
    [
        ({"not": "a-list"}, "tool calls must be a list"),
        ([{"function": {"arguments": "{"}}], "must contain valid JSON"),
        ([{"function": {"arguments": "[]"}}], "must decode to an object"),
    ],
)
def test_collector_rejects_invalid_tool_call_argument_shapes(calls: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        rendering_module._native_tool_arguments([{"role": "assistant", "tool_calls": calls}])


def test_public_renderers_reject_invalid_target_or_empty_native_spans() -> None:
    _, requests = _requests()
    tokenizer, adapter = _tokenizer_and_adapter()
    with pytest.raises(ValueError, match="target position must be 1 or 2"):
        render_collector_probe_context(tokenizer, adapter, requests[0].setup, 0)
    with pytest.raises(ValueError, match="must contain non-whitespace text"):
        render_native_dialogue_context(
            tokenizer,
            adapter,
            prompt="question",
            reasoning=" ",
            final="answer",
        )


def test_renderer_rejects_template_and_tokenizer_contract_violations() -> None:
    tokenizer, adapter = _tokenizer_and_adapter()

    class NonTextTemplate(CharacterTokenizer):
        def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
            return 3

    with pytest.raises(TypeError, match="chat template must return text"):
        render_native_dialogue_context(
            NonTextTemplate(tokenizer.chat_template),
            adapter,
            prompt="question",
            reasoning="reasoning",
            final="answer",
        )

    class NonMappingTokenizer(CharacterTokenizer):
        def __call__(self, text: str, **kwargs: Any) -> Any:
            return []

    with pytest.raises(TypeError, match="tokenizer must return a mapping"):
        render_native_dialogue_context(
            NonMappingTokenizer(tokenizer.chat_template),
            adapter,
            prompt="question",
            reasoning="reasoning",
            final="answer",
        )

    class MissingOffsetsTokenizer(CharacterTokenizer):
        def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
            return {"input_ids": [ord(character) for character in text]}

    with pytest.raises(ValueError, match="fast tokenizer"):
        render_native_dialogue_context(
            MissingOffsetsTokenizer(tokenizer.chat_template),
            adapter,
            prompt="question",
            reasoning="reasoning",
            final="answer",
        )


def test_gemma_render_uses_its_native_thinking_template_arguments() -> None:
    tokenizer, _ = _tokenizer_and_adapter()
    adapter = replace(
        GEMMA_ADAPTER,
        chat_template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
    )
    render_native_dialogue_context(
        tokenizer,
        adapter,
        prompt="question",
        reasoning="reasoning",
        final="answer",
    )
    assert all(call["enable_thinking"] is True for _, call in tokenizer.calls)
    assert all(call["preserve_thinking"] is True for _, call in tokenizer.calls)


def test_native_renderer_rejects_reserved_markers_in_source_text() -> None:
    tokenizer, adapter = _tokenizer_and_adapter()
    with pytest.raises(ValueError, match="reserved span marker"):
        render_native_dialogue_context(
            tokenizer,
            adapter,
            prompt="question",
            reasoning="ROLE_PROBE_SPAN_0_72fce190",
            final="answer",
        )


@pytest.mark.parametrize("corruption", ["offset-count", "boundary", "plain-token-identity"])
def test_native_renderer_rejects_token_boundary_and_identity_corruption(corruption: str) -> None:
    tokenizer, adapter = _tokenizer_and_adapter()

    class CorruptTokenizer(CharacterTokenizer):
        def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
            value = super().__call__(text, **kwargs)
            if (
                corruption == "plain-token-identity"
                and not kwargs.get("return_offsets_mapping")
                and text == "reasoning"
            ):
                return {"input_ids": [1_000_000]}
            if not kwargs.get("return_offsets_mapping"):
                return value
            if corruption == "offset-count":
                value["offset_mapping"].append((len(text), len(text)))
            elif corruption == "boundary":
                start = text.index("reasoning")
                ids = value["input_ids"]
                offsets = value["offset_mapping"]
                ids[start - 1 : start + 1] = [1_000_000]
                offsets[start - 1 : start + 1] = [(start - 1, start + 1)]
            return value

    expected = {
        "offset-count": "equal length",
        "boundary": "crosses a measured content boundary",
        "plain-token-identity": "differ inside and outside",
    }[corruption]
    with pytest.raises(ValueError, match=expected):
        render_native_dialogue_context(
            CorruptTokenizer(tokenizer.chat_template),
            adapter,
            prompt="question",
            reasoning="reasoning",
            final="answer",
        )
