"""Exact native-template rendering and measured token spans for role probes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from beartype import beartype

from reasonese.axes import Channel
from reasonese.conversation import ConversationSetup
from reasonese.role_probe_extraction import (
    GEMMA_ADAPTER,
    NEMOTRON_ADAPTER,
    NativeTemplateAdapter,
    validate_native_template,
)
from reasonese.tools import ASSISTANT_TOOLS

FUNCTION_TOOLS = ASSISTANT_TOOLS[:-1]
SERVER_TOOL_LIMITATION = (
    "OpenRouter injects openrouter:web_search server-side; its provider-native prompt "
    "representation is unavailable to the local renderer."
)


@beartype
@dataclass(frozen=True, slots=True)
class RenderedProbeContext:
    """One exact token sequence and one or more non-overlapping measured spans."""

    input_ids: tuple[int, ...]
    token_positions: tuple[tuple[int, ...], ...]
    content_token_ids: tuple[tuple[int, ...], ...]
    render_config_sha256: str

    def __post_init__(self) -> None:
        if not self.input_ids or not self.token_positions:
            raise ValueError("rendered probe context must contain input and measured tokens")
        if len(self.token_positions) != len(self.content_token_ids):
            raise ValueError("rendered token positions and IDs must contain the same spans")
        flattened = [position for span in self.token_positions for position in span]
        if any(not span for span in self.token_positions):
            raise ValueError("every measured probe span must contain tokens")
        if len(set(flattened)) != len(flattened) or flattened != sorted(flattened):
            raise ValueError("measured probe spans must be ordered and non-overlapping")
        for positions, token_ids in zip(self.token_positions, self.content_token_ids, strict=True):
            if tuple(self.input_ids[position] for position in positions) != token_ids:
                raise ValueError("measured token IDs do not match the rendered input")


def _template_kwargs(adapter: NativeTemplateAdapter) -> dict[str, bool]:
    if adapter.name == NEMOTRON_ADAPTER.name:
        return {"enable_thinking": True, "truncate_history_thinking": False}
    if adapter.name == GEMMA_ADAPTER.name:
        return {"enable_thinking": True, "preserve_thinking": True}
    raise ValueError(f"unsupported native adapter: {adapter.name}")


def _render_config(adapter: NativeTemplateAdapter, *, tools: bool, generation: bool) -> str:
    payload = {
        "adapter": adapter.name,
        "add_generation_prompt": generation,
        "template_kwargs": _template_kwargs(adapter),
        "function_tools": FUNCTION_TOOLS if tools else [],
        "full_assistant_tools_sha256": hashlib.sha256(
            json.dumps(
                ASSISTANT_TOOLS,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        "server_tool_limitation": SERVER_TOOL_LIMITATION if tools else None,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def _apply_template(
    tokenizer: Any,
    adapter: NativeTemplateAdapter,
    messages: list[dict[str, Any]],
    *,
    tools: bool,
    generation: bool,
) -> str:
    kwargs: dict[str, Any] = _template_kwargs(adapter)
    if tools:
        kwargs["tools"] = list(FUNCTION_TOOLS)
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=generation,
        **kwargs,
    )
    if not isinstance(rendered, str):
        raise TypeError("native chat template must return text when tokenize=False")
    return rendered


def _ids(tokenizer: Any, text: str, *, offsets: bool = False) -> Mapping[str, Any]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        **({"return_offsets_mapping": True} if offsets else {}),
    )
    if not isinstance(encoded, Mapping):
        raise TypeError("tokenizer must return a mapping")
    return encoded


def _marker(index: int) -> str:
    return f"ROLE_PROBE_SPAN_{index}_72fce190"


def _render_spans(
    tokenizer: Any,
    adapter: NativeTemplateAdapter,
    messages: list[dict[str, Any]],
    replacements: tuple[tuple[int, str, str], ...],
    *,
    tools: bool,
    generation: bool,
) -> RenderedProbeContext:
    """Render marked fields, verify exact substitution, and recover content tokens."""
    validate_native_template(tokenizer, adapter)
    marked = [dict(message) for message in messages]
    spans: list[tuple[str, str]] = []
    for marker_index, (message_index, field, content) in enumerate(replacements):
        marker = _marker(marker_index)
        if marker in content:
            raise ValueError("probe content contains a reserved span marker")
        if marked[message_index].get(field) != content:
            raise ValueError("measured probe content does not match its message field")
        marked[message_index][field] = marker
        spans.append((marker, content))
    skeleton = _apply_template(tokenizer, adapter, marked, tools=tools, generation=generation)
    rendered = skeleton
    character_spans: list[tuple[int, int]] = []
    for marker, content in spans:
        if rendered.count(marker) != 1:
            raise ValueError("native template did not preserve a unique probe marker")
        start = rendered.index(marker)
        rendered = rendered.replace(marker, content, 1)
        character_spans.append((start, start + len(content)))
    actual = _apply_template(tokenizer, adapter, messages, tools=tools, generation=generation)
    if rendered != actual:
        raise ValueError("native template transformed measured content unexpectedly")

    encoded = _ids(tokenizer, actual, offsets=True)
    raw_ids = encoded.get("input_ids")
    raw_offsets = encoded.get("offset_mapping")
    if not isinstance(raw_ids, Sequence) or not isinstance(raw_offsets, Sequence):
        raise ValueError("a fast tokenizer with input IDs and offset mappings is required")
    input_ids = tuple(int(value) for value in raw_ids)
    offsets = tuple((int(value[0]), int(value[1])) for value in raw_offsets)
    if len(input_ids) != len(offsets):
        raise ValueError("token IDs and offset mappings must have equal length")

    positions_by_span: list[tuple[int, ...]] = []
    token_ids_by_span: list[tuple[int, ...]] = []
    for (start, stop), (_, content) in zip(character_spans, spans, strict=True):
        positions: list[int] = []
        for position, (token_start, token_stop) in enumerate(offsets):
            intersects = token_stop > start and token_start < stop
            contained = token_start >= start and token_stop <= stop and token_stop > token_start
            if intersects and not contained:
                raise ValueError("a tokenizer token crosses a measured content boundary")
            if contained:
                positions.append(position)
        if not positions:
            raise ValueError("measured content tokenized to an empty span")
        measured_ids = tuple(input_ids[position] for position in positions)
        plain = _ids(tokenizer, content).get("input_ids")
        if not isinstance(plain, Sequence) or measured_ids != tuple(int(value) for value in plain):
            raise ValueError("measured content tokens differ inside and outside the template")
        positions_by_span.append(tuple(positions))
        token_ids_by_span.append(measured_ids)
    return RenderedProbeContext(
        input_ids,
        tuple(positions_by_span),
        tuple(token_ids_by_span),
        _render_config(adapter, tools=tools, generation=generation),
    )


def _target_message_index(setup: ConversationSetup, position: int) -> int:
    if position not in {1, 2}:
        raise ValueError("probe target position must be 1 or 2")
    index = 0
    for spec in setup.matchup.inputs[: position - 1]:
        index += 2 if spec.channel is Channel.README else 1
    if setup.matchup.inputs[position - 1].channel is Channel.README:
        index += 1
    return index


@beartype
def render_collector_probe_context(
    tokenizer: Any,
    adapter: NativeTemplateAdapter,
    setup: ConversationSetup,
    position: int,
) -> RenderedProbeContext:
    """Render the actual pre-generation collector context and one input span."""
    messages = [dict(message) for message in setup.openrouter_messages()]
    message_index = _target_message_index(setup, position)
    content = str(setup.content_for_input(position - 1))
    return _render_spans(
        tokenizer,
        adapter,
        messages,
        ((message_index, "content", content),),
        tools=True,
        generation=True,
    )


@beartype
def render_native_dialogue_context(
    tokenizer: Any,
    adapter: NativeTemplateAdapter,
    *,
    prompt: str,
    reasoning: str,
    final: str,
) -> RenderedProbeContext:
    """Replay one completed native dialogue and retain reasoning/final spans only."""
    if not all(value and value.strip() for value in (prompt, reasoning, final)):
        raise ValueError("native prompt, reasoning, and final text must contain non-whitespace text")
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": final, adapter.reasoning_field: reasoning},
    ]
    return _render_spans(
        tokenizer,
        adapter,
        messages,
        ((1, adapter.reasoning_field, reasoning), (1, "content", final)),
        tools=False,
        generation=False,
    )
