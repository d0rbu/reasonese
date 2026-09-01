from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from reasonese.axes import Assistant, Author, Channel, Framing, Instruction
from reasonese.config import load_matchup
from reasonese.conversation import (
    ChatMessage,
    ChatRole,
    ConversationSetup,
    GeneratedMessage,
    GeneratedText,
    ToolCall,
    ToolCallId,
    ToolName,
    ToolResult,
    authoring_request,
    construct_conversation,
)
from reasonese.matchup import (
    MatchupInputs,
    make_matchup,
    matchup_from_dict,
    matchup_to_dict,
    prompt_spec_from_dict,
)
from reasonese.planning import PromptSpec


def _spec(
    text: str,
    channel: Channel,
    *,
    author: Author = Author.INKLING,
    framing: Framing = Framing.NORMAL,
) -> PromptSpec:
    return PromptSpec(Instruction.parse(text), framing, channel, author)


def test_matchup_refined_type_allows_repeated_channels_and_three_or_more_inputs() -> None:
    inputs = (
        _spec("First system instruction.", Channel.SYSTEM),
        _spec("Second system instruction.", Channel.SYSTEM),
        _spec("First user instruction.", Channel.USER),
        _spec("Second user instruction.", Channel.USER),
    )
    matchup = make_matchup(inputs, Assistant.QWEN3_8_FLASH)

    assert isinstance(matchup.inputs, MatchupInputs)
    assert matchup.inputs == inputs
    assert matchup.assistant is Assistant.QWEN3_8_FLASH


def test_matchup_rejects_too_few_inputs_or_no_explicit_user_message() -> None:
    with pytest.raises(ValueError, match="at least two"):
        make_matchup((_spec("Only one.", Channel.USER),), Assistant.INKLING)
    with pytest.raises(ValueError, match="at least one user"):
        make_matchup(
            (
                _spec("System.", Channel.SYSTEM),
                _spec("Read me.", Channel.README),
            ),
            Assistant.INKLING,
        )


def test_matchup_yaml_round_trip_and_example() -> None:
    example = load_matchup(Path("configs/example_matchup.yaml"))
    assert len(example.inputs) == 2
    assert example.assistant is Assistant.QWEN3_8_FLASH
    assert matchup_from_dict(matchup_to_dict(example)) == example


@pytest.mark.parametrize(
    ("raw", "error"),
    [
        ([], "matchup must be a mapping"),
        ({"assistant": "Inkling"}, "matchup fields"),
        ({"assistant": "Inkling", "inputs": "bad"}, "inputs must be a list"),
        (
            {
                "assistant": "Inkling",
                "inputs": [{"instruction": "One.", "framing": "normal", "channel": "user message"}],
            },
            "input fields",
        ),
    ],
)
def test_matchup_yaml_validation(raw: object, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        matchup_from_dict(raw)


def test_prompt_spec_yaml_requires_a_mapping() -> None:
    with pytest.raises(ValueError, match="input must be a mapping"):
        prompt_spec_from_dict("bad")


def test_authoring_request_includes_all_treatment_coordinates() -> None:
    spec = _spec(
        "Do the task.",
        Channel.README,
        framing=Framing.REASONESE_PERSUASIVE,
    )
    request = authoring_request(spec)
    assert [message["role"] for message in request["messages"]] == ["user"]
    prompt = request["messages"][0]["content"]
    assert "README.md" in prompt
    assert "compressed planning shorthand" in prompt
    assert "agent-consensus cues" in prompt
    assert "Do the task." in prompt
    assert "Delivery channel:" not in prompt
    assert "Framing:" not in prompt
    assert request["reasoning"] == {"enabled": True, "exclude": False}
    assert request["temperature"] == 0.7


@pytest.mark.parametrize(
    ("framing", "phrase"),
    [
        (Framing.NORMAL, "clear, neutral"),
        (Framing.CASUAL, "mostly lowercase"),
        (Framing.PERSUASIVE, "social proof"),
        (Framing.SUBAGENT, "parent agent"),
        (Framing.REASONESE_NORMAL, "internal reasoning trace"),
        (Framing.REASONESE_PERSUASIVE, "agent-consensus cues"),
    ],
)
def test_every_framing_has_an_explicit_natural_brief(framing: Framing, phrase: str) -> None:
    prompt = authoring_request(_spec("Task.", Channel.USER, framing=framing))["messages"][0][
        "content"
    ]
    assert phrase in prompt


@pytest.mark.parametrize(
    ("channel", "phrase"),
    [
        (Channel.SYSTEM, "another model's system message"),
        (Channel.USER, "sent directly to another model"),
        (Channel.README, "repository's README.md"),
    ],
)
def test_every_channel_explains_how_the_target_encounters_text(
    channel: Channel, phrase: str
) -> None:
    prompt = authoring_request(_spec("Task.", channel))["messages"][0]["content"]
    assert phrase in prompt


def test_conversation_preserves_input_order_and_reads_readme_with_a_tool() -> None:
    specs = (
        _spec("System one.", Channel.SYSTEM),
        _spec("File instruction.", Channel.README),
        _spec("User one.", Channel.USER),
        _spec("User two.", Channel.USER),
    )
    matchup = make_matchup(specs, Assistant.INKLING_SMALL)
    generated = tuple(
        GeneratedMessage(spec, GeneratedText.parse(f"generated {index}"), {})
        for index, spec in enumerate(specs)
    )

    setup = construct_conversation(matchup, generated)

    assert [message.role for message in setup.messages] == [
        ChatRole.SYSTEM,
        ChatRole.ASSISTANT,
        ChatRole.TOOL,
        ChatRole.USER,
        ChatRole.USER,
    ]
    assistant_message = setup.openrouter_messages()[1]
    call_id = assistant_message["tool_calls"][0]["id"]
    assert call_id.startswith("call_")
    digest = call_id.removeprefix("call_")
    assert len(digest) == 32
    assert int(digest, 16) > 0
    assert assistant_message == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
            }
        ],
    }
    assert setup.openrouter_messages()[2] == {
        "role": "tool",
        "tool_call_id": call_id,
        "content": "generated 1",
    }
    assert setup.openrouter_messages()[3] == {"role": "user", "content": "generated 2"}
    assert setup.content_for_input(1) == "generated 1"
    assert setup.readme_contents() == ("generated 1",)
    assert construct_conversation(matchup, generated).openrouter_messages() == (
        setup.openrouter_messages()
    )


def test_repeated_identical_readme_datapoints_receive_distinct_stable_ids() -> None:
    readme = _spec("File instruction.", Channel.README)
    user = _spec("User instruction.", Channel.USER)
    matchup = make_matchup((readme, readme, user), Assistant.INKLING)
    generated = (
        GeneratedMessage(readme, GeneratedText.parse("same file text"), {}),
        GeneratedMessage(readme, GeneratedText.parse("same file text"), {}),
        GeneratedMessage(user, GeneratedText.parse("user text"), {}),
    )

    first = construct_conversation(matchup, generated).openrouter_messages()
    second = construct_conversation(matchup, generated).openrouter_messages()

    assert first[0]["tool_calls"][0]["id"] != first[2]["tool_calls"][0]["id"]
    assert first == second


def test_chat_messages_and_setup_reject_invalid_role_shapes() -> None:
    text = GeneratedText.parse("text")
    call_id = ToolCallId.parse("call")
    call = ToolCall(call_id, ToolName.parse("read_file"), '{"path":"README.md"}')
    with pytest.raises(ValueError, match="system and user"):
        ChatMessage(ChatRole.SYSTEM)
    with pytest.raises(ValueError, match="assistant setup"):
        ChatMessage(ChatRole.ASSISTANT, text)
    with pytest.raises(ValueError, match="tool messages"):
        ChatMessage(ChatRole.TOOL, text)

    matchup = make_matchup(
        (_spec("System.", Channel.SYSTEM), _spec("User.", Channel.USER)),
        Assistant.INKLING,
    )
    with pytest.raises(ValueError, match="do not match"):
        ConversationSetup(
            matchup,
            (
                ChatMessage(ChatRole.USER, text),
                ChatMessage(ChatRole.USER, text),
            ),
        )
    with pytest.raises(ValueError, match="do not map"):
        ConversationSetup(
            matchup,
            (
                ChatMessage(ChatRole.SYSTEM, text),
                ChatMessage(ChatRole.USER, text),
                ChatMessage(ChatRole.ASSISTANT, tool_calls=(call,)),
            ),
        )
    assert ToolResult(call_id, text).openrouter_dict()["content"] == "text"


def test_setup_content_lookup_checks_bounds() -> None:
    specs = (_spec("System.", Channel.SYSTEM), _spec("User.", Channel.USER))
    matchup = make_matchup(specs, Assistant.INKLING)
    setup = construct_conversation(
        matchup,
        tuple(GeneratedMessage(spec, GeneratedText.parse("text"), None) for spec in specs),
    )
    with pytest.raises(IndexError):
        setup.content_for_input(-1)
    with pytest.raises(IndexError):
        setup.content_for_input(2)


@pytest.mark.parametrize(
    ("calls", "result_id", "error"),
    [
        (
            (
                ToolCall(
                    ToolCallId.parse("one"),
                    ToolName.parse("read_file"),
                    '{"path":"README.md"}',
                ),
                ToolCall(
                    ToolCallId.parse("two"),
                    ToolName.parse("read_file"),
                    '{"path":"README.md"}',
                ),
            ),
            "one",
            "exactly one",
        ),
        (
            (ToolCall(ToolCallId.parse("one"), ToolName.parse("read_file"), "{"),),
            "one",
            "valid JSON",
        ),
        (
            (ToolCall(ToolCallId.parse("one"), ToolName.parse("bash"), "{}"),),
            "one",
            "matching read_file",
        ),
        (
            (
                ToolCall(
                    ToolCallId.parse("one"),
                    ToolName.parse("read_file"),
                    '{"path":"README.md"}',
                ),
            ),
            "different",
            "matching read_file",
        ),
    ],
)
def test_readme_setup_requires_a_matching_file_read_pair(
    calls: tuple[ToolCall, ...], result_id: str, error: str
) -> None:
    matchup = make_matchup(
        (_spec("File.", Channel.README), _spec("User.", Channel.USER)), Assistant.INKLING
    )
    with pytest.raises(ValueError, match=error):
        ConversationSetup(
            matchup,
            (
                ChatMessage(ChatRole.ASSISTANT, tool_calls=calls),
                ChatMessage(
                    ChatRole.TOOL,
                    GeneratedText.parse("file"),
                    tool_call_id=ToolCallId.parse(result_id),
                ),
                ChatMessage(ChatRole.USER, GeneratedText.parse("user")),
            ),
        )


def test_conversation_rejects_missing_or_misordered_generated_messages() -> None:
    first = _spec("System.", Channel.SYSTEM)
    second = _spec("User.", Channel.USER)
    matchup = make_matchup((first, second), Assistant.INKLING)
    first_message = GeneratedMessage(first, GeneratedText.parse("first"), {})
    second_message = GeneratedMessage(second, GeneratedText.parse("second"), {})

    with pytest.raises(ValueError, match="count"):
        construct_conversation(matchup, (first_message,))
    with pytest.raises(ValueError, match="order"):
        construct_conversation(matchup, (second_message, first_message))


def test_channel_rendering_fails_closed_for_future_unhandled_values() -> None:
    system = _spec("System.", Channel.SYSTEM)
    user = _spec("User.", Channel.USER)
    matchup = make_matchup((system, user), Assistant.INKLING)
    object.__setattr__(user, "channel", cast(Any, "future channel"))
    generated = (
        GeneratedMessage(system, GeneratedText.parse("system"), {}),
        GeneratedMessage(user, GeneratedText.parse("user"), {}),
    )

    with pytest.raises(ValueError, match="unsupported channel"):
        construct_conversation(matchup, generated)
