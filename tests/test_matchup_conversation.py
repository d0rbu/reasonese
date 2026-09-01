from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from reasonese.axes import Assistant, Author, Channel, Framing, Instruction
from reasonese.config import load_matchup
from reasonese.conversation import (
    ChatRole,
    GeneratedMessage,
    GeneratedText,
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
    prompt = request["messages"][1]["content"]
    assert "README.md" in prompt
    assert "reasonese-persuasive" in prompt
    assert "Do the task." in prompt
    assert request["reasoning"] == {"enabled": True, "exclude": False}
    assert request["temperature"] == 0.7


def test_conversation_preserves_input_order_and_wraps_readme_content() -> None:
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
        ChatRole.USER,
        ChatRole.USER,
        ChatRole.USER,
    ]
    assert "<README.md>\ngenerated 1\n</README.md>" in setup.messages[1].content
    assert setup.openrouter_messages()[2] == {"role": "user", "content": "generated 2"}


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
