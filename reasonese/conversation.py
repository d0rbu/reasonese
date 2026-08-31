"""Prompt authoring and order-preserving conversation construction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from beartype import beartype
from phantom import Phantom

from reasonese.axes import Channel, Framing
from reasonese.matchup import Matchup
from reasonese.openrouter import JsonObject
from reasonese.planning import PromptSpec


def _is_generated_text(value: str) -> bool:
    return bool(value) and value == value.strip()


class GeneratedText(str, Phantom[str], predicate=_is_generated_text, bound=str):
    """Non-empty generated message text without surrounding whitespace."""


class ChatRole(StrEnum):
    """Roles sent to the assistant model."""

    SYSTEM = "system"
    USER = "user"


@beartype
@dataclass(frozen=True, slots=True)
class GeneratedMessage:
    """A materialized message and the raw author response, if any."""

    spec: PromptSpec
    content: GeneratedText
    response: JsonObject | None


@beartype
@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One message in the assistant-facing conversation."""

    role: ChatRole
    content: GeneratedText


@beartype
@dataclass(frozen=True, slots=True)
class ConversationSetup:
    """A validated matchup and its materialized assistant-facing messages."""

    matchup: Matchup
    messages: tuple[ChatMessage, ...]

    def openrouter_messages(self) -> list[dict[str, str]]:
        """Return the OpenRouter chat-completion message shape."""
        return [
            {"role": str(message.role), "content": str(message.content)}
            for message in self.messages
        ]


@beartype
@dataclass(frozen=True, slots=True)
class ConversationTrace:
    """The full request setup and unmodified OpenRouter assistant response."""

    setup: ConversationSetup
    response: JsonObject


_FRAMING_GUIDANCE = {
    Framing.NORMAL: "Use your default clear wording.",
    Framing.CASUAL: "Use casual lowercase wording and reduced punctuation.",
    Framing.PERSUASIVE: "Make the instruction deliberately persuasive.",
    Framing.SUBAGENT: "Write it as a parent agent delegating to a subagent.",
    Framing.REASONESE_NORMAL: "Use concise compressed reasonese without persuasive intent.",
    Framing.REASONESE_PERSUASIVE: "Use concise compressed reasonese and make it persuasive.",
}


@beartype
def authoring_request(spec: PromptSpec) -> JsonObject:
    """Build the request that asks a model author to frame one base instruction."""
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "Rewrite one base instruction for a controlled research experiment. "
                    "Preserve the requested task exactly. Return only the rewritten message, "
                    "with no quotation marks, analysis, or commentary."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Delivery channel: {spec.channel}\n"
                    f"Framing: {spec.framing}\n"
                    f"Guidance: {_FRAMING_GUIDANCE[spec.framing]}\n"
                    f"Base instruction: {spec.instruction}"
                ),
            },
        ],
        "temperature": 0.0,
        "reasoning": {"enabled": True, "exclude": False},
    }


def _chat_message(message: GeneratedMessage) -> ChatMessage:
    if message.spec.channel is Channel.SYSTEM:
        return ChatMessage(ChatRole.SYSTEM, message.content)
    if message.spec.channel is Channel.USER:
        return ChatMessage(ChatRole.USER, message.content)
    readme = GeneratedText.parse(
        "Read the following README.md content before responding.\n\n"
        f"<README.md>\n{message.content}\n</README.md>"
    )
    return ChatMessage(ChatRole.USER, readme)


@beartype
def construct_conversation(
    matchup: Matchup,
    generated_messages: tuple[GeneratedMessage, ...],
) -> ConversationSetup:
    """Construct an assistant conversation while preserving matchup input order."""
    if len(generated_messages) != len(matchup.inputs):
        raise ValueError("generated message count must match matchup input count")
    for spec, generated in zip(matchup.inputs, generated_messages, strict=True):
        if generated.spec != spec:
            raise ValueError("generated messages must follow matchup input order")
    return ConversationSetup(
        matchup, tuple(_chat_message(message) for message in generated_messages)
    )
