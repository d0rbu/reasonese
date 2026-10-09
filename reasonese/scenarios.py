"""Scenario conversations that embed a pair's two instructions in shared context.

A scenario belongs to one instruction pair and is stored as ``<pair id>.yaml``.
It lists the messages of a conversation as Jinja templates over six slots:
``system``, ``user``, and ``tool``, each with an ``early`` and a ``late``
position. The first matchup input fills the early slot of its channel and the
second fills the late slot of its channel, so the two delivery orders of a cell
pair stay distinct and delivery position keeps its meaning. Every other slot
is empty. Pairs without a scenario file keep the bare conversation, which is
also what every pair gets when no scenario directory is selected.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import yaml
from beartype import beartype
from jinja2 import StrictUndefined, Template, TemplateError
from jinja2.sandbox import SandboxedEnvironment

from reasonese.axes import Assistant, Channel, Instruction
from reasonese.conversation import (
    ChatMessage,
    ChatRole,
    ConversationSetup,
    GeneratedMessage,
    GeneratedText,
    Placement,
    ToolCall,
    ToolName,
    construct_conversation,
    tool_call_id,
)
from reasonese.instructions import (
    InstructionPair,
    PairId,
    PairMembership,
    instruction_index,
    load_instruction_pairs,
)
from reasonese.matchup import Matchup

_SLOT = {Channel.SYSTEM: "system", Channel.USER: "user", Channel.README: "tool"}
_ROLE = {Channel.SYSTEM: ChatRole.SYSTEM, Channel.USER: ChatRole.USER, Channel.README: ChatRole.TOOL}
_POSITIONS = ("early", "late")
_README = "README.md"
_README_ARGUMENTS = json.dumps({"path": _README}, separators=(",", ":"))
_SCENARIO_FIELDS = frozenset({"source", "adaptation", "messages"})
_MESSAGE_FIELDS = frozenset({"role", "content", "when", "reads"})
# Every ordered channel combination a matchup can take: it needs one user message.
_ORDERINGS = (
    (Channel.SYSTEM, Channel.USER),
    (Channel.USER, Channel.SYSTEM),
    (Channel.USER, Channel.USER),
    (Channel.USER, Channel.README),
    (Channel.README, Channel.USER),
)
# Two paragraphs each, so a template that reflows or indents authored text fails at load.
_PROBE_TEXTS = (
    "probe text for the input delivered first\n\nwith a second paragraph",
    "probe text for the other input\n\nalso in two paragraphs",
)
# Block tags on their own line leave no blank line behind, so conditional sections read cleanly.
_ENVIRONMENT = SandboxedEnvironment(
    undefined=StrictUndefined, autoescape=False, trim_blocks=True, lstrip_blocks=True
)

Slots = dict[str, dict[str, str]]


@cache
def _template(source: str) -> Template:
    return _ENVIRONMENT.from_string(source)


@cache
def _condition(source: str) -> Any:
    return _ENVIRONMENT.compile_expression(source, undefined_to_none=False)


def _slots(channels: tuple[Channel, ...], contents: tuple[str, ...]) -> Slots:
    slots: Slots = {name: dict.fromkeys(_POSITIONS, "") for name in _SLOT.values()}
    for position, channel, content in zip(_POSITIONS, channels, contents, strict=True):
        slots[_SLOT[channel]][position] = content
    return slots


def _render(source: str, slots: Slots) -> str:
    try:
        return _template(source).render(**slots).strip()
    except TemplateError as error:
        raise ValueError(f"scenario template failed to render: {error}") from error


def _holds(source: str, slots: Slots) -> bool:
    try:
        return bool(_condition(source)(**slots))
    except TemplateError as error:
        raise ValueError(f"scenario condition failed to evaluate: {error}") from error


@beartype
@dataclass(frozen=True, slots=True)
class ScenarioMessage:
    """One templated message; ``when`` is a Jinja expression over the slots."""

    role: ChatRole
    content: str | None = None
    when: str | None = None
    reads_readme: bool = False


@beartype
@dataclass(frozen=True, slots=True)
class Scenario:
    """A pair's shared conversation context, with its provenance.

    ``source`` says where the real case came from and ``adaptation`` says what
    was changed, so a scenario's realism can be audited and revised later.
    """

    pair_id: PairId
    source: str
    adaptation: str
    messages: tuple[ScenarioMessage, ...]

    def __post_init__(self) -> None:
        if not self.source.strip() or not self.adaptation.strip():
            raise ValueError(f"scenario {self.pair_id} must state its source and adaptation")
        if not self.messages:
            raise ValueError(f"scenario {self.pair_id} must contain messages")
        for index, message in enumerate(self.messages):
            following = self.messages[index + 1] if index + 1 < len(self.messages) else None
            match message.role:
                case ChatRole.SYSTEM | ChatRole.USER:
                    if message.content is None or message.reads_readme:
                        raise ValueError("scenario system and user messages must contain only text")
                case ChatRole.ASSISTANT:
                    if message.content is None and not message.reads_readme:
                        raise ValueError("scenario assistant messages need text or a README read")
                    if message.reads_readme and (
                        following is None or following.role is not ChatRole.TOOL
                    ):
                        raise ValueError("a scenario README read must be followed by its result")
                case ChatRole.TOOL:
                    caller = self.messages[index - 1] if index else None
                    if (
                        message.content is None
                        or message.when is not None
                        or message.reads_readme
                        or caller is None
                        or caller.role is not ChatRole.ASSISTANT
                        or not caller.reads_readme
                    ):
                        raise ValueError(
                            "a scenario tool message is the unconditional text result of the "
                            "README read before it"
                        )
                case _:
                    raise ValueError(f"unsupported scenario role: {message.role}")
            try:
                if message.content is not None:
                    _template(message.content)
                if message.when is not None:
                    _condition(message.when)
            except TemplateError as error:
                raise ValueError(f"scenario {self.pair_id} has an invalid template: {error}") from error
        # A scenario has to place both inputs for every ordering a study can sample.
        assistant = next(iter(Assistant))
        for channels in _ORDERINGS:
            messages, _ = self.render(channels, _PROBE_TEXTS, assistant)
            conversation = "\n".join(str(message.content) for message in messages if message.content)
            if any(conversation.count(probe) != 1 for probe in _PROBE_TEXTS):
                raise ValueError(f"scenario {self.pair_id} must deliver each input exactly once")

    def _layout(self, slots: Slots) -> dict[int, tuple[ScenarioMessage, str]]:
        """Return the included messages and their rendered text, keyed by template index."""
        layout: dict[int, tuple[ScenarioMessage, str]] = {}
        included = True
        for index, message in enumerate(self.messages):
            if message.role is not ChatRole.TOOL:
                # A tool result shares the fate of the README read before it.
                included = message.when is None or _holds(message.when, slots)
            if included:
                text = "" if message.content is None else _render(message.content, slots)
                layout[index] = (message, text)
        return layout

    @beartype
    def render(
        self,
        channels: tuple[Channel, ...],
        contents: tuple[str, ...],
        assistant: Assistant,
    ) -> tuple[tuple[ChatMessage, ...], tuple[Placement, ...]]:
        """Render the conversation and locate the message that carries each input."""
        slots = _slots(channels, contents)
        layout = self._layout(slots)
        order = list(layout)
        rendered = [layout[index] for index in order]

        placements: list[Placement] = []
        for index, (position, channel) in enumerate(zip(_POSITIONS, channels, strict=True)):
            slot = f"{_SLOT[channel]}.{position}"
            blanked = {name: dict(values) for name, values in slots.items()}
            blanked[_SLOT[channel]][position] = ""
            without = self._layout(blanked)
            carriers: list[int] = []
            for template_index in sorted(set(layout) | set(without)):
                if layout.get(template_index) == without.get(template_index):
                    continue
                message = self.messages[template_index]
                depends_on_slot = (
                    template_index in layout
                    and message.content is not None
                    and _render(message.content, blanked) != layout[template_index][1]
                )
                if depends_on_slot:
                    carriers.append(template_index)
                else:
                    raise ValueError(
                        f"scenario {self.pair_id} changes shared context with {slot}; "
                        "only the message that carries a slot may depend on it"
                    )
            if len(carriers) != 1:
                raise ValueError(
                    f"scenario {self.pair_id} must render {slot} in exactly one message, "
                    f"found {len(carriers)}"
                )
            carrier = order.index(carriers[0])
            message, text = rendered[carrier]
            if message.role is not _ROLE[channel]:
                raise ValueError(f"scenario {self.pair_id} renders {slot} in a {message.role} message")
            if contents[index] not in text:
                raise ValueError(f"scenario {self.pair_id} does not render {slot} verbatim")
            if text.count(contents[index]) != 1:
                raise ValueError(f"scenario {self.pair_id} must deliver {slot} exactly once")
            if placements and carrier <= placements[-1].message:
                raise ValueError(f"scenario {self.pair_id} must render early slots before late ones")
            placements.append(Placement(carrier, GeneratedText.parse(contents[index])))

        if sum(1 for message, _ in rendered if message.reads_readme) > 1:
            raise ValueError(f"scenario {self.pair_id} reads {_README} more than once")
        messages: list[ChatMessage] = []
        call_id = None
        for index, (message, text) in enumerate(rendered):
            if message.role is not ChatRole.ASSISTANT and not text:
                raise ValueError(f"scenario {self.pair_id} rendered an empty {message.role} message")
            content = GeneratedText.parse(text) if text else None
            if message.reads_readme:
                identity = json.dumps(
                    {"assistant": assistant, "pair": self.pair_id, _README: rendered[index + 1][1]},
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                call_id = tool_call_id(assistant, identity)
                call = ToolCall(call_id, ToolName.parse("read_file"), _README_ARGUMENTS)
                messages.append(ChatMessage(ChatRole.ASSISTANT, content, (call,)))
            elif message.role is ChatRole.TOOL:
                messages.append(ChatMessage(ChatRole.TOOL, content, tool_call_id=call_id))
            else:
                if content is None:
                    raise ValueError(f"scenario {self.pair_id} rendered an empty assistant message")
                messages.append(ChatMessage(message.role, content))
        if messages[-1].role is ChatRole.ASSISTANT:
            # A trailing assistant turn would be continued as a prefill, not answered.
            raise ValueError(f"scenario {self.pair_id} must not end on an assistant turn")
        return tuple(messages), tuple(placements)

    @beartype
    def construct(
        self,
        matchup: Matchup,
        generated_messages: tuple[GeneratedMessage, ...],
    ) -> ConversationSetup:
        """Place a matchup's authored messages inside this scenario."""
        if tuple(generated.spec for generated in generated_messages) != tuple(matchup.inputs):
            raise ValueError("generated messages must follow matchup input order")
        messages, placements = self.render(
            tuple(spec.channel for spec in matchup.inputs),
            tuple(str(generated.content) for generated in generated_messages),
            matchup.assistant,
        )
        return ConversationSetup(matchup, messages, placements)


def _message_from_dict(raw: object) -> ScenarioMessage:
    if not isinstance(raw, dict) or not set(raw) <= _MESSAGE_FIELDS or "role" not in raw:
        raise ValueError(f"a scenario message needs a role and only {sorted(_MESSAGE_FIELDS)}")
    data = cast(dict[str, Any], raw)
    content, when, reads = data.get("content"), data.get("when"), data.get("reads")
    if content is not None and not isinstance(content, str):
        raise ValueError("scenario message content must be text")
    if when is not None and not isinstance(when, str):
        raise ValueError("scenario message 'when' must be a Jinja expression")
    if reads not in (None, _README):
        raise ValueError(f"a scenario can only read {_README}")
    return ScenarioMessage(ChatRole(data["role"]), content, when, reads is not None)


@beartype
def scenario_from_dict(pair_id: PairId, raw: object) -> Scenario:
    """Parse one scenario from YAML-compatible data."""
    if not isinstance(raw, dict) or set(raw) != _SCENARIO_FIELDS:
        raise ValueError(f"scenario fields must be {sorted(_SCENARIO_FIELDS)}")
    data = cast(dict[str, Any], raw)
    if not isinstance(data["source"], str) or not isinstance(data["adaptation"], str):
        raise ValueError("scenario source and adaptation must be text")
    if not isinstance(data["messages"], list):
        raise ValueError("scenario messages must be a list")
    return Scenario(
        pair_id,
        data["source"],
        data["adaptation"],
        tuple(_message_from_dict(message) for message in data["messages"]),
    )


@beartype
@dataclass(frozen=True, slots=True)
class ScenarioLibrary:
    """The scenarios selected for a run, keyed by instruction pair."""

    scenarios: Mapping[PairId, Scenario]
    memberships: Mapping[Instruction, PairMembership]

    @beartype
    def scenario_for(self, matchup: Matchup) -> Scenario | None:
        """Return the matchup's pair scenario, or ``None`` when its pair has none."""
        pair_ids: set[PairId] = set()
        for spec in matchup.inputs:
            membership = self.memberships.get(spec.instruction)
            if membership is None:
                raise ValueError(
                    f"scenario runs need every instruction in the pair bank: {spec.instruction}"
                )
            pair_ids.add(membership.pair.pair_id)
        if len(pair_ids) != 1:
            raise ValueError("a scenario matchup must hold the two sides of one instruction pair")
        return self.scenarios.get(next(iter(pair_ids)))


@beartype
def load_scenarios(root: Path, pairs: tuple[InstructionPair, ...]) -> ScenarioLibrary:
    """Load every ``<pair id>.yaml`` scenario below ``root`` against the pair bank."""
    if not root.is_dir():
        raise ValueError(f"scenario directory does not exist: {root}")
    pair_ids = {str(pair.pair_id): pair.pair_id for pair in pairs}
    scenarios: dict[PairId, Scenario] = {}
    for path in sorted(root.glob("*.yaml")):
        pair_id = pair_ids.get(path.stem)
        if pair_id is None:
            raise ValueError(f"scenario file does not name an instruction pair: {path}")
        with path.open(encoding="utf-8") as handle:
            scenarios[pair_id] = scenario_from_dict(pair_id, yaml.safe_load(handle))
    if not scenarios:
        # A mistyped directory must not silently fall back to bare conversations.
        raise ValueError(f"scenario directory holds no <pair id>.yaml files: {root}")
    return ScenarioLibrary(
        MappingProxyType(scenarios), MappingProxyType(instruction_index(pairs))
    )


@beartype
def build_conversation(
    matchup: Matchup,
    generated_messages: tuple[GeneratedMessage, ...],
    scenarios: ScenarioLibrary | None = None,
) -> ConversationSetup:
    """Build the selected scenario, falling back to a bare conversation."""
    scenario = scenarios.scenario_for(matchup) if scenarios is not None else None
    if scenario is None:
        return construct_conversation(matchup, generated_messages)
    return scenario.construct(matchup, generated_messages)


@beartype
def require_scenario_selection(scenarios: ScenarioLibrary | None, setup: ConversationSetup) -> None:
    """Refuse a cached setup from another scenario selection instead of collecting over it."""
    try:
        scenario = scenarios.scenario_for(setup.matchup) if scenarios is not None else None
        if scenario is None or setup.placements is None:
            matches = scenario is None and setup.placements is None
        else:
            generated = tuple(
                GeneratedMessage(spec, setup.content_for_input(index), None)
                for index, spec in enumerate(setup.matchup.inputs)
            )
            matches = scenario.construct(setup.matchup, generated) == setup
    except ValueError:
        matches = False
    if not matches:
        raise ValueError(
            "cached traces were collected under a different scenario selection or an edited "
            "scenario; use a fresh output directory"
        )


def add_scenario_arguments(parser: argparse.ArgumentParser) -> None:
    """Keep the scenario options identical across collection commands."""
    parser.add_argument(
        "--scenarios",
        type=Path,
        help=(
            "directory of <pair id>.yaml scenarios; omit for bare conversations. Use a fresh "
            "output directory when changing this selection"
        ),
    )
    parser.add_argument("--pairs", type=Path, default=Path("configs/instruction_pairs.yaml"))


def scenarios_from_arguments(args: argparse.Namespace) -> ScenarioLibrary | None:
    """Load the selected scenario directory, if any."""
    if args.scenarios is None:
        return None
    return load_scenarios(args.scenarios, load_instruction_pairs(args.pairs))
