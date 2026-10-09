"""Tests for scenario conversations: slots, placement, caching, and the collection wiring.

The fixture scenario is synthetic and exists only to exercise the template
slots. Structural rules are checked on a small inline scenario so each
rejection changes one field at a time.
"""

from __future__ import annotations

import copy
import json
from functools import cache
from pathlib import Path
from typing import Any

import pytest
import yaml

from reasonese.axes import Assistant, Author, Channel, Framing, Instruction, author_framings
from reasonese.cache import trace_from_dict, trace_to_dict
from reasonese.collect_data import collect_study
from reasonese.collect_data import main as collect_data
from reasonese.collect_studies import main as collect_studies
from reasonese.conversation import (
    ChatMessage,
    ChatRole,
    ConversationSetup,
    ConversationTrace,
    GeneratedMessage,
    GeneratedText,
    Placement,
    ToolCall,
    ToolCallId,
    ToolName,
    construct_conversation,
)
from reasonese.instructions import InstructionPair, PairId, load_instruction_pairs
from reasonese.judging import judge_requests, trace_fingerprint
from reasonese.manual_messages import ManualMessageLibrary
from reasonese.matchup import Matchup, make_matchup, matchup_to_dict
from reasonese.openrouter import JsonObject, OpenRouterClient, RoutePreference
from reasonese.planning import PromptSpec
from reasonese.probe_qa import ProbeQaMode
from reasonese.probe_rendering import _target_message_index
from reasonese.routing import CollectionRouting
from reasonese.run_conversation import main as run_conversation
from reasonese.scenarios import (
    Scenario,
    ScenarioLibrary,
    build_conversation,
    load_scenarios,
    require_scenario_selection,
    scenario_from_dict,
)
from reasonese.show_scenario import main as show_scenario
from reasonese.study import Study, make_study, study_to_dict

BANK = Path("configs/instruction_pairs.yaml")
SCENARIOS = Path("tests/fixtures/scenarios")
PAIR_ID = "project-tree-bash-vs-python"
ORDERINGS = (
    (Channel.SYSTEM, Channel.USER),
    (Channel.USER, Channel.SYSTEM),
    (Channel.USER, Channel.USER),
    (Channel.USER, Channel.README),
    (Channel.README, Channel.USER),
)
TEXTS = ("Authored FIRST text.", "Authored SECOND text.")
ROLE = {Channel.SYSTEM: ChatRole.SYSTEM, Channel.USER: ChatRole.USER, Channel.README: ChatRole.TOOL}


@cache
def _pairs() -> tuple[InstructionPair, ...]:
    return load_instruction_pairs(BANK)


def _pair(pair_id: str = PAIR_ID) -> InstructionPair:
    return next(pair for pair in _pairs() if str(pair.pair_id) == pair_id)


@cache
def _library() -> ScenarioLibrary:
    return load_scenarios(SCENARIOS, _pairs())


def _matchup(
    first: Channel,
    second: Channel,
    *,
    assistant: Assistant = Assistant.INKLING,
    pair_id: str = PAIR_ID,
) -> Matchup:
    pair = _pair(pair_id)
    return make_matchup(
        (
            PromptSpec(pair.first, Framing.NORMAL, first, Author.USER),
            PromptSpec(pair.second, Framing.NORMAL, second, Author.USER),
        ),
        assistant,
    )


def _generated(
    matchup: Matchup, texts: tuple[str, str] = TEXTS
) -> tuple[GeneratedMessage, ...]:
    return tuple(
        GeneratedMessage(spec, GeneratedText.parse(text), None)
        for spec, text in zip(matchup.inputs, texts, strict=True)
    )


def _setup(first: Channel, second: Channel, **options: Any) -> ConversationSetup:
    matchup = _matchup(first, second, **options)
    return build_conversation(matchup, _generated(matchup), _library())


def _chat(content: str, response_id: str = "response-1") -> JsonObject:
    return {"id": response_id, "choices": [{"message": {"role": "assistant", "content": content}}]}


# --------------------------------------------------------------------------
# Rendering and placement
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("channels", "roles", "indices"),
    [
        (ORDERINGS[0], ["system", "system", "user", "assistant", "tool", "user"], [1, 5]),
        (ORDERINGS[1], ["system", "user", "user", "assistant", "tool", "system"], [2, 5]),
        (ORDERINGS[2], ["system", "user", "user", "assistant", "tool", "user"], [2, 5]),
        (ORDERINGS[3], ["system", "user", "user", "assistant", "tool"], [2, 4]),
        (ORDERINGS[4], ["system", "user", "assistant", "tool", "user"], [3, 4]),
    ],
)
def test_every_ordering_places_each_input_once_in_matchup_order(
    channels: tuple[Channel, Channel], roles: list[str], indices: list[int]
) -> None:
    setup = _setup(*channels)

    assert [str(message.role) for message in setup.messages] == roles
    assert "coding agent" in str(setup.messages[0].content)
    assert setup.placements is not None
    # Delivery position is the matchup order, so the first input is always rendered first.
    assert [placement.message for placement in setup.placements] == indices
    everything = "\n".join(str(message.content) for message in setup.messages if message.content)
    for index, (channel, text) in enumerate(zip(channels, TEXTS, strict=True)):
        carrier = setup.messages[setup.message_index_for_input(index)]
        assert carrier.role is ROLE[channel]
        assert text in str(carrier.content)
        assert setup.content_for_input(index) == text
        assert everything.count(text) == 1


def test_the_two_delivery_orders_of_a_cell_pair_differ_only_by_order() -> None:
    forward = _setup(Channel.README, Channel.USER)
    reverse_matchup = make_matchup(tuple(reversed(forward.matchup.inputs)), Assistant.INKLING)
    reverse = build_conversation(
        reverse_matchup, tuple(reversed(_generated(forward.matchup))), _library()
    )

    def texts(setup: ConversationSetup) -> list[str]:
        return sorted(str(message.content) for message in setup.messages if message.content)

    # Same messages with the same text; only their order changes with delivery order.
    assert texts(forward) == texts(reverse)
    assert [str(m.role) for m in forward.messages] != [str(m.role) for m in reverse.messages]


def test_unfilled_slots_leave_no_trace_in_the_shared_context() -> None:
    setup = _setup(Channel.SYSTEM, Channel.USER)

    readme = str(setup.messages[4].content)
    assert readme.startswith("# scaffold-tools")
    assert "Notes for agents" not in readme
    # The README is the same file whether or not an input was placed in it.
    assert setup.readme_contents() == (setup.messages[4].content,)
    with_readme = _setup(Channel.USER, Channel.README)
    assert "## Notes for agents\n\nAuthored SECOND text." in str(with_readme.readme_contents()[0])


def test_authored_text_is_inserted_verbatim_even_when_it_looks_like_a_template() -> None:
    matchup = _matchup(Channel.USER, Channel.README)
    texts = (
        "Use {{ braces }} and {% tags %} literally.\n\nSecond paragraph.",
        "Plain <b>text</b> & more.",
    )
    setup = build_conversation(matchup, _generated(matchup, texts), _library())

    assert setup.messages[2].content == texts[0]
    assert texts[1] in str(setup.messages[4].content)
    assert setup.content_for_input(0) == texts[0]


@pytest.mark.parametrize(
    ("assistant", "prefix"),
    [(Assistant.QWEN3_8_2_4T, "chatcmpl-tool-"), (Assistant.INKLING, "call_")],
)
def test_readme_read_uses_the_assistant_tool_call_format(assistant: Assistant, prefix: str) -> None:
    setup = _setup(Channel.USER, Channel.README, assistant=assistant)

    call = setup.messages[3].tool_calls[0]
    assert str(call.call_id).startswith(prefix)
    assert json.loads(call.arguments) == {"path": "README.md"}
    assert setup.messages[4].tool_call_id == call.call_id
    assert setup == _setup(Channel.USER, Channel.README, assistant=assistant)


def test_setup_lookups_check_bounds_in_both_layouts() -> None:
    scenario = _setup(Channel.SYSTEM, Channel.USER)
    matchup = _matchup(Channel.SYSTEM, Channel.USER)
    bare = construct_conversation(matchup, _generated(matchup))

    assert [bare.message_index_for_input(index) for index in range(2)] == [0, 1]
    for setup in (scenario, bare):
        with pytest.raises(IndexError):
            setup.content_for_input(2)
        with pytest.raises(IndexError):
            setup.message_index_for_input(-1)


def test_scenario_construction_rejects_messages_out_of_matchup_order() -> None:
    matchup = _matchup(Channel.SYSTEM, Channel.USER)
    scenario = _library().scenarios[_pair().pair_id]

    with pytest.raises(ValueError, match="matchup input order"):
        scenario.construct(matchup, tuple(reversed(_generated(matchup))))


def test_the_judge_sees_the_authored_text_and_the_whole_scenario() -> None:
    setup = _setup(Channel.USER, Channel.README)
    requests = judge_requests(ConversationTrace(setup, _chat("answer")))

    for request, text in zip(requests, TEXTS, strict=True):
        evidence = request["messages"][1]["content"]
        assert f"<content>{text}</content>" in evidence
        assert "coding agent" in evidence and "scaffold-tools" in evidence


# --------------------------------------------------------------------------
# The library: bare fallback and cache validity
# --------------------------------------------------------------------------


def test_pairs_without_a_scenario_keep_the_bare_conversation() -> None:
    matchup = _matchup(Channel.SYSTEM, Channel.USER, pair_id="prime-1234-bare-vs-table")
    generated = _generated(matchup)
    bare = construct_conversation(matchup, generated)

    assert _library().scenario_for(matchup) is None
    assert build_conversation(matchup, generated, _library()) == bare
    assert build_conversation(matchup, generated) == bare
    assert bare.placements is None
    require_scenario_selection(_library(), bare)


def test_cached_setups_match_only_the_same_scenario_selection() -> None:
    matchup = _matchup(Channel.README, Channel.USER)
    generated = _generated(matchup)
    scenario_setup = build_conversation(matchup, generated, _library())
    bare = construct_conversation(matchup, generated)

    assert scenario_setup.placements is not None
    require_scenario_selection(_library(), scenario_setup)
    require_scenario_selection(None, bare)
    with pytest.raises(ValueError, match="different scenario selection"):
        require_scenario_selection(_library(), bare)
    with pytest.raises(ValueError, match="different scenario selection"):
        require_scenario_selection(None, scenario_setup)

    raw = yaml.safe_load((SCENARIOS / f"{PAIR_ID}.yaml").read_text(encoding="utf-8"))
    raw["messages"][0]["content"] = "A different system prompt."
    edited = ScenarioLibrary(
        {_pair().pair_id: scenario_from_dict(_pair().pair_id, raw)}, _library().memberships
    )
    with pytest.raises(ValueError, match="different scenario selection"):
        require_scenario_selection(edited, scenario_setup)


def test_scenario_runs_need_both_sides_of_one_banked_pair() -> None:
    other = _pair("prime-1234-bare-vs-table")
    mixed = make_matchup(
        (
            PromptSpec(_pair().first, Framing.NORMAL, Channel.SYSTEM, Author.USER),
            PromptSpec(other.second, Framing.NORMAL, Channel.USER, Author.USER),
        ),
        Assistant.INKLING,
    )
    unbanked = make_matchup(
        (
            PromptSpec(
                Instruction.parse("Not in the bank."), Framing.NORMAL, Channel.SYSTEM, Author.USER
            ),
            PromptSpec(_pair().second, Framing.NORMAL, Channel.USER, Author.USER),
        ),
        Assistant.INKLING,
    )

    with pytest.raises(ValueError, match="two sides of one instruction pair"):
        _library().scenario_for(mixed)
    with pytest.raises(ValueError, match="pair bank"):
        build_conversation(unbanked, _generated(unbanked), _library())
    with pytest.raises(ValueError, match="different scenario selection"):
        require_scenario_selection(
            _library(), construct_conversation(unbanked, _generated(unbanked))
        )


def test_loading_rejects_missing_empty_and_misnamed_directories(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        load_scenarios(tmp_path / "absent", _pairs())
    # A mistyped directory or extension must not silently mean bare conversations.
    (tmp_path / f"{PAIR_ID}.yml").write_text("source: x\n")
    with pytest.raises(ValueError, match="holds no"):
        load_scenarios(tmp_path, _pairs())
    (tmp_path / "not-a-pair.yaml").write_text("source: x\nadaptation: y\nmessages: []\n")
    with pytest.raises(ValueError, match="does not name an instruction pair"):
        load_scenarios(tmp_path, _pairs())


# --------------------------------------------------------------------------
# Scenario file rules
# --------------------------------------------------------------------------

SYS, HELLO, PLAIN, OPEN, READ, DOC, LATE_SYSTEM, LATE_USER = range(8)


def _minimal() -> dict[str, Any]:
    return {
        "source": "test",
        "adaptation": "none",
        "messages": [
            {"role": "system", "content": "Context. {{ system.early }}"},
            {"role": "user", "content": "Hello."},
            {"role": "assistant", "content": "Hello, how can I help?"},
            {"role": "user", "content": "Opening. {{ user.early }}"},
            {"role": "assistant", "content": "Reading the README.", "reads": "README.md"},
            {"role": "tool", "content": "# Readme {{ tool.early }}{{ tool.late }}"},
            {"role": "system", "when": "system.late", "content": "{{ system.late }}"},
            {"role": "user", "when": "user.late", "content": "{{ user.late }}"},
        ],
    }


def _parse(raw: object) -> Scenario:
    return scenario_from_dict(PairId.parse(PAIR_ID), raw)


@pytest.mark.parametrize(
    ("channels", "message_index", "slot"),
    [
        ((Channel.SYSTEM, Channel.USER), SYS, "system.early"),
        ((Channel.USER, Channel.SYSTEM), LATE_SYSTEM, "system.late"),
        ((Channel.USER, Channel.USER), OPEN, "user.early"),
        ((Channel.USER, Channel.USER), LATE_USER, "user.late"),
        ((Channel.README, Channel.USER), DOC, "tool.early"),
        ((Channel.USER, Channel.README), DOC, "tool.late"),
    ],
)
@pytest.mark.parametrize(
    "duplicate",
    [
        "{% if SLOT | length < 50 %}{{ SLOT }}{% endif %}",
        "{{ SLOT | replace('\\n', ' ') }}",
    ],
    ids=["short-text-branch", "newline-normalized-copy"],
)
def test_actual_authored_text_must_be_inserted_once(
    channels: tuple[Channel, Channel], message_index: int, slot: str, duplicate: str
) -> None:
    raw = _minimal()
    raw["messages"][message_index]["content"] += " " + duplicate.replace("SLOT", slot)
    # The longer, two-paragraph load probes do not expose either repeated copy.
    scenario = _parse(raw)
    matchup = _matchup(*channels)

    with pytest.raises(ValueError, match="exactly once"):
        scenario.construct(matchup, _generated(matchup, ("Do X.", "Do Y.")))


def test_a_scenario_can_hold_plain_assistant_turns_and_survives_the_trace_cache() -> None:
    scenario = _parse(_minimal())
    matchup = _matchup(Channel.USER, Channel.README)
    setup = scenario.construct(matchup, _generated(matchup))
    trace = ConversationTrace(setup, _chat("answer"))

    assert [str(message.role) for message in setup.messages] == [
        "system", "user", "assistant", "user", "assistant", "tool",
    ]
    assert setup.messages[2].openrouter_dict() == {
        "role": "assistant",
        "content": "Hello, how can I help?",
    }
    assert setup.messages[4].content == "Reading the README."
    stored = trace_to_dict(trace)
    assert stored["placements"] == [
        {"message": 3, "content": TEXTS[0]},
        {"message": 5, "content": TEXTS[1]},
    ]
    assert trace_from_dict(json.loads(json.dumps(stored))) == trace
    assert trace_from_dict(stored, matchup).setup.placements == setup.placements

    bare = ConversationTrace(construct_conversation(matchup, _generated(matchup)), _chat("answer"))
    assert "placements" not in trace_to_dict(bare)
    assert trace_fingerprint(bare) != trace_fingerprint(trace)


def test_a_readme_read_cannot_move_between_tool_slots() -> None:
    raw = _minimal()
    raw["messages"][READ]["when"] = "not tool.late"
    raw["messages"].extend([
        {**raw["messages"][READ], "when": "tool.late"},
        dict(raw["messages"][DOC]),
    ])

    with pytest.raises(ValueError, match="changes shared context with tool.late"):
        _parse(raw)


@pytest.mark.parametrize(
    "condition",
    [
        "system.early", "not system.early", "system.late", "not system.late",
        "user.early", "not user.early", "user.late", "not user.late",
        "tool.early", "not tool.early", "tool.late", "not tool.late",
    ],
)
def test_readme_read_is_independent_of_filled_slots(condition: str) -> None:
    raw = yaml.safe_load((SCENARIOS / f"{PAIR_ID}.yaml").read_text(encoding="utf-8"))
    read = next(message for message in raw["messages"] if "reads" in message)
    read["when"] = condition

    with pytest.raises(ValueError, match="changes shared context|must render tool"):
        _parse(raw)


def test_readme_context_is_checked_again_for_actual_authored_text() -> None:
    raw = _minimal()
    raw["messages"][READ]["when"] = "system.late != 'hide the README'"
    scenario = _parse(raw)
    matchup = _matchup(Channel.USER, Channel.SYSTEM)

    with pytest.raises(ValueError, match="changes shared context with system.late"):
        scenario.construct(matchup, _generated(matchup, (TEXTS[0], "hide the README")))


@pytest.mark.parametrize(
    ("placements", "error"),
    [
        ("bad", "must be a list"),
        ([1], "invalid fields"),
        ([{"message": True, "content": "x"}, {"message": 3, "content": "y"}], "invalid fields"),
        ([{"message": 1, "content": "x", "extra": 0}], "invalid fields"),
    ],
)
def test_trace_cache_rejects_malformed_placements(placements: object, error: str) -> None:
    matchup = _matchup(Channel.USER, Channel.README)
    setup = _parse(_minimal()).construct(matchup, _generated(matchup))
    stored = trace_to_dict(ConversationTrace(setup, _chat("answer")))
    stored["placements"] = placements

    with pytest.raises(ValueError, match=error):
        trace_from_dict(stored)


def _change(index: int, **fields: object) -> dict[str, Any]:
    raw = _minimal()
    raw["messages"][index] = {**raw["messages"][index], **fields}
    return raw


def _without(index: int) -> dict[str, Any]:
    raw = _minimal()
    del raw["messages"][index]
    return raw


def _insert(index: int, *messages: dict[str, object]) -> dict[str, Any]:
    raw = _minimal()
    raw["messages"][index:index] = list(messages)
    return raw


def _reordered(*indices: int) -> dict[str, Any]:
    raw = _minimal()
    raw["messages"] = [raw["messages"][index] for index in indices]
    return raw


@pytest.mark.parametrize(
    ("raw", "error"),
    [
        ("bad", "scenario fields"),
        ({**_minimal(), "extra": 1}, "scenario fields"),
        ({**_minimal(), "source": 3}, "must be text"),
        ({**_minimal(), "messages": "bad"}, "must be a list"),
        ({**_minimal(), "source": "  "}, "source and adaptation"),
        ({**_minimal(), "messages": []}, "must contain messages"),
        ({**_minimal(), "messages": ["bad"]}, "needs a role"),
        (_change(SYS, extra=1), "needs a role"),
        (_change(SYS, content=3), "content must be text"),
        (_change(LATE_SYSTEM, when=3), "must be a Jinja expression"),
        (_change(READ, reads="notes.txt"), "can only read README.md"),
        (_change(SYS, reads="README.md"), "only text"),
        (_change(PLAIN, content=None), "need text or a README read"),
        (_without(DOC), "followed by its result"),
        (_insert(1, {"role": "tool", "content": "orphan"}), "unconditional text result"),
        (_change(DOC, when="tool.late"), "unconditional text result"),
        (_change(SYS, content="{% if %}"), "invalid template"),
        (_change(LATE_SYSTEM, when="system.late +"), "invalid template"),
        (_change(LATE_SYSTEM, when="sytem.late"), "condition failed to evaluate"),
        (_change(SYS, content="Context. {{ missing }} {{ system.early }}"), "failed to render"),
        (_without(LATE_USER), "user.late in exactly one message, found 0"),
        (_change(OPEN, content="Opening. {{ user.early }} {{ system.early }}"), "found 2"),
        (_change(SYS, content="Context."), "system.early in exactly one message, found 0"),
        (_change(SYS, role="user"), "system.early in a user message"),
        (_change(SYS, content="Context. {{ system.early | upper }}"), "system.early verbatim"),
        (
            _change(OPEN, content="Opening.\n> {{ user.early | indent(2) }}"),
            "user.early verbatim",
        ),
        (
            _reordered(SYS, HELLO, PLAIN, LATE_USER, OPEN, READ, DOC, LATE_SYSTEM),
            "early slots before late",
        ),
        (
            _insert(
                LATE_SYSTEM,
                {"role": "assistant", "reads": "README.md"},
                {"role": "tool", "content": "Again."},
            ),
            "more than once",
        ),
        (
            _change(OPEN, content="Opening. {{ user.early }} Again: {{ user.early }}"),
            "exactly once",
        ),
        (
            _insert(
                LATE_SYSTEM,
                {"role": "user", "when": "system.late", "content": "A reminder is coming."},
            ),
            "changes shared context with system.late",
        ),
        (_insert(READ, {"role": "user", "content": "{% if tool.early %}x{% endif %}"}), "empty user"),
        (
            _change(PLAIN, content="{% if tool.early %}Hello.{% endif %}"),
            "empty assistant message",
        ),
        (
            _insert(LATE_USER + 1, {"role": "assistant", "content": "Done reading."}),
            "must not end on an assistant turn",
        ),
        (_change(SYS, role="narrator"), "is not a valid ChatRole"),
    ],
)
def test_scenario_files_are_validated_when_loaded(raw: object, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        _parse(copy.deepcopy(raw))


# --------------------------------------------------------------------------
# Conversation invariants
# --------------------------------------------------------------------------


def test_scenario_setups_validate_placements_and_tool_structure() -> None:
    matchup = _matchup(Channel.USER, Channel.README)
    messages = _setup(Channel.USER, Channel.README).messages
    first, second = (GeneratedText.parse(text) for text in TEXTS)
    good = (Placement(2, first), Placement(4, second))
    call = ToolCall(ToolCallId.parse("call-1"), ToolName.parse("bash"), '{"command":"ls"}')
    broken_json = ToolCall(ToolCallId.parse("call-1"), ToolName.parse("read_file"), "{")
    orphan = ChatMessage(ChatRole.TOOL, second, tool_call_id=ToolCallId.parse("call-9"))
    read, result = messages[3], messages[4]

    assert ConversationSetup(matchup, messages, good).placements == good
    for placements, error in (
        ((Placement(2, first),), "one placement per matchup input"),
        ((Placement(2, first), Placement(2, second)), "follow matchup order"),
        ((Placement(2, first), Placement(9, second)), "follow matchup order"),
        ((Placement(0, first), Placement(4, second)), "role does not match"),
        ((Placement(2, second), Placement(4, second)), "verbatim"),
    ):
        with pytest.raises(ValueError, match=error):
            ConversationSetup(matchup, messages, placements)
    for changed, error in (
        (
            (*messages[:3], ChatMessage(ChatRole.ASSISTANT, tool_calls=(call,)), result),
            "one README.md read",
        ),
        (
            (*messages[:3], ChatMessage(ChatRole.ASSISTANT, tool_calls=(broken_json,)), result),
            "one README.md read",
        ),
        ((*messages[:3], read, orphan), "one README.md read"),
        (
            (
                *messages[:3],
                ChatMessage(ChatRole.ASSISTANT, tool_calls=read.tool_calls * 2),
                result,
            ),
            "one README.md read",
        ),
        ((*messages[:3], messages[1], result), "must follow their README.md read"),
        ((*messages, read), "one README.md read"),
        ((*messages, read, result), "at most once"),
        ((*messages, ChatMessage(ChatRole.ASSISTANT, first)), "must not end on an assistant turn"),
    ):
        with pytest.raises(ValueError, match=error):
            ConversationSetup(matchup, changed, good)


def test_role_probes_refuse_scenario_conversations() -> None:
    matchup = _matchup(Channel.USER, Channel.README)
    bare = construct_conversation(matchup, _generated(matchup))

    assert [_target_message_index(bare, position) for position in (1, 2)] == [0, 2]
    with pytest.raises(ValueError, match="does not support scenario conversations"):
        _target_message_index(_setup(Channel.USER, Channel.README), 1)


# --------------------------------------------------------------------------
# Collection wiring
# --------------------------------------------------------------------------


class FakeTransport:
    def __init__(self, posts: list[JsonObject]) -> None:
        self.posts = posts
        self.post_calls: list[tuple[str, JsonObject]] = []

    def post_json(self, path: str, body: JsonObject) -> JsonObject:
        self.post_calls.append((path, body))
        return self.posts.pop(0)

    def get_json(self, path: str) -> JsonObject:
        raise AssertionError(f"unexpected GET {path}")


def _batch(batch_id: str, bodies: list[str]) -> JsonObject:
    return {
        "id": batch_id,
        "status": "completed",
        "results": [
            {
                "custom_id": f"request-{index}",
                "response": {"status_code": 200, "body": _chat(body, f"{batch_id}-{index}")},
                "error": None,
            }
            for index, body in enumerate(bodies)
        ],
    }


QA = json.dumps({"complies": True, "issues": []})
VERDICTS = [json.dumps({"completed": value}) for value in (True, False, False, True)]


def _collection_posts() -> list[JsonObject]:
    return [
        _batch("message-qa", [QA, QA]),
        _chat("answer 0", "assistant-0"),
        _chat("answer 1", "assistant-1"),
        _batch("judge", VERDICTS),
    ]


def _study() -> Study:
    pair = _pair()
    return make_study(
        (
            PromptSpec(pair.first, Framing.NORMAL, Channel.README, Author.USER),
            PromptSpec(pair.second, Framing.NORMAL, Channel.USER, Author.USER),
        ),
        Assistant.INKLING,
        1,
    )


def _manual_library(tmp_path: Path) -> ManualMessageLibrary:
    root = tmp_path / "manual"
    for index, instruction in enumerate(_pair().instructions):
        directory = root / f"instruction-{index}"
        directory.mkdir(parents=True)
        (directory / "instruction.txt").write_text(str(instruction))
        for framing in author_framings(Author.USER):
            (directory / f"{framing}.txt").write_text(str(instruction))
    return ManualMessageLibrary(root)


def test_collection_runs_inside_the_scenario_and_never_collects_over_another_selection(
    tmp_path: Path,
) -> None:
    study = _study()
    manual = _manual_library(tmp_path)
    output = tmp_path / "collection"
    routing = CollectionRouting(RoutePreference.BATCH, True)
    transport = FakeTransport(_collection_posts())

    cold = collect_study(
        study, output, OpenRouterClient(transport), manual,
        routing=routing, prefer_batch=True, scenarios=_library(),
    )
    warm = collect_study(
        study, output, None, manual, routing=routing, prefer_batch=True, scenarios=_library()
    )

    assert cold.trace_cache_hits == 0
    assert warm.trace_cache_hits == 2
    assert warm.observations == cold.observations
    first, second = str(_pair().first), str(_pair().second)
    sent = [call[1]["messages"] for call in transport.post_calls[1:3]]
    # Both permutations carry the same scenario context around the inputs.
    assert all("coding agent" in messages[0]["content"] for messages in sent)
    assert [len(messages) for messages in sent] == [5, 5]
    assert first in sent[0][3]["content"] and sent[0][4] == {"role": "user", "content": second}
    assert sent[1][2] == {"role": "user", "content": second} and first in sent[1][4]["content"]
    # The position recorded for each observation is still the matchup order.
    assert [int(row.position) for row in cold.observations] == [1, 2, 1, 2]

    # A bare run on the same output directory must refuse, not re-collect over the traces.
    spare = FakeTransport(_collection_posts())
    with pytest.raises(ValueError, match="different scenario selection"):
        collect_study(
            study, output, OpenRouterClient(spare), manual, routing=routing, prefer_batch=True
        )
    assert spare.post_calls == []


def test_scenario_runs_reject_unplaceable_studies_before_any_request(tmp_path: Path) -> None:
    study = make_study(
        (
            PromptSpec(
                Instruction.parse("Not in the bank."), Framing.NORMAL, Channel.SYSTEM, Author.USER
            ),
            PromptSpec(_pair().second, Framing.NORMAL, Channel.USER, Author.USER),
        ),
        Assistant.INKLING,
        1,
    )
    transport = FakeTransport(_collection_posts())

    with pytest.raises(ValueError, match="pair bank"):
        collect_study(
            study, tmp_path / "out", OpenRouterClient(transport), _manual_library(tmp_path),
            routing=CollectionRouting(RoutePreference.BATCH, True), prefer_batch=True,
            scenarios=_library(),
        )
    assert transport.post_calls == []


def test_inline_probe_diagnostics_are_rejected_with_scenarios(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="do not support scenario conversations"):
        collect_study(
            _study(), tmp_path / "out", None, _manual_library(tmp_path),
            prefer_batch=True, probe_mode=ProbeQaMode.INLINE, scenarios=_library(),
        )


@pytest.mark.parametrize(
    ("command", "module"),
    [(collect_data, "reasonese.collect_data"), (collect_studies, "reasonese.collect_studies")],
)
def test_collection_commands_accept_a_scenario_directory(
    command: Any,
    module: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    study_path = tmp_path / "study.yaml"
    study_path.write_text(yaml.safe_dump(study_to_dict(_study()), sort_keys=False))
    manual = _manual_library(tmp_path)
    transport = FakeTransport(_collection_posts())
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(f"{module}.RequestsTransport", lambda key: transport)
    args = [
        "--allow-paid", "--route", "paid",
        "--study", str(study_path),
        "--output", str(tmp_path / "output"),
        "--user-messages", str(manual.root),
    ]

    assert command([*args, "--scenarios", str(SCENARIOS), "--pairs", str(BANK)]) == 0
    capsys.readouterr()

    assert len(transport.post_calls) == 4
    assert "coding agent" in transport.post_calls[1][1]["messages"][0]["content"]
    with pytest.raises(SystemExit):
        command(args)
    assert "different scenario selection" in capsys.readouterr().err


def test_run_conversation_cli_selects_scenarios_and_refuses_another_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    matchup = _matchup(Channel.SYSTEM, Channel.USER)
    matchup_path = tmp_path / "matchup.yaml"
    matchup_path.write_text(yaml.safe_dump(matchup_to_dict(matchup)), encoding="utf-8")
    manual = _manual_library(tmp_path)
    transport = FakeTransport([_chat(QA, "qa-0"), _chat(QA, "qa-1"), _chat("answer", "live-id")])
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("reasonese.run_conversation.RequestsTransport", lambda key: transport)
    args = [
        "--allow-paid", "--route", "paid", "--no-batch",
        "--matchup", str(matchup_path),
        "--message-cache", str(tmp_path / "messages.yaml"),
        "--message-qa-cache", str(tmp_path / "message-qa.yaml"),
        "--trace-cache", str(tmp_path / "traces.yaml"),
        "--user-messages", str(manual.root),
    ]
    scenario_args = [*args, "--scenarios", str(SCENARIOS), "--pairs", str(BANK)]

    assert run_conversation(scenario_args) == 0
    cold = json.loads(capsys.readouterr().out)
    assert run_conversation(scenario_args) == 0
    warm = json.loads(capsys.readouterr().out)

    assert cold["cache_hit"] is False
    assert warm["cache_hit"] is True
    assert warm["messages"] == 6
    assert [message["role"] for message in transport.post_calls[2][1]["messages"]] == [
        "system", "system", "user", "assistant", "tool", "user",
    ]
    # Without the scenario directory the cached scenario trace is refused, not replaced.
    with pytest.raises(SystemExit):
        run_conversation(args)
    assert "different scenario selection" in capsys.readouterr().err
    assert len(transport.post_calls) == 3


def test_show_scenario_prints_the_rendered_conversation(capsys: pytest.CaptureFixture[str]) -> None:
    base = ["--scenarios", str(SCENARIOS), "--pairs", str(BANK)]

    assert show_scenario(
        [*base, "--pair", PAIR_ID, "--first-channel", "user message", "--second-channel", "README.md"]
    ) == 0
    shown = json.loads(capsys.readouterr().out)

    assert shown["pair"] == PAIR_ID
    assert shown["source"].startswith("Synthetic example")
    assert [message["role"] for message in shown["messages"]] == [
        "system", "user", "user", "assistant", "tool",
    ]
    assert shown["placements"] == [2, 4]
    assert str(_pair().second) in shown["messages"][4]["content"]
    for pair in ("no-such-pair", "prime-1234-bare-vs-table"):
        with pytest.raises(SystemExit):
            show_scenario([*base, "--pair", pair])
    assert "has no scenario" in capsys.readouterr().err
