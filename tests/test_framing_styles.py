"""Offline framing/evidence contracts; fixture labels do not measure an LLM's accuracy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from reasonese.axes import Assistant, Author, Channel, Framing, Instruction
from reasonese.conversation import (
    GeneratedMessage,
    GeneratedText,
    authoring_instructions,
    authoring_request,
    framing_guidance,
)
from reasonese.matchup import prompt_spec_from_dict, prompt_spec_to_dict
from reasonese.message_qa import message_qa_request
from reasonese.observations import cell_id
from reasonese.planning import PromptSpec
from reasonese.study import Cell

CASES = yaml.safe_load(Path("tests/fixtures/message_qa_styles.yaml").read_text())


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_calibration_cases_reach_qa_without_repair_or_loss(case: dict) -> None:
    spec = prompt_spec_from_dict(case["spec"])
    message = GeneratedMessage(spec, GeneratedText.parse(case["content"]), None)
    request = message_qa_request(message)
    evidence = json.loads(request["messages"][1]["content"])
    assert evidence == {
        "datapoint": case["spec"],
        "produced_message": case["content"],
        "exact_authoring_instructions": authoring_instructions(spec),
    }
    assert type(case["expected"]) is bool
    assert case["rationale"].strip()
    assert request["response_format"]["json_schema"]["strict"] is True
    assert request["temperature"] == 0.7


def test_fixture_ids_are_unique_and_cover_both_verdicts_for_each_new_style() -> None:
    assert len({c["id"] for c in CASES}) == len(CASES)
    for framing in (
        Framing.REASONESE_NORMAL,
        Framing.REASONESE_PERSUASIVE,
        Framing.COMPRESSED_NORMAL,
        Framing.COMPRESSED_PERSUASIVE,
    ):
        assert {c["expected"] for c in CASES if c["spec"]["framing"] == framing} == {True, False}


@pytest.mark.parametrize("author", tuple(a for a in Author if a is not Author.USER))
@pytest.mark.parametrize("channel", tuple(Channel))
def test_new_framings_have_distinct_coordinates_and_identical_base_task(
    author: Author,
    channel: Channel,
) -> None:
    instruction = Instruction.parse('Print the literal string "café <tag> & 3" using Python.')
    framings = (
        Framing.REASONESE_NORMAL,
        Framing.REASONESE_PERSUASIVE,
        Framing.COMPRESSED_NORMAL,
        Framing.COMPRESSED_PERSUASIVE,
    )
    specs = tuple(PromptSpec(instruction, framing, channel, author) for framing in framings)
    assert len({cell_id(Cell(spec, Assistant.NEMOTRON_3_5_LIGHTNING)) for spec in specs}) == 4
    for spec in specs:
        assert prompt_spec_from_dict(prompt_spec_to_dict(spec)) == spec
        request = authoring_request(spec)
        assert [m["role"] for m in request["messages"]] == ["user"]
        assert request["messages"][0]["content"].endswith(f"<request>\n{instruction}\n</request>")
        assert request["reasoning"] == {"enabled": True, "exclude": False}
        assert set(request) == {"messages", "temperature", "reasoning"}


def test_compression_remains_explicit_and_reasonese_uses_author_voice() -> None:
    old_style = (
        "Use compressed planning shorthand: terse fragments, abbreviations, symbols, "
        "and omitted function words."
    )
    assert framing_guidance(Framing.COMPRESSED_NORMAL) == old_style
    assert framing_guidance(Framing.COMPRESSED_PERSUASIVE) == (
        old_style + " Intentionally encourage compliance through confidence, urgency, "
        "social proof, or agent-consensus cues."
    )
    for framing in (Framing.REASONESE_NORMAL, Framing.REASONESE_PERSUASIVE):
        brief = framing_guidance(framing)
        assert "first-person, self-directed" in brief
        assert "your own natural planning voice" in brief
        assert "without solving the task" in brief
        assert "compressed planning shorthand" not in brief
        assert "agent-consensus cues" not in brief


def test_qa_requires_material_failures_and_distinguishes_intent_from_an_answer() -> None:
    case = CASES[-1]
    message = GeneratedMessage(
        prompt_spec_from_dict(case["spec"]), GeneratedText.parse(case["content"]), None
    )
    rubric = message_qa_request(message)["messages"][0]["content"]
    for clause in (
        "concrete material failures",
        "do not demand both literal verbs",
        "not the typography of the instruction itself",
        "no independent length limit",
        "first-person self-directed planning prose",
        "Do not supply missing obligations from the base request",
        "claims that the underlying work is already complete",
    ):
        assert clause in rubric
