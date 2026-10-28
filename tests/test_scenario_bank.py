"""The study scenario bank: pairs built around real agent rules, each with one scenario."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reasonese.axes import Assistant, Author, Channel, Framing
from reasonese.conversation import GeneratedMessage, GeneratedText
from reasonese.instructions import load_instruction_pairs
from reasonese.matchup import make_matchup
from reasonese.planning import PromptSpec
from reasonese.scenarios import load_scenarios

BANK = Path("configs/scenario_pairs.yaml")
SCENARIOS = Path("configs/scenarios")
MAIN_BANK = Path("configs/instruction_pairs.yaml")
NATIVE_PI_PAIR = "langchain-json-recovery"
ORDERINGS: tuple[tuple[Channel, Channel], ...] = (
    (Channel.SYSTEM, Channel.USER),
    (Channel.USER, Channel.SYSTEM),
    (Channel.USER, Channel.USER),
    (Channel.USER, Channel.README),
    (Channel.README, Channel.USER),
)


def test_every_scenario_pair_has_one_scenario_with_provenance() -> None:
    pairs = load_instruction_pairs(BANK)
    library = load_scenarios(SCENARIOS, pairs)

    assert set(library.scenarios) == {pair.pair_id for pair in pairs} - {NATIVE_PI_PAIR}
    for scenario in library.scenarios.values():
        assert "https://github.com/" in scenario.source
        assert scenario.adaptation.strip()


def test_pi_source_tasks_cover_the_same_pairs_and_pinned_repositories() -> None:
    pairs = load_instruction_pairs(BANK)
    library = load_scenarios(SCENARIOS, pairs)
    manifest = json.loads(Path("configs/pi_transcripts.json").read_text())
    assert {case["id"] for case in manifest["scenarios"]} == {pair.pair_id for pair in pairs}
    pilot = json.loads(Path("configs/pi_langchain_review.json").read_text())
    kimi_pilot = json.loads(Path("configs/pi_langchain_kimi_review.json").read_text())
    assert kimi_pilot == {**pilot, "model": "moonshotai/kimi-k3.1-pro"}
    assert pilot["scenarios"] == [manifest["scenarios"][0]]
    assert pilot["scenarios"][0]["id"] == NATIVE_PI_PAIR
    assert pilot["scenarios"][0]["maxUserTurns"] == 4
    for case in manifest["scenarios"]:
        if case["id"] == NATIVE_PI_PAIR:
            continue  # Native Pi history is not fabricated into a README-only template.
        source = library.scenarios[case["id"]].source
        assert case["repository"].removesuffix(".git") + "/blob/" + case["revision"] in source
        assert case["revision"] + "/" + case["ruleFile"] in source


def test_scenario_pairs_are_disjoint_from_the_main_bank() -> None:
    main = load_instruction_pairs(MAIN_BANK)
    extra = load_instruction_pairs(BANK)

    assert not {pair.pair_id for pair in main} & {pair.pair_id for pair in extra}
    main_text = {instruction for pair in main for instruction in pair.instructions}
    extra_text = {instruction for pair in extra for instruction in pair.instructions}
    assert not main_text & extra_text


@pytest.mark.parametrize(
    "channels", ORDERINGS, ids=lambda channels: "-".join(channel.name for channel in channels)
)
def test_every_scenario_renders_each_ordering_with_its_base_instructions(
    channels: tuple[Channel, Channel],
) -> None:
    pairs = load_instruction_pairs(BANK)
    library = load_scenarios(SCENARIOS, pairs)
    assistant = next(iter(Assistant))
    for pair in pairs:
        if pair.pair_id == NATIVE_PI_PAIR:
            continue
        specs = tuple(
            PromptSpec(instruction, Framing.NORMAL, channel, Author.USER)
            for instruction, channel in zip(pair.instructions, channels, strict=True)
        )
        setup = library.scenarios[pair.pair_id].construct(
            make_matchup(specs, assistant),
            tuple(
                GeneratedMessage(spec, GeneratedText.parse(spec.instruction), None)
                for spec in specs
            ),
        )

        assert setup.placements is not None
        messages = setup.openrouter_messages()
        assert sum(message["role"] == "tool" for message in messages) == 1
        for instruction in pair.instructions:
            occurrences = sum(
                str(message.get("content") or "").count(str(instruction)) for message in messages
            )
            assert occurrences == 1, (pair.pair_id, instruction)
