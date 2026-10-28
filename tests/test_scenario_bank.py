"""Native Pi task coverage and provenance; no invented conversation history."""

from __future__ import annotations

import json
from pathlib import Path

from reasonese.instructions import load_instruction_pairs

BANK = Path("configs/scenario_pairs.yaml")
MAIN_BANK = Path("configs/instruction_pairs.yaml")


def test_pi_tasks_cover_all_pairs_and_preserve_the_reviewed_first_pilot() -> None:
    pairs = load_instruction_pairs(BANK)
    manifest = json.loads(Path("configs/pi_transcripts.json").read_text())
    pilot = json.loads(Path("configs/pi_langchain_review.json").read_text())
    kimi = json.loads(Path("configs/pi_langchain_kimi_review.json").read_text())
    remaining = json.loads(Path("configs/pi_remaining_review.json").read_text())
    assert len(pairs) == len(manifest["scenarios"]) == 12
    assert {case["id"] for case in manifest["scenarios"]} == {pair.pair_id for pair in pairs}
    assert pilot["scenarios"] == [manifest["scenarios"][0]]
    assert kimi == {**pilot, "model": "moonshotai/kimi-k3.1-pro"}
    assert remaining["model"] == kimi["model"]
    assert remaining["scenarios"] == manifest["scenarios"][1:]
    assert not list(Path("configs/scenarios").glob("*.yaml"))


def test_remaining_tasks_have_pinned_sources_and_explicit_reviewable_conflicts() -> None:
    manifest = json.loads(Path("configs/pi_remaining_review.json").read_text())
    replaced = set()
    for case in manifest["scenarios"]:
        assert case["repository"].startswith("https://github.com/")
        assert len(case["revision"]) == 40
        assert set(case["revision"]) <= set("0123456789abcdef")
        assert case["sourceFiles"]
        assert all(not Path(path).is_absolute() and ".." not in Path(path).parts
                   for path in [case["ruleFile"], *case["sourceFiles"]])
        assert case["maxUserTurns"] == 4
        first, second = case["conflictInstructions"]
        assert (first["side"], second["side"]) == ("first", "second")
        assert (first["intendedFraming"], second["intendedFraming"]) == ("casual", "persuasive")
        assert first["channel"] == second["channel"] == "user message"
        assert first["author"] == second["author"] == "Codex operator"
        assert first["text"] != second["text"]
        assert case["sharedImplementationRequest"] and case["decisiveCase"]
        assert len(case["reviewChecks"]) >= 4
        replaced.add(case["replaces"])
    assert len(replaced) == 11


def test_scenario_pairs_are_disjoint_from_the_main_bank() -> None:
    main = load_instruction_pairs(MAIN_BANK)
    extra = load_instruction_pairs(BANK)
    assert not {pair.pair_id for pair in main} & {pair.pair_id for pair in extra}
    main_text = {instruction for pair in main for instruction in pair.instructions}
    extra_text = {instruction for pair in extra for instruction in pair.instructions}
    assert not main_text & extra_text
