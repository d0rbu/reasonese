"""Native Pi task coverage and provenance; no invented conversation history."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from reasonese.instructions import load_instruction_pairs

BANK = Path("configs/scenario_pairs.yaml")
MAIN_BANK = Path("configs/instruction_pairs.yaml")


def test_base_tasks_cover_all_pairs_and_preserve_historical_conflict_manifests() -> None:
    pairs = load_instruction_pairs(BANK)
    manifest = json.loads(Path("configs/pi_transcripts.json").read_text())
    pilot = json.loads(Path("configs/pi_langchain_review.json").read_text())
    kimi = json.loads(Path("configs/pi_langchain_kimi_review.json").read_text())
    remaining = json.loads(Path("configs/pi_remaining_review.json").read_text())
    assert len(pairs) == len(manifest["scenarios"]) == 12
    assert {case["id"] for case in manifest["scenarios"]} == {pair.pair_id for pair in pairs}
    first_base = json.loads(Path("configs/pi_langchain_base.json").read_text())
    assert first_base["scenarios"] == [manifest["scenarios"][0]]
    assert first_base["model"] == kimi["model"]
    assert kimi == {**pilot, "model": "moonshotai/kimi-k3.1-pro"}
    for name, expected in {
        "pi_langchain_review.json": "c9d141f132714f4b0d3a671c8dad1b0d900cb68e496744616fa02e42528c15f9",
        "pi_langchain_kimi_review.json": "926f2a90f77a2f0f6e9719cbcbe0f052dae791111f4c4120c62354c1d18c3c5d",
    }.items():
        assert hashlib.sha256((Path("configs") / name).read_bytes()).hexdigest() == expected
    assert remaining["model"] == kimi["model"]
    assert remaining["scenarios"] == manifest["scenarios"][1:]
    assert not list(Path("configs/scenarios").glob("*.yaml"))


def test_base_tasks_use_only_the_first_instruction_without_framing_labels() -> None:
    manifest = json.loads(Path("configs/pi_transcripts.json").read_text())
    assert manifest["collectionPurpose"] == "base_transcript"
    pairs = {pair.pair_id: pair for pair in load_instruction_pairs(BANK)}
    replaced = set()
    for case in manifest["scenarios"]:
        assert case["repository"].startswith("https://github.com/")
        assert len(case["revision"]) == 40
        assert set(case["revision"]) <= set("0123456789abcdef")
        assert case["sourceFiles"]
        assert all(not Path(path).is_absolute() and ".." not in Path(path).parts
                   for path in [case["ruleFile"], *case["sourceFiles"]])
        assert case["maxUserTurns"] == 4
        assert case["baseInstruction"] == str(pairs[case["id"]].instructions[0])
        assert case["baseInstruction"] != str(pairs[case["id"]].instructions[1])
        assert case["baseOutcome"]
        for field in ("conflictInstructions", "intendedFraming", "decisiveCase"):
            assert field not in case
        assert len(case["reviewChecks"]) >= 4
        if "replaces" in case:
            replaced.add(case["replaces"])
    assert len(replaced) == 11


def test_scenario_pairs_are_disjoint_from_the_main_bank() -> None:
    main = load_instruction_pairs(MAIN_BANK)
    extra = load_instruction_pairs(BANK)
    assert not {pair.pair_id for pair in main} & {pair.pair_id for pair in extra}
    main_text = {instruction for pair in main for instruction in pair.instructions}
    extra_text = {instruction for pair in extra for instruction in pair.instructions}
    assert not main_text & extra_text
