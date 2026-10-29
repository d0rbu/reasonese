from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from reasonese.axes import Assistant, Author, Channel, Framing
from reasonese.instructions import load_instruction_pairs
from reasonese.matchup import make_matchup
from reasonese.native_templates import (
    NativeSource,
    TurnVariant,
    compile_matchup,
    digest,
    encoded,
    load_variant,
    main,
    pilot_studies,
    replace_user_turn,
    turn_main,
    validate_history,
    write_plan,
)
from reasonese.planning import PromptSpec
from reasonese.study import make_study, study_to_dict


@pytest.fixture
def source(tmp_path: Path) -> NativeSource:
    root = tmp_path / "corpus" / "langchain-json-recovery"
    root.mkdir(parents=True)
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Investigate JSON."}], "timestamp": 1},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "Inspect existing code."},
            {"type": "toolCall", "id": "read-1", "name": "read", "arguments": {"path": "parser.py"}}]},
        {"role": "toolResult", "toolCallId": "read-1", "toolName": "read", "content": [
            {"type": "text", "text": "\n# file with whitespace\n\n"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Investigation complete."}], "stopReason": "stop"},
        {"role": "user", "content": [{"type": "text", "text": "Source A implementation."}], "timestamp": 2},
        {"role": "assistant", "content": [{"type": "text", "text": "Implementation complete."}], "stopReason": "stop"},
    ]
    for message in messages:
        if message["role"] == "assistant":
            message.update(model="moonshotai/kimi-k3.1-pro", provider="openrouter", api="openai-completions")
            message.setdefault("stopReason", "toolUse")
    def ref(name: str, data: bytes) -> dict[str, str]:
        (root / name).write_bytes(data)
        return {"path": name, "sha256": digest(data)}
    descriptor = {"kind": "pi_source_template", "scenario": "langchain-json-recovery",
        "manifest": {"model": "moonshotai/kimi-k3.1-pro", "thinking": "high"},
        "messages": ref("messages.json", encoded(messages)),
        "system": ref("system.txt", b"You are a coding assistant.\n"),
        "tools": ref("tools.json", encoded([{"name": "read"}])),
        "checkpoints": {str(i): {"workspace": ref(f"{i}.tar.gz", f"checkpoint{i}".encode())}
                        for i in range(3)}}
    (root / "template.json").write_bytes(encoded(descriptor))
    return NativeSource.load(root / "template.json")


@pytest.fixture
def bank(tmp_path: Path, source: NativeSource) -> Path:
    root = tmp_path / "variants"
    pair = load_instruction_pairs(Path("configs/scenario_pairs.yaml"))[0]
    for slot in ("investigation", "implementation/a", "implementation/b"):
        for author in ("human", "kimi"):
            path = root / source.descriptor["scenario"] / slot / author
            path.mkdir(parents=True)
            (path.parent / "canonical-instruction.txt").write_text(str(pair.second if slot.endswith("/b") else pair.first))
            (path.parent / "authoring-source.txt").write_text("Complete assignment.")
            for framing in ("normal", "casual", "persuasive"):
                value = f"TODO: write {slot} {framing}" if author == "human" else f"{slot} {framing} literal {{{{ x }}}}"
                (path / f"{framing}.txt").write_text(value + "\n")
    slots = {}
    for slot in ("investigation", "implementation/a", "implementation/b"):
        base = root / source.descriptor["scenario"] / slot
        (base / "source-provenance.json").write_bytes(encoded({"authoringSourceSha256": digest(b"Complete assignment.")}))
        files = {"authoringSource": base / "authoring-source.txt",
                 "sourceProvenance": base / "source-provenance.json",
                 "instructionSource": base / ("authoring-source.txt" if slot == "investigation" else "canonical-instruction.txt")}
        slots[slot] = {key: digest(path.read_bytes()) for key, path in files.items()}
    assert source.descriptor_reference is not None
    binding = {"kind": "reviewed_native_source_binding", "reviewNote": "Explicit synthetic fixture binding",
        "target": {"descriptorSha256": source.descriptor_reference["sha256"],
                   "messagesSha256": source.descriptor["messages"]["sha256"],
                   "checkpoints": {k: v["workspace"]["sha256"] for k, v in source.descriptor["checkpoints"].items()}}, "slots": slots}
    (root / source.descriptor["scenario"] / "source-binding.json").write_bytes(encoded(binding))
    return root


def matchup(pair, first_author=Author.KIMI_K3_1_PRO, channel=Channel.USER):
    return make_matchup((PromptSpec(pair.first, Framing.NORMAL, channel, first_author),
                        PromptSpec(pair.second, Framing.CASUAL, Channel.USER, Author.KIMI_K3_1_PRO)),
                       Assistant.KIMI_K3_1_PRO)


def test_lossless_prefix_replacement_and_append(source: NativeSource) -> None:
    variant = TurnVariant("Replacement", {"framing": "normal"}, False)
    plan = replace_user_turn(source, 2, variant)
    assert plan["prefix"] == list(source.messages[:4])
    assert plan["prefix"][2]["content"][0]["text"] == "\n# file with whitespace\n\n"
    assert plan["prefix"][1]["content"][0]["thinking"] == "Inspect existing code."
    assert plan["turns"] == [{"prompt": "Replacement", "variant": variant.metadata}]
    assert "Implementation complete." not in json.dumps(plan["prefix"])
    earlier = replace_user_turn(source, 1, variant)
    assert not earlier["prefix"]
    assert earlier["turns"] == [{"prompt": "Replacement", "variant": variant.metadata}]
    later = replace_user_turn(source, 3, variant)
    assert later["prefix"] == list(source.messages)
    assert plan["fingerprint"] != earlier["fingerprint"] != later["fingerprint"]
    plan["prefix"][0]["content"][0]["text"] = "mutated plan"
    assert source.messages[0]["content"][0]["text"] == "Investigate JSON."


def test_both_momentum_modes_order_and_placeholder_gate(source: NativeSource, bank: Path) -> None:
    pair = load_instruction_pairs(Path("configs/scenario_pairs.yaml"))[0]
    ordered = matchup(pair)
    low = compile_matchup(source, pair, ordered, bank, "after_investigation")
    high = compile_matchup(source, pair, ordered, bank, "after_implementation")
    assert low["prefix"][:-1] == high["prefix"] == list(source.messages[:4])
    assert low["prefix"][-1]["role"] == "user"
    assert low["prefix"][-1]["content"][0]["text"] == "implementation/a normal literal {{ x }}"
    assert len(low["turns"]) == 1 and len(high["turns"]) == 2
    assert low["turns"][0] == high["turns"][1]
    assert low["fingerprint"] != high["fingerprint"]
    reverse = make_matchup(tuple(reversed(ordered.inputs)), ordered.assistant)
    rev = compile_matchup(source, pair, reverse, bank, "after_investigation")
    assert rev["prefix"][-1]["content"][0]["text"].startswith("implementation/b")
    human = matchup(pair, Author.USER)
    with pytest.raises(ValueError, match="placeholder"):
        compile_matchup(source, pair, human, bank, "after_implementation")
    preview = compile_matchup(source, pair, human, bank, "after_implementation", allow_placeholders=True)
    assert preview["containsPlaceholders"]
    output = source.root / "preview.json"
    write_plan(output, preview)
    assert json.loads(output.read_bytes()) == preview


@pytest.mark.parametrize("kwargs,error", [
    ({"framing": Framing.SUBAGENT}, "framings"),
    ({"author": Author.INKLING}, "authors"),
    ({"slot": "missing"}, "slot"),
    ({"side": "a"}, "no side"),
    ({"slot": "implementation"}, "needs side"),
    ({"scenario": "../../escape"}, "escapes"),
])
def test_variant_validation(bank: Path, kwargs, error: str) -> None:
    args: dict[str, Any] = {"root": bank, "scenario": "langchain-json-recovery", "slot": "investigation",
            "author": Author.KIMI_K3_1_PRO, "framing": Framing.NORMAL}
    args.update(kwargs)
    with pytest.raises(ValueError, match=error):
        load_variant(**args)


def test_written_human_and_empty_variant(bank: Path) -> None:
    p = bank / "langchain-json-recovery/investigation/human/normal.txt"
    p.write_text("Human written request.\n")
    value = load_variant(bank, "langchain-json-recovery", "investigation", Author.USER, Framing.NORMAL)
    assert not value.placeholder and value.content == "Human written request."
    p.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_variant(bank, "langchain-json-recovery", "investigation", Author.USER, Framing.NORMAL,
                     allow_placeholders=True)


@pytest.mark.parametrize("change,error", [
    (lambda d: d.update(kind="other"), "pi_source_template"),
    (lambda d: d["system"].update(sha256="0" * 64), "hash mismatch"),
    (lambda d: d["system"].update(path="../../outside"), "escapes"),
])
def test_source_rejects_changed_identity(source: NativeSource, change, error: str) -> None:
    descriptor = copy.deepcopy(source.descriptor)
    change(descriptor)
    (source.root / "bad.json").write_bytes(encoded(descriptor))
    with pytest.raises(ValueError, match=error):
        NativeSource.load(source.root / "bad.json")


@pytest.mark.parametrize("messages,error", [
    ([], "completed assistant"),
    ([{"role": "user"}], "completed assistant"),
    ([{"role": "assistant", "stopReason": "error"}], "normal final"),
    ([{"role": "other"}, {"role": "assistant", "stopReason": "stop"}], "unsupported"),
    ([{"role": "user", "content": 1}, {"role": "assistant", "stopReason": "stop"}], "only text"),
    ([{"role": "user", "content": [{"type": "image"}]}, {"role": "assistant", "stopReason": "stop"}], "only text"),
])
def test_source_message_boundaries(source: NativeSource, messages, error: str) -> None:
    for message in messages:
        if message.get("role") == "assistant":
            message.update(model="moonshotai/kimi-k3.1-pro", provider="openrouter", api="openai-completions")
    data = encoded(messages)
    (source.root / "messages.json").write_bytes(data)
    descriptor = copy.deepcopy(source.descriptor)
    descriptor["messages"]["sha256"] = digest(data)
    (source.root / "bad.json").write_bytes(encoded(descriptor))
    with pytest.raises(ValueError, match=error):
        NativeSource.load(source.root / "bad.json")


def test_string_user_content_and_missing_checkpoint(source: NativeSource) -> None:
    messages = copy.deepcopy(list(source.messages))
    messages[0]["content"] = "Plain user content"
    modified = NativeSource(source.root, source.descriptor, tuple(messages), source.system, source.tools)
    assert replace_user_turn(modified, 1, TurnVariant("new", {}, False))["turns"][0]["prompt"] == "new"
    for turn in (0, 4):
        with pytest.raises(ValueError, match="outside"):
            source.before_turn(turn)
    changed = copy.deepcopy(source.descriptor)
    changed["checkpoints"].pop("1")
    with pytest.raises(ValueError, match="no workspace"):
        NativeSource(source.root, changed, source.messages, source.system, source.tools).before_turn(2)
    (source.root / "1.tar.gz").write_bytes(b"tampered checkpoint")
    with pytest.raises(ValueError, match="hash mismatch"):
        source.before_turn(2)


def test_matchup_rejects_incompatible_treatments(source: NativeSource, bank: Path) -> None:
    pairs = load_instruction_pairs(Path("configs/scenario_pairs.yaml"))
    pair = pairs[0]
    m = matchup(pair)
    for bad_momentum in ("", "other"):
        with pytest.raises(ValueError, match="momentum"):
            compile_matchup(source, pair, m, bank, bad_momentum)
    with pytest.raises(ValueError, match="does not match"):
        compile_matchup(source, pairs[1], matchup(pairs[1]), bank, "after_investigation")
    with pytest.raises(ValueError, match="both sides"):
        compile_matchup(source, pair, matchup(pairs[1]), bank, "after_investigation")
    with pytest.raises(ValueError, match="fixes channel"):
        compile_matchup(source, pair, matchup(pair, channel=Channel.SYSTEM), bank, "after_investigation")
    other = make_matchup(tuple(m.inputs), Assistant.INKLING)
    with pytest.raises(ValueError, match="uses Kimi"):
        compile_matchup(source, pair, other, bank, "after_investigation")


def test_suite_cli_uses_existing_permutations_and_early_placeholder_fail(
    source: NativeSource, bank: Path, tmp_path: Path, capsys,
) -> None:
    pair = load_instruction_pairs(Path("configs/scenario_pairs.yaml"))[0]
    m = matchup(pair)
    suite = tmp_path / "suite.yaml"
    suite.write_text(yaml.safe_dump({"studies": [study_to_dict(make_study(tuple(m.inputs), m.assistant, 1))]}))
    output = tmp_path / "plans"
    args = ["--corpus", str(source.root.parent), "--variants", str(bank), "--suite", str(suite),
            "--output", str(output)]
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["plans"] == 4 and result["inferenceRequests"] == 0
    assert len(list(output.rglob("*.json"))) == 4
    human = matchup(pair, Author.USER)
    suite.write_text(yaml.safe_dump({"studies": [study_to_dict(make_study(tuple(human.inputs), m.assistant, 1))]}))
    with pytest.raises(ValueError, match="placeholder"):
        main(args)
    assert main(args + ["--allow-placeholders"]) == 0
    assert json.loads(capsys.readouterr().out)["placeholderPreviews"]


def test_native_pilot_grid_balances_six_formats() -> None:
    pairs = load_instruction_pairs(Path("configs/scenario_pairs.yaml"))
    studies = pilot_studies(tuple(pairs[:1]))
    assert len(studies) == 36
    assert len({study.inputs for study in studies}) == 36
    assert all(study.assistant is Assistant.KIMI_K3_1_PRO for study in studies)
    assert {s.inputs[0].author for s in studies} == {Author.USER, Author.KIMI_K3_1_PRO}


def test_native_pilot_cli(source: NativeSource, bank: Path, tmp_path: Path, capsys) -> None:
    pairfile = Path("configs/scenario_pairs.yaml")
    single = tmp_path / "pairs.yaml"
    raw = yaml.safe_load(pairfile.read_text())
    raw["pairs"] = raw["pairs"][:1]
    single.write_text(yaml.safe_dump(raw))
    assert main(["--corpus", str(source.root.parent), "--variants", str(bank),
                 "--pairs", str(single), "--pilot-grid", "--allow-placeholders",
                 "--output", str(tmp_path / "grid")]) == 0
    assert json.loads(capsys.readouterr().out)["plans"] == 144


def test_replacement_has_full_turn_qa_inputs(source: NativeSource, bank: Path) -> None:
    variant = load_variant(bank, "langchain-json-recovery", "investigation", Author.KIMI_K3_1_PRO, Framing.NORMAL)
    plan = replace_user_turn(source, 1, variant)
    assert [i["content"] for i in plan["metadata"]["authoringInputs"]] == [t["prompt"] for t in plan["turns"]]
    assert plan["metadata"]["authoringInputs"][0]["spec"]["instruction"] == "Complete assignment."
    p = bank / "langchain-json-recovery/investigation/authoring-source.txt"
    p.write_text("")
    with pytest.raises(ValueError, match="authoring text"):
        load_variant(bank, "langchain-json-recovery", "investigation", Author.KIMI_K3_1_PRO, Framing.NORMAL)


@pytest.mark.parametrize("messages,error", [
    ([{"role": "toolResult", "toolCallId": "x", "toolName": "read"}], "orphan"),
    ([{"role": "assistant", "content": [{"type": "toolCall", "id": "x", "name": "read"}]}], "unresolved"),
    ([{"role": "assistant", "content": [{"type": "toolCall", "id": "x", "name": "read"},
                                        {"type": "toolCall", "id": "x", "name": "read"}]}], "duplicate"),
    ([{"role": "assistant", "content": [{"type": "toolCall", "id": "x", "name": "read"}]},
      {"role": "user", "content": "next"}], "interrupts"),
])
def test_tool_history_rejects_invalid_pairs(messages, error) -> None:
    with pytest.raises(ValueError, match=error):
        validate_history(messages)


def test_one_turn_cli(source: NativeSource, bank: Path, tmp_path: Path, capsys) -> None:
    output = tmp_path / "replacement.json"
    assert turn_main(["--source", str(source.root / "template.json"), "--variants", str(bank),
        "--turn", "2", "--slot", "implementation", "--side", "b", "--author", "Kimi K3.1 Pro",
        "--framing", "normal", "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["inferenceRequests"] == 0
    plan = json.loads(output.read_bytes())
    assert plan["metadata"]["replacedUserTurn"] == 2
    assert plan["turns"][0]["prompt"].startswith("implementation/b")


def test_source_rebinding_and_policy_guard(source: NativeSource, bank: Path) -> None:
    pair = load_instruction_pairs(Path("configs/scenario_pairs.yaml"))[0]
    base = bank / str(pair.pair_id)
    binding_file = base / "source-binding.json"
    binding = json.loads(binding_file.read_bytes())
    binding["target"]["messagesSha256"] = "0" * 64
    binding_file.write_bytes(encoded(binding))
    with pytest.raises(ValueError, match="history/state"):
        compile_matchup(source, pair, matchup(pair), bank, "after_implementation")
    binding["target"]["messagesSha256"] = source.descriptor["messages"]["sha256"]
    binding["reviewNote"] = ""
    binding_file.write_bytes(encoded(binding))
    with pytest.raises(ValueError, match="review note"):
        compile_matchup(source, pair, matchup(pair), bank, "after_implementation")
    binding["reviewNote"] = "Reviewed"
    instruction = base / "implementation/a/canonical-instruction.txt"
    instruction.write_text("Wrong policy")
    binding["slots"]["implementation/a"]["instructionSource"] = digest(instruction.read_bytes())
    binding_file.write_bytes(encoded(binding))
    with pytest.raises(ValueError, match="canonical policy"):
        compile_matchup(source, pair, matchup(pair), bank, "after_implementation")


def test_replay_requires_same_model_and_completed_messages(source: NativeSource) -> None:
    descriptor = copy.deepcopy(source.descriptor)
    descriptor["manifest"]["model"] = "another/model"
    path = source.root / "other.json"
    path.write_bytes(encoded(descriptor))
    with pytest.raises(ValueError, match="must be Kimi"):
        NativeSource.load(path)
    descriptor = copy.deepcopy(source.descriptor)
    messages = copy.deepcopy(list(source.messages))
    messages[1]["model"] = "another/model"
    data = encoded(messages)
    (source.root / "messages.json").write_bytes(data)
    descriptor["messages"]["sha256"] = digest(data)
    path.write_bytes(encoded(descriptor))
    with pytest.raises(ValueError, match="cross-model"):
        NativeSource.load(path)


def test_binding_rejects_changed_receipt_and_sources(source: NativeSource, bank: Path) -> None:
    from reasonese.native_templates import validate_binding
    variant = load_variant(bank, "langchain-json-recovery", "implementation", Author.KIMI_K3_1_PRO, Framing.NORMAL, "a")
    with pytest.raises(ValueError, match="explicit reviewed"):
        validate_binding(source, TurnVariant("x", {}, False))
    binding = Path(variant.metadata["sourceBinding"]["path"])
    original = binding.read_bytes()
    binding.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="binding changed"):
        validate_binding(source, variant)
    binding.write_bytes(original)
    content = json.loads(original)
    content["slots"]["implementation/a"]["authoringSource"] = "0" * 64
    binding.write_bytes(encoded(content))
    changed = load_variant(bank, "langchain-json-recovery", "implementation", Author.KIMI_K3_1_PRO, Framing.NORMAL, "a")
    with pytest.raises(ValueError, match="authoring source differs"):
        validate_binding(source, changed)
    (bank / "langchain-json-recovery/implementation/a/canonical-instruction.txt").write_text("")
    with pytest.raises(ValueError, match="instruction is empty"):
        load_variant(bank, "langchain-json-recovery", "implementation", Author.KIMI_K3_1_PRO, Framing.NORMAL, "a")


@pytest.mark.parametrize("field,value", [("system", {"path": "different.txt", "sha256": "0" * 64}),
                                       ("prepared", {"environment": {"mounts": [{"source": "/", "target": "/host"}]}}),
                                       ("manifest", {"model": "moonshotai/kimi-k3.1-pro", "thinking": "high", "maxModelRequests": 1000000})])
def test_binding_pins_complete_descriptor(source: NativeSource, bank: Path, field: str, value: Any) -> None:
    from reasonese.native_templates import validate_binding

    variant = load_variant(bank, "langchain-json-recovery", "implementation", Author.KIMI_K3_1_PRO,
                           Framing.NORMAL, "a")
    changed = copy.deepcopy(source.descriptor)
    changed[field] = value
    modified = NativeSource(source.root, changed, source.messages, source.system, source.tools,
                            source.descriptor_reference)
    with pytest.raises(ValueError, match="descriptor changed"):
        validate_binding(modified, variant)
    replacement = source.root / "new-descriptor.json"
    replacement.write_bytes(encoded(changed))
    rebound = NativeSource(source.root, changed, source.messages, source.system, source.tools,
                           {"path": str(replacement), "sha256": digest(replacement.read_bytes())})
    with pytest.raises(ValueError, match="complete source descriptor"):
        validate_binding(rebound, variant)
