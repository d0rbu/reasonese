"""Lossless Pi user-turn templates sharing the ordinary experiment axes.

Plans retain native messages/tools and a matching filesystem checkpoint. They do
not coerce a tool trajectory into the older README-only scenario representation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Any

from beartype import beartype

from reasonese.axes import Assistant, Author, Channel, Framing
from reasonese.config import load_study_suite
from reasonese.instructions import InstructionPair, load_instruction_pairs
from reasonese.matchup import Matchup, matchup_to_dict, prompt_spec_to_dict
from reasonese.planning import PromptSpec
from reasonese.scenarios import _template
from reasonese.study import Study, build_trials, make_study


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encoded(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def _read_ref(root: Path, reference: dict[str, Any]) -> tuple[Path, bytes]:
    path = (root / reference["path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("source artifact escapes corpus directory")
    data = path.read_bytes()
    if digest(data) != reference["sha256"]:
        raise ValueError(f"source artifact hash mismatch: {reference['path']}")
    return path, data


@lru_cache(maxsize=128)
def _verify_archive(path: Path, expected: str, size: int, modified: int, changed: int) -> None:
    # Stat identity only controls repeat verification during one offline planning run.
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError("source artifact hash mismatch: workspace checkpoint")


def validate_history(messages: list[dict[str, Any]]) -> None:
    pending: dict[str, str] = {}
    for message in messages:
        if message["role"] == "assistant":
            for block in message.get("content", []):
                if block["type"] == "toolCall":
                    if block["id"] in pending:
                        raise ValueError("duplicate pending tool call")
                    pending[block["id"]] = block["name"]
        elif message["role"] == "toolResult":
            if pending.pop(message["toolCallId"], None) != message["toolName"]:
                raise ValueError("orphan or mismatched tool result")
        elif pending:
            raise ValueError("user turn interrupts pending tools")
    if pending:
        raise ValueError("unresolved source tool calls")


def _text(message: dict[str, Any]) -> str:
    content = message["content"]
    if isinstance(content, str):
        return content
    if not isinstance(content, list) or any(block.get("type") != "text" for block in content):
        raise ValueError("user messages must contain only text")
    return "".join(block["text"] for block in content)


@beartype
@dataclass(frozen=True, slots=True)
class NativeSource:
    """Verified source history and checkpoint descriptor, not experimental output."""

    root: Path
    descriptor: dict[str, Any]
    messages: tuple[dict[str, Any], ...]
    system: str
    tools: list[dict[str, Any]]
    descriptor_reference: dict[str, str] | None = None

    @classmethod
    def load(cls, path: Path) -> NativeSource:
        descriptor_bytes = path.read_bytes()
        descriptor = json.loads(descriptor_bytes)
        if descriptor.get("kind") != "pi_source_template":
            raise ValueError("expected pi_source_template descriptor")
        root = path.parent
        _, messages = _read_ref(root, descriptor["messages"])
        _, system = _read_ref(root, descriptor["system"])
        _, tools = _read_ref(root, descriptor["tools"])
        parsed = json.loads(messages)
        if not parsed or parsed[-1].get("role") != "assistant":
            raise ValueError("source must end on a completed assistant turn")
        if parsed[-1].get("stopReason") != "stop":
            raise ValueError("source must have a normal final response")
        if any(m.get("role") not in {"user", "assistant", "toolResult"} for m in parsed):
            raise ValueError("unsupported Pi source role")
        if descriptor.get("manifest", {}).get("model") != "moonshotai/kimi-k3.1-pro" or descriptor["manifest"].get("thinking") != "high":
            raise ValueError("native source must be Kimi with high reasoning")
        if any((m.get("model"), m.get("provider"), m.get("api")) !=
               ("moonshotai/kimi-k3.1-pro", "openrouter", "openai-completions") or
               m.get("stopReason") not in {"stop", "toolUse"}
               for m in parsed if m["role"] == "assistant"):
            raise ValueError("native source cannot replay cross-model or aborted assistant messages")
        validate_history(parsed)
        for message in parsed:
            if message["role"] == "user":
                _text(message)
        return cls(root, descriptor, tuple(parsed), system.decode(), json.loads(tools),
                   {"path": str(path.resolve()), "sha256": digest(descriptor_bytes)})

    def before_turn(self, turn: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        indices = [i for i, m in enumerate(self.messages) if m["role"] == "user"]
        if not 1 <= turn <= len(indices) + 1:
            raise ValueError("user-turn boundary is outside source history")
        end = indices[turn - 1] if turn <= len(indices) else len(self.messages)
        checkpoint = self.descriptor["checkpoints"].get(str(turn - 1))
        if checkpoint is None:
            raise ValueError(f"no workspace checkpoint before user turn {turn}")
        path = (self.root / checkpoint["workspace"]["path"]).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("source artifact escapes corpus directory")
        stat = path.stat()
        _verify_archive(path, checkpoint["workspace"]["sha256"], stat.st_size,
                        stat.st_mtime_ns, stat.st_ctime_ns)
        state = copy.deepcopy(checkpoint)
        state["workspace"]["path"] = str(path)
        return copy.deepcopy(list(self.messages[:end])), state


@beartype
@dataclass(frozen=True, slots=True)
class TurnVariant:
    content: str
    metadata: dict[str, Any]
    placeholder: bool


def bank_ref(root: Path, path: Path) -> tuple[dict[str, str], bytes]:
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("variant source escapes variant bank")
    data = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": digest(data)}, data


def validate_binding(source: NativeSource, variant: TurnVariant) -> None:
    reference = variant.metadata.get("sourceBinding")
    if reference is None:
        raise ValueError("variant needs an explicit reviewed source binding")
    data = Path(reference["path"]).read_bytes()
    if digest(data) != reference["sha256"]:
        raise ValueError("variant source binding changed")
    binding = json.loads(data)
    target = binding["target"]
    if binding.get("kind") != "reviewed_native_source_binding" or not binding.get("reviewNote", "").strip():
        raise ValueError("variant source binding needs a review note")
    if (source.descriptor_reference is None or
        target.get("descriptorSha256") != source.descriptor_reference["sha256"]):
        raise ValueError("variant source binding does not match complete source descriptor")
    _, raw_descriptor = _read_ref(source.root, source.descriptor_reference)
    if json.loads(raw_descriptor) != source.descriptor:
        raise ValueError("source descriptor changed after loading")
    if target["messagesSha256"] != source.descriptor["messages"]["sha256"] or target["checkpoints"] != {
        key: value["workspace"]["sha256"] for key, value in source.descriptor["checkpoints"].items()
    }:
        raise ValueError("variant source binding does not match native history/state")
    key = variant.metadata["slot"]
    if variant.metadata["side"]:
        key += "/" + variant.metadata["side"]
    expected = binding["slots"][key]
    if any(expected[name] != variant.metadata[name]["sha256"] for name in
           ("authoringSource", "sourceProvenance", "instructionSource")):
        raise ValueError("variant authoring source differs from reviewed binding")


def load_variant(root: Path, scenario: str, slot: str, author: Author, framing: Framing,
                 side: str | None = None, *, allow_placeholders: bool = False) -> TurnVariant:
    if framing not in (Framing.NORMAL, Framing.CASUAL, Framing.PERSUASIVE):
        raise ValueError("native pilot uses normal/casual/persuasive framings")
    if author not in (Author.USER, Author.KIMI_K3_1_PRO):
        raise ValueError("native pilot uses human or Kimi authors")
    if slot not in ("investigation", "implementation"):
        raise ValueError("unknown user-turn slot")
    if (slot == "implementation" and side not in ("a", "b")) or (
        slot == "investigation" and side is not None
    ):
        raise ValueError("implementation needs side a/b; investigation has no side")
    base = root / scenario / slot
    if side is not None:
        base /= side
    filename = base / ("human" if author is Author.USER else "kimi") / f"{framing}.txt"
    if not filename.resolve().is_relative_to(root.resolve()):
        raise ValueError("variant escapes variant bank")
    data = filename.read_bytes()
    content = data.decode().strip()
    placeholder = not content or content.startswith("TODO:")
    if placeholder and not allow_placeholders:
        raise ValueError(f"variant is still a placeholder: {filename}")
    if not content:
        raise ValueError("variant text must not be empty")
    authoring_ref, authoring_data = bank_ref(root, base / "authoring-source.txt")
    provenance_ref, provenance_data = bank_ref(root, base / "source-provenance.json")
    binding_ref, _ = bank_ref(root, root / scenario / "source-binding.json")
    instruction_ref, instruction_data = bank_ref(root, base / ("canonical-instruction.txt" if slot == "implementation" else "authoring-source.txt"))
    provenance = json.loads(provenance_data)
    if provenance["authoringSourceSha256"] != digest(authoring_data.decode().strip().encode()):
        raise ValueError("variant authoring text differs from declared provenance")
    instruction = instruction_data.decode().strip()
    if not instruction:
        raise ValueError("variant authoring instruction is empty")
    return TurnVariant(_template("{{ turn }}").render(turn=content),
        {"author": str(author), "framing": str(framing), "channel": str(Channel.USER),
         "slot": slot, "side": side,
         "authoringSource": authoring_ref, "sourceProvenance": provenance_ref,
         "instructionSource": instruction_ref, "sourceBinding": binding_ref,
         "spec": {"instruction": instruction, "author": str(author), "framing": str(framing),
                  "channel": str(Channel.USER)}, "path": str(filename.resolve()), "sha256": digest(data)},
        placeholder)


def _plan(source: NativeSource, prefix: list[dict[str, Any]], checkpoint: dict[str, Any],
          turns: list[dict[str, Any]], placeholders: bool, mode: str,
          metadata: dict[str, Any]) -> dict[str, Any]:
    value = {"kind": "pi_experiment_plan", "mode": mode,
        "model": "moonshotai/kimi-k3.1-pro", "thinking": "high",
        "source": copy.deepcopy(source.descriptor),
        "sourceDescriptor": copy.deepcopy(source.descriptor_reference),
        "corpusRoot": str(source.root.resolve()),
        "system": source.system,
        "tools": copy.deepcopy(source.tools), "prefix": prefix,
        "checkpoint": checkpoint, "turns": turns, "containsPlaceholders": placeholders,
        "metadata": metadata}
    value["fingerprint"] = digest(encoded(value))
    return value


@beartype
def replace_user_turn(source: NativeSource, turn: int, variant: TurnVariant) -> dict[str, Any]:
    """Discard downstream assistant/tool history after a changed user turn."""
    prefix, checkpoint = source.before_turn(turn)
    if "spec" in variant.metadata:
        validate_binding(source, variant)
    pending = [{"prompt": variant.content, "variant": variant.metadata}]
    inputs = []
    if "spec" in variant.metadata:
        inputs = [{"spec": variant.metadata["spec"], "content": variant.content}]
    return _plan(source, prefix, checkpoint, pending, variant.placeholder, "replace_user_turn",
                 {"replacedUserTurn": turn, "authoringInputs": inputs})


@beartype
def compile_matchup(source: NativeSource, pair: InstructionPair, matchup: Matchup, variants: Path,
                    momentum: str, *, allow_placeholders: bool = False) -> dict[str, Any]:
    """Render the two cell inputs at explicitly different momentum levels.

    After investigation: distinct consecutive user messages, then let Pi respond.
    After implementation: actually execute the first input before delivering the
    other. Both begin at checkpoint1; recorded implementation outputs are not
    reused after replacing the prompt that elicited them.
    """
    if momentum not in {"after_investigation", "after_implementation"}:
        raise ValueError("unknown momentum level")
    if source.descriptor["scenario"] != str(pair.pair_id):
        raise ValueError("source template does not match instruction pair")
    if {spec.instruction for spec in matchup.inputs} != set(pair.instructions):
        raise ValueError("native matchup must contain both sides of this pair")
    if any(spec.channel is not Channel.USER for spec in matchup.inputs):
        raise ValueError("native user-turn pilot fixes channel to user message")
    if str(matchup.assistant) != "Kimi K3.1 Pro":
        raise ValueError("native Pi pilot uses Kimi K3.1 Pro")
    prefix, checkpoint = source.before_turn(2)
    selected = [load_variant(variants, str(pair.pair_id), "implementation", spec.author,
        spec.framing, "a" if spec.instruction == pair.first else "b",
        allow_placeholders=allow_placeholders) for spec in matchup.inputs]
    for spec, variant in zip(matchup.inputs, selected, strict=True):
        validate_binding(source, variant)
        if variant.metadata["spec"] != prompt_spec_to_dict(spec):
            raise ValueError("variant canonical policy differs from instruction pair")
    pending = [{"prompt": item.content, "variant": item.metadata} for item in selected]
    placements = []
    if momentum == "after_investigation":
        placements.append({"position": 1, "message": len(prefix)})
        prefix.append({"role": "user", "content": [{"type": "text", "text": pending[0]["prompt"]}],
                       "timestamp": 0})
        pending = pending[1:]
    return _plan(source, prefix, checkpoint, pending, any(v.placeholder for v in selected),
        "conflicting_user_turns", {"momentum": momentum, "matchup": matchup_to_dict(matchup),
                                  "historicalPlacements": placements,
                                  "authoringVariants": [v.metadata for v in selected],
                                  "authoringInputs": [{"spec": prompt_spec_to_dict(spec),
                                      "content": variant.content}
                                      for spec, variant in zip(matchup.inputs, selected, strict=True)]})


def write_plan(path: Path, plan: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plan["fingerprint"] = digest(encoded({k: v for k, v in plan.items() if k != "fingerprint"}))
    path.write_bytes(encoded(plan))


def pilot_studies(pairs: tuple[InstructionPair, ...]) -> tuple[Study, ...]:
    """Six formats on each policy side, with the ordinary balanced trial planner."""
    formats = tuple(product((Author.USER, Author.KIMI_K3_1_PRO),
                            (Framing.NORMAL, Framing.CASUAL, Framing.PERSUASIVE)))
    return tuple(make_study((PromptSpec(pair.first, left[1], Channel.USER, left[0]),
                             PromptSpec(pair.second, right[1], Channel.USER, right[0])),
                            Assistant.KIMI_K3_1_PRO, 1)
                 for pair in pairs for left, right in product(formats, repeat=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reasonese-native-plan")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--variants", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, default=Path("configs/scenario_pairs.yaml"))
    design = parser.add_mutually_exclusive_group(required=True)
    design.add_argument("--suite", type=Path)
    design.add_argument("--pilot-grid", action="store_true", help="human/Kimi × three framings on both sides")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-placeholders", action="store_true", help="offline previews only")
    args = parser.parse_args(argv)
    pairs = load_instruction_pairs(args.pairs)
    by_instruction = {instruction: pair for pair in pairs for instruction in pair.instructions}
    count = 0
    sources: dict[str, NativeSource] = {}
    studies = pilot_studies(pairs) if args.pilot_grid else load_study_suite(args.suite)
    for study in studies:
        for trial in build_trials(study):
            pair = by_instruction[trial.matchup.inputs[0].instruction]
            case_id = str(pair.pair_id)
            if case_id not in sources:
                sources[case_id] = NativeSource.load(args.corpus / case_id / "template.json")
            source = sources[case_id]
            for momentum in ("after_investigation", "after_implementation"):
                plan = compile_matchup(source, pair, trial.matchup, args.variants, momentum,
                                       allow_placeholders=args.allow_placeholders)
                plan["trial"] = {"trial_id": str(trial.trial_id), "permutation": int(trial.permutation),
                                 "rollout": int(trial.rollout)}
                write_plan(args.output / momentum / f"{trial.trial_id}.json", plan)
                count += 1
    print(json.dumps({"plans": count, "output": str(args.output),
                     "placeholderPreviews": args.allow_placeholders, "inferenceRequests": 0}))
    return 0


def turn_main(argv: list[str] | None = None) -> int:
    """Compile one named user-turn replacement or append, without inference."""
    parser = argparse.ArgumentParser(prog="reasonese-native-turn")
    parser.add_argument("--source", type=Path, required=True, help="CASE/template.json")
    parser.add_argument("--variants", type=Path, default=Path("prompts/scenario_turns"))
    parser.add_argument("--turn", type=int, required=True)
    parser.add_argument("--slot", choices=("investigation", "implementation"), required=True)
    parser.add_argument("--side", choices=("a", "b"))
    parser.add_argument("--author", choices=(str(Author.USER), str(Author.KIMI_K3_1_PRO)), required=True)
    parser.add_argument("--framing", choices=("normal", "casual", "persuasive"), required=True)
    parser.add_argument("--allow-placeholders", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source = NativeSource.load(args.source)
    variant = load_variant(args.variants, source.descriptor["scenario"], args.slot,
                           Author(args.author), Framing(args.framing), args.side,
                           allow_placeholders=args.allow_placeholders)
    plan = replace_user_turn(source, args.turn, variant)
    write_plan(args.output, plan)
    print(json.dumps({"output": str(args.output), "replacedUserTurn": args.turn,
                      "containsPlaceholders": plan["containsPlaceholders"], "inferenceRequests": 0}))
    return 0
