"""Audit exact native Pi user turns before launching an authorized continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

from beartype import beartype

from reasonese.axes import Channel
from reasonese.check_messages import audit_messages, require_compliant_messages
from reasonese.conversation import GeneratedMessage, GeneratedText
from reasonese.matchup import prompt_spec_from_dict, prompt_spec_to_dict
from reasonese.message_qa import message_qa_request_fingerprint
from reasonese.message_qa_cache import YamlMessageQaCache
from reasonese.openrouter import JsonTransport, OpenRouterClient, RequestsTransport
from reasonese.routing import CollectionRouting


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return cast(dict[str, Any], value)


def _text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("native user turns must contain nonempty text")
    if re.search(r"\bTODO\b", value, re.IGNORECASE):
        raise ValueError("native user turn contains a TODO placeholder")
    return value


def _messages(plan: dict[str, Any]) -> tuple[GeneratedMessage, ...]:
    if plan.get("kind") != "pi_experiment_plan":
        raise ValueError("expected a pi_experiment_plan")
    if plan.get("containsPlaceholders") is not False:
        raise ValueError("native plan contains placeholders or omits placeholder status")
    metadata = _object(plan.get("metadata"), "plan metadata")
    inputs = metadata.get("authoringInputs")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("native plan requires nonempty metadata.authoringInputs for QA")
    messages = []
    for raw in inputs:
        item = _object(raw, "authoring input")
        spec = prompt_spec_from_dict(item.get("spec"))
        if spec.channel is not Channel.USER:
            raise ValueError("native user-turn QA requires user message specifications")
        messages.append(
            GeneratedMessage(spec, GeneratedText.parse(_text(item.get("content"))), None)
        )

    # Audit the actual delivered turns, including the first input inserted into
    # the prefix for the after-investigation design, rather than a detached copy.
    placements = metadata.get("historicalPlacements", [])
    prefix = plan.get("prefix")
    turns = plan.get("turns")
    if not isinstance(placements, list) or not isinstance(prefix, list):
        raise ValueError("invalid native prefix or historical placements")
    if not isinstance(turns, list) or not turns:
        raise ValueError("native plan requires pending user turns")
    delivered = []
    for position, raw in enumerate(placements, 1):
        placement = _object(raw, "historical placement")
        index = placement.get("message")
        if placement.get("position") != position or type(index) is not int:
            raise ValueError("historical placements must identify consecutive authoring inputs")
        if not 0 <= index < len(prefix):
            raise ValueError("historical placement lies outside native prefix")
        message = _object(prefix[index], "placed user message")
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, list) or len(content) != 1:
            raise ValueError("historical placement must contain one user text block")
        block = _object(content[0], "placed user content")
        if block.get("type") != "text":
            raise ValueError("historical placement must be text")
        delivered.append(_text(block.get("text")))
    delivered.extend(_text(_object(turn, "pending user turn").get("prompt")) for turn in turns)
    if delivered != [str(message.content) for message in messages]:
        raise ValueError("QA inputs do not match the exact delivered native user turns")
    return tuple(messages)


def _transport() -> JsonTransport | None:
    key = os.environ.get("OPENROUTER_API_KEY")
    return RequestsTransport(key) if key is not None else None


def _launch(command: Sequence[str]) -> None:
    subprocess.run(command, check=True)


@beartype
def run_native(
    plan_path: Path,
    output: Path,
    qa_cache_path: Path,
    *,
    allow_paid: bool = False,
    transport_factory: Callable[[], JsonTransport | None] = _transport,
    launcher: Callable[[Sequence[str]], None] = _launch,
) -> Path:
    """Cache exact Luna QA, bind its receipt to the plan bytes, then invoke Pi.

    The injected boundaries support offline checks. The explicit opt-in covers
    both uncached QA and the assistant continuation, even when QA is cached.
    """
    plan_path = plan_path.resolve()
    raw = plan_path.read_bytes()
    plan_sha256 = hashlib.sha256(raw).hexdigest()
    messages = _messages(_object(json.loads(raw), "native plan"))
    routing = CollectionRouting(allow_paid=allow_paid)
    routing.require_paid("native Pi continuation and any uncached message QA")
    if output.exists() or output.is_symlink():
        raise ValueError("native continuation output must be fresh; retain failed attempts")
    receipt_path = output.resolve().with_name(output.name + f".message-qa.{plan_sha256}.json")
    if receipt_path.exists() or receipt_path.is_symlink():
        raise ValueError("native QA receipt already exists; retain attempts and choose a fresh output name")
    fixture = transport_factory is not _transport or launcher is not _launch
    if fixture:
        # Injected judgments must never populate the production QA cache.
        qa_cache_path = qa_cache_path.with_name(
            f"{qa_cache_path.stem}.fixture{qa_cache_path.suffix}"
        )
    runner = Path(__file__).resolve().parents[1] / "tools/pi-transcripts/continue.mjs"
    launcher(["node", str(runner), "--plan", str(plan_path), "--preflight-only"])
    transport = transport_factory()
    result = audit_messages(
        messages,
        YamlMessageQaCache(qa_cache_path),
        OpenRouterClient(transport) if transport is not None else None,
        routing=routing,
    )
    receipt = {
        "kind": "pi_message_qa_receipt",
        "fixture": fixture,
        "planSha256": plan_sha256,
        "passed": all(verdict.complies for verdict in result.verdicts),
        "judge": "openai/gpt-5.6-luna:batch",
        "cacheHits": int(result.cache_hits),
        "qaCache": str(qa_cache_path.resolve()),
        "inputs": [
            {
                "spec": prompt_spec_to_dict(verdict.spec),
                "contentSha256": hashlib.sha256(verdict.content.encode()).hexdigest(),
                "requestFingerprint": message_qa_request_fingerprint(message),
                "complies": verdict.complies,
                "issues": list(verdict.issues),
                "judgeResponse": verdict.response,
            }
            for message, verdict in zip(messages, result.verdicts, strict=True)
        ],
    }
    # The Node collector requires a fresh output directory. Keep the prelaunch
    # receipt beside it; the collector copies it into the completed run.
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    require_compliant_messages(result)
    if hashlib.sha256(plan_path.read_bytes()).hexdigest() != plan_sha256:
        raise ValueError("native plan changed during message QA; recompile and audit it again")
    launcher(
        [
            "node",
            str(runner),
            "--plan",
            str(plan_path),
            "--qa-receipt",
            str(receipt_path),
            "--out",
            str(output.resolve()),
            "--allow-paid",
        ]
    )
    return receipt_path


@beartype
def main(argv: Sequence[str] | None = None) -> int:
    """Run one native plan after exact-text QA and explicit spending opt-in."""
    parser = argparse.ArgumentParser(prog="reasonese-native-run")
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--qa-cache", type=Path, default=Path("out/native-message-qa.yaml"))
    parser.add_argument("--allow-paid", action="store_true")
    args = parser.parse_args(argv)
    try:
        receipt = run_native(args.plan, args.out, args.qa_cache, allow_paid=args.allow_paid)
    except (
        OSError,
        RuntimeError,
        TimeoutError,
        TypeError,
        ValueError,
        subprocess.SubprocessError,
    ) as error:
        parser.error(str(error))
    print(
        json.dumps({"qaReceipt": str(receipt), "output": str(args.out.resolve())}, sort_keys=True)
    )
    return 0
