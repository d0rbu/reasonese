#!/usr/bin/env python3
"""Export a fresh, hash-bound native Pi review without changing its evidence."""

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tarfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

MODEL = "moonshotai/kimi-k3.1-pro"
CHUNK = 1024 * 1024
ASSETS = Path(__file__).with_name("viewer")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def relative(name):
    require(isinstance(name, str) and name and "\\" not in name, "Invalid relative path")
    path = PurePosixPath(name)
    require(not path.is_absolute() and all(p not in {"", ".", ".."} for p in name.split("/")),
            "Unsafe relative path")
    return path


def tree_files(root):
    """Reject links/special files rather than following assistant-controlled paths."""
    require(root.is_dir() and not root.is_symlink(), "Expected a real directory")
    files = {}
    for current, directories, names in os.walk(root, followlinks=False):
        for name in directories + names:
            path = Path(current) / name
            mode = path.lstat().st_mode
            require(stat.S_ISDIR(mode) or stat.S_ISREG(mode), "Source contains a link or special file")
        for name in names:
            path = Path(current) / name
            files[path.relative_to(root).as_posix()] = path
    return files


def open_member(root, name):
    """Open a workspace file through directory fds, without symlink traversal."""
    parts = relative(name).parts
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        file_descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=descriptor)
        require(stat.S_ISREG(os.fstat(file_descriptor).st_mode), "Expected a regular changed file")
        return os.fdopen(file_descriptor, "rb")
    finally:
        os.close(descriptor)


def copy_member(root, name, destination, expected):
    digest = hashlib.sha256()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open_member(root, name) as source, destination.open("xb") as target:
        for chunk in iter(lambda: source.read(CHUNK), b""):
            digest.update(chunk)
            target.write(chunk)
    require(digest.hexdigest() == expected and sha(destination) == expected,
            "Changed file hash does not match file receipt")


def text(message):
    content = message.get("content", "")
    return content if isinstance(content, str) else "".join(
        block.get("text", "") for block in content if block.get("type") == "text")


def thinking(message):
    return "".join(block.get("thinking", "") for block in message["content"]
                   if block.get("type") == "thinking")


def compact_hashes(paths):
    # The collector hashes JSON.stringify, whose floats differ from Python JSON.
    script = """import{readFileSync}from'node:fs';import{createHash}from'node:crypto';
console.log(JSON.stringify(process.argv.slice(1).map(p=>createHash('sha256').update(
JSON.stringify(JSON.parse(readFileSync(p,'utf8')))).digest('hex'))));"""
    result = subprocess.run(["node", "--input-type=module", "-e", script, *map(str, paths)],
                            check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def verify_reasoning(source, messages, prompts, system, tools, manifest):
    assistants = [m for m in messages if m["role"] == "assistant"]
    requests = sorted((source / "http").glob("*.request.json"))
    responses = sorted((source / "http").glob("*.response.txt"))
    metadata = sorted((source / "http").glob("*.response.json"))
    require(len(requests) == len(responses) == len(metadata) == len(assistants),
            "HTTP/native response count differs")
    records, reported_models, reported_providers = [], set(), set()
    native_reasoning = [thinking(assistant) for assistant in assistants]
    for index, (request_path, response_path, meta_path, assistant) in enumerate(
            zip(requests, responses, metadata, assistants, strict=True)):
        prefix = f"{index + 1:03d}"
        require(request_path.name == prefix + ".request.json"
                and response_path.name == prefix + ".response.txt"
                and meta_path.name == prefix + ".response.json", "HTTP request indices differ")
        request = read(request_path)
        require(read(meta_path)["status"] == 200, "Unsuccessful HTTP response")
        require(request["model"] == MODEL and request.get("stream") is True
                and request.get("reasoning", {}).get("effort") == "high",
                "Unexpected HTTP model or reasoning route")
        require(request.get("max_tokens") == manifest["maxOutputTokens"], "Output token limit differs")
        actual_prompts = [text(m) for m in request["messages"] if m["role"] == "user"]
        require(actual_prompts and actual_prompts == prompts[:len(actual_prompts)],
                "HTTP user prompts differ from native inputs")
        require([m.get("content") for m in request["messages"] if m["role"] == "system"] == [system],
                "HTTP system prompt differs")
        delivered_tools = [item["function"] for item in request.get("tools", [])]
        require([{k: tool[k] for k in ("name", "description", "parameters")} for tool in delivered_tools]
                == tools, "HTTP tools differ from captured definitions")
        previous = [m for m in request["messages"] if m["role"] == "assistant"]
        require(len(previous) == index, "HTTP assistant history count differs")
        for prior_index, (old, expected) in enumerate(zip(previous, native_reasoning[:index], strict=True)):
            require(not expected or any(old.get(field) == expected for field in
                    ("reasoning", "reasoning_content", "reasoning_text")),
                    "Prior readable reasoning was not carried into an HTTP request")
            require((old.get("content") or "") == text(assistants[prior_index]),
                    "Prior assistant text differs in an HTTP request")
        parts, fields, types, models, providers, generation_ids = [], set(), set(), set(), set(), set()
        text_parts = []
        encrypted = 0
        for line in response_path.read_text().splitlines():
            if not line.startswith("data: {"):
                continue
            chunk = json.loads(line[6:])
            require(not chunk.get("error"), "Provider stream contains an error")
            for key, values in (("model", models), ("provider", providers), ("id", generation_ids)):
                if chunk.get(key):
                    values.add(chunk[key])
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                if isinstance(delta.get("content"), str):
                    text_parts.append(delta["content"])
                for field in ("reasoning_content", "reasoning", "reasoning_text"):
                    if isinstance(delta.get(field), str) and delta[field]:
                        parts.append(delta[field])
                        fields.add(field)
                        break
                for detail in delta.get("reasoning_details") or []:
                    types.add(detail.get("type", "unknown"))
                    encrypted += detail.get("type") == "reasoning.encrypted"
        require("".join(parts) == native_reasoning[index], "Raw SSE/native reasoning differs")
        require("".join(text_parts) == text(assistant), "Raw SSE/native visible text differs")
        require(models == {MODEL} and providers and generation_ids == {assistant["responseId"]},
                "Raw SSE/native response identity differs")
        reported_models.update(models)
        reported_providers.update(providers)
        records.append({"request": index + 1, "responseId": assistant["responseId"],
                        "provider": sorted(providers), "characters": len(native_reasoning[index]),
                        "readableReasoning": bool(native_reasoning[index].strip()),
                        "reasoningFields": sorted(fields), "detailTypes": sorted(types),
                        "encryptedBlocks": encrypted, "rawMatchesPi": True,
                        "preservedInNextRequest": True if index + 1 < len(assistants) else None,
                        "reasoningSha256": hashlib.sha256(native_reasoning[index].encode()).hexdigest()})
    count = sum(r["readableReasoning"] for r in records)
    characters = sum(r["characters"] for r in records)
    encrypted = sum(r["encryptedBlocks"] for r in records)
    report = {"summary": f"{count} of {len(records)} assistant responses contain readable reasoning "
              f"({characters:,} characters). Raw SSE text matches Pi's stored reasoning; every subsequent "
              f"request preserves all prior returned reasoning. {encrypted} encrypted blocks were observed.",
              "assistantResponses": len(records), "responsesWithReadableReasoning": count,
              "readableReasoningCharacters": characters, "encryptedBlocks": encrypted,
              "allRawMatchesPi": True, "allAvailableReasoningPreserved": True, "records": records,
              "scope": "Checks provider-returned text and transmission fidelity; does not establish "
                       "complete or causally faithful access to internal computation."}
    return report, sorted(reported_models), sorted(reported_providers)


def verify_source(case, *, fixture=False):
    source = case / "trajectory"
    artifacts = tree_files(source)
    outcome = read(source / "outcome.json")
    require(outcome["status"] == "completed" and not outcome.get("captureErrors")
            and outcome.get("error") is None, "Expected a completed capture without errors")
    expected = outcome["artifactHashes"]
    require(set(artifacts) == {*expected, "outcome.json"}, "Source artifact manifest is incomplete")
    hashes = {}
    for name, path in artifacts.items():
        relative(name)
        hashes[name] = sha(path)
        require(name == "outcome.json" or hashes[name] == expected[name], "Source artifact hash differs: " + name)
    provenance = read(source / "provenance.json")
    require(provenance["kind"] == "live_pi_session" and provenance["collectionPurpose"] == "base_transcript"
            and (fixture or not provenance.get("fixture")), "Expected a live base transcript")
    manifest, scenario = provenance["manifest"], provenance["scenario"]
    require(manifest["model"] == MODEL and manifest["thinking"] == "high", "Unexpected source model")
    require(not {"conflictInstructions", "intendedFraming", "decisiveCase"}.intersection(scenario),
            "Base transcript contains experimental treatments")
    messages = read(source / "messages.json")
    require(hashes["messages.json"] == outcome["messagesSha256"], "Outcome message hash differs")
    require(messages and messages[-1]["role"] == "assistant" and messages[-1]["stopReason"] == "stop",
            "Expected a normal final assistant response")
    assistants = [m for m in messages if m["role"] == "assistant"]
    require(len(assistants) == outcome["modelRequests"], "Outcome request count differs")
    for message in assistants:
        require((message.get("model"), message.get("provider"), message.get("api"))
                == (MODEL, "openrouter", "openai-completions")
                and message.get("stopReason") in {"stop", "toolUse"}, "Native model/API identity differs")
    pending = set()
    tool_results = 0
    for message in messages:
        if message["role"] == "assistant":
            require(not pending, "Native tool results are missing")
            calls = [b["id"] for b in message["content"] if b["type"] == "toolCall"]
            require(len(calls) == len(set(calls)), "Duplicate tool call IDs")
            pending.update(calls)
        elif message["role"] == "toolResult":
            require(message["toolCallId"] in pending, "Native tool result has no matching call")
            pending.remove(message["toolCallId"])
            tool_results += 1
        elif message["role"] == "user":
            require(not pending, "User input interrupts pending tool results")
        else:
            raise ValueError("Unexpected native message role")
    require(not pending and tool_results == outcome["toolResults"], "Native tool accounting differs")
    input_paths = sorted((source / "turns").glob("*.input.json"))
    turns = [read(path) for path in input_paths]
    prompts = [text(m) for m in messages if m["role"] == "user"]
    require(len(turns) == outcome["userTurns"] and 2 <= len(turns) <= scenario.get("maxUserTurns", 4),
            "Unexpected user-turn count")
    require(prompts == [turn["prompt"] for turn in turns] and prompts[0] == scenario["prompt"],
            "Native user inputs differ")
    prefix_paths = []
    prior = []
    for index, (turn, input_path) in enumerate(zip(turns, input_paths, strict=True)):
        prefix = f"{index + 1:03d}"
        require(input_path.name == prefix + ".input.json" and turn["turn"] == index + 1, "Turn indices differ")
        observed_path = source / "turns" / (prefix + ".messages.json")
        observed = read(observed_path)
        require(messages[:len(observed)] == observed and observed[:len(prior)] == prior
                and observed[len(prior)]["role"] == "user"
                and len([m for m in observed if m["role"] == "user"]) == index + 1
                and observed[-1]["role"] == "assistant" and observed[-1]["stopReason"] == "stop",
                "Recorded history is not a completed native turn prefix")
        if index:
            prefix_paths.append(source / "turns" / f"{index:03d}.messages.json")
        prior = observed
    require(prior == messages, "Final turn capture differs from native messages")
    prefix_hashes = [hashlib.sha256(b"[]").hexdigest(), *compact_hashes(prefix_paths)]
    require(prefix_hashes == [t["basedOnMessagesSha256"] for t in turns], "Observed-history prefix hash differs")
    session_name = Path(outcome["sessionFile"]).name
    relative(session_name)
    entries = [json.loads(line) for line in (source / "sessions" / session_name).read_text().splitlines()]
    require([entry["message"] for entry in entries if entry["type"] == "message"] == messages,
            "Native session differs from messages")
    system, tools = (source / "system.txt").read_text(), read(source / "tools.json")
    reasoning, models, providers = verify_reasoning(source, messages, prompts, system, tools, manifest)
    snapshots = []
    for name, path in artifacts.items():
        if not name.endswith(".workspace.tar.gz"):
            continue
        metadata_name = name.removesuffix(".tar.gz") + ".json"
        snapshot = read(source / metadata_name)
        require(snapshot["archive"] == path.name and snapshot["sha256"] == hashes[name]
                and snapshot["bytes"] == path.stat().st_size, "Workspace snapshot metadata differs")
        snapshots.append({"name": name, "bytes": path.stat().st_size, "sha256": hashes[name],
                          "metadata": metadata_name, "metadataSha256": hashes[metadata_name],
                          "omittedFromWebBundle": True})
    return {"source": source, "artifacts": artifacts, "hashes": hashes, "outcome": outcome,
            "provenance": provenance, "messages": messages, "turns": turns, "system": system,
            "toolDefinitions": tools, "reasoning": reasoning, "reportedModels": models,
            "reportedProviders": providers, "snapshots": snapshots}


def validate_support(inputs, source):
    message_sha, patch_sha = source["hashes"]["messages.json"], source["hashes"]["changes.patch"]
    validation, explainer, files, selection = inputs
    for item in inputs:
        require(item.get("sourceMessagesSha256") == message_sha, "Support receipt has a stale message hash")
    for item in (validation, explainer, files):
        require(item.get("sourcePatchSha256") == patch_sha, "Support receipt has a stale patch hash")
    require(validation.get("sourceTranscriptUnchanged") is True
            and validation.get("sourcePatchUnchanged") is True, "Validation must preserve source evidence")
    checks = validation["checks"]
    require(checks and all(type(c.get("passed")) is bool for c in checks)
            and type(validation.get("allChecksPassed")) is bool
            and validation.get("allChecksPassed") == all(c["passed"] for c in checks), "Validation checks disagree")
    indices = [i for i, m in enumerate(source["messages"]) if m["role"] == "user"]
    require(len(explainer["turns"]) == len(indices), "Explainer must cover every actual user turn")
    for number, (step, index) in enumerate(zip(explainer["turns"], indices, strict=True), 1):
        require(step["turn"] == number and step.get("messageIndex", index) == index
                and step.get("title") and step.get("summary"), "Explainer turn does not match actual source")
    require(all(explainer.get(key) for key in ("findingTitle", "finding", "limit")), "Explainer is incomplete")
    require(selection.get("selectionConditioned") is True and selection.get("criteria"),
            "Selection-conditioned resampling must be disclosed")
    attempts = selection["attempts"]
    require(attempts and len({a["id"] for a in attempts}) == len(attempts)
            and all(a.get("retained") is True and a.get("status") for a in attempts),
            "Selection attempts must be uniquely identified and retained")
    selected = [a for a in attempts if a["id"] == selection["selectedAttempt"]]
    require(len(selected) == 1 and selected[0].get("sourceMessagesSha256") == message_sha
            and selected[0]["status"] == "completed", "Selected attempt differs from exported source")
    if "attemptCount" in selection:
        require(selection["attemptCount"] == len(attempts), "Selection attempt count differs")
    corpus = selection.get("downloadableCorpus")
    if corpus:
        require(urlparse(corpus["url"]).scheme == "https" and len(corpus["sha256"]) == 64
                and isinstance(corpus["bytes"], int) and corpus["bytes"] > 0, "Invalid full-corpus download pin")
    require(isinstance(files["files"], list), "Expected a file-list receipt")
    names = []
    for item in files["files"]:
        relative(item["path"])
        names.append(item["path"])
        require((item.get("deleted") is True and item.get("sha256") is None)
                or (not item.get("deleted") and isinstance(item.get("sha256"), str)
                    and len(item["sha256"]) == 64), "Invalid changed-file hash")
    require(len(names) == len(set(names)), "Duplicate changed file")


def scan_credentials(root):
    secret = os.environ.get("OPENROUTER_API_KEY", "").encode()
    if not secret:
        return "not_available"
    for name, path in tree_files(root).items():
        require(secret not in name.encode(), "Credential scan found a credential; export rejected")
        overlap = b""
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(CHUNK), b""):
                joined = overlap + chunk
                require(secret not in joined, "Credential scan found a credential; export rejected")
                overlap = joined[-max(0, len(secret) - 1):] if len(secret) > 1 else b""
    return "passed"


def export_review(case_dir, validation_path, explainer_path, files_path, selection_path, output, *, fixture=False):
    case = Path(case_dir).resolve(strict=True)
    output = Path(output).absolute()
    require(not output.exists() and not output.is_symlink(), "Output must be a new page directory")
    require(not output.resolve().is_relative_to(case), "Output must be outside the source case")
    source = verify_source(case, fixture=fixture)
    support_paths = [Path(p) for p in (validation_path, explainer_path, files_path, selection_path)]
    support_bytes = [path.read_bytes() for path in support_paths]
    inputs = [json.loads(value) for value in support_bytes]
    validate_support(inputs, source)
    output.mkdir(parents=True)
    try:
        return write_export(case, output, source, support_bytes, inputs, fixture)
    except BaseException:
        # This invocation created the directory. A rejected export must not leave
        # a partial page (in particular one that failed the credential scan).
        shutil.rmtree(output)
        raise


def write_export(case, output, source, support_bytes, inputs, fixture):
    validation, explainer, file_receipt, selection = inputs
    copied, files = [], []
    for name in sorted(source["artifacts"]):
        if name.endswith(".workspace.tar.gz"):
            continue
        destination = output / "evidence" / name
        copy_member(source["source"], name, destination, source["hashes"][name])
        copied.append({"name": name, "bytes": destination.stat().st_size, "sha256": source["hashes"][name]})
    for item in file_receipt["files"]:
        name = item["path"]
        if item.get("deleted"):
            try:
                with open_member(case / "workspace", name):
                    pass
            except FileNotFoundError:
                pass
            else:
                raise ValueError("Deleted changed file is still present")
            files.append({"path": name, "deleted": True, "sha256": None,
                          "content": "[File deleted in this session]"})
        else:
            destination = output / "files" / name
            copy_member(case / "workspace", name, destination, item["sha256"])
            files.append({"path": name, "sha256": item["sha256"],
                          "content": destination.read_text(errors="replace")})
    for name, content in zip(("validation.json", "explainer.json", "files-receipt.json", "selection.json"),
                             support_bytes, strict=True):
        (output / name).write_bytes(content)
    write(output / "reasoning-verification.json", source["reasoning"])
    write(output / "source-artifact-hashes.json", source["hashes"])
    corpus = {"notice": "Full workspace snapshots are retained separately in the downloadable corpus. "
              "They are excluded from this smaller website evidence bundle; snapshot metadata and SHA-256 "
              "pins remain here. The transcript and provider evidence retain their original bytes.",
              "snapshots": source["snapshots"], "download": selection.get("downloadableCorpus")}
    write(output / "corpus.json", corpus)
    for step, index in zip(explainer["turns"], [i for i, m in enumerate(source["messages"]) if m["role"] == "user"], strict=True):
        step["messageIndex"] = index
    payload = {key: source[key] for key in ("provenance", "outcome", "messages", "turns", "system",
                                           "toolDefinitions", "reasoning", "reportedModels", "reportedProviders")}
    payload.update(files=files, patch=(source["source"] / "changes.patch").read_text(), artifacts=copied,
                   validation=validation, explainer=explainer, selection=selection, corpus=corpus,
                   catalogUrl="catalog.json", fixture=fixture)
    write(output / "data.json", payload)
    scenario = source["provenance"]["scenario"]
    write(output / "catalog.json", {"scenarios": [{"id": scenario["id"], "title": scenario["title"],
          "project": scenario["repository"].removesuffix(".git").rsplit("/", 1)[-1],
          "caption": "Fresh native Pi base session; selected genuine resample.", "path": "./",
          "userTurns": source["outcome"]["userTurns"], "checksPassed": validation["allChecksPassed"]}]})
    for name in ("index.html", "app.js", "style.css", "favicon.svg"):
        shutil.copyfile(ASSETS / name, output / name)
    shutil.copyfile(Path(__file__), output / "export-review.py")
    (output / "robots.txt").write_text("User-agent: *\nDisallow: /\n")
    credential_scan = scan_credentials(output)
    # Recheck the outcome, which binds all other raw artifacts, before finalizing.
    require(sha(source["source"] / "outcome.json") == source["hashes"]["outcome.json"],
            "Source outcome changed during export")
    with tarfile.open(output / "transcript-evidence.tar.gz", "w:gz") as archive:
        for name in ("evidence", "files", "validation.json", "explainer.json", "files-receipt.json",
                     "selection.json", "reasoning-verification.json", "source-artifact-hashes.json",
                     "corpus.json", "export-review.py"):
            if (output / name).exists():
                archive.add(output / name, arcname=name)
    receipt = {"sourceMessagesSha256": source["hashes"]["messages.json"],
               "sourcePatchSha256": source["hashes"]["changes.patch"], "fixture": fixture,
               "sourceHashesVerified": True, "nativeSessionMatches": True,
               "userMessagesMatchInputs": True, "observedHistoryPrefixesVerified": True,
               "allHttpUserPromptPrefixesVerified": True, "allPriorReasoningPreserved": True,
               "rawEvidenceBytesUnchanged": True, "exportedArtifacts": len(copied),
               "omittedWorkspaceSnapshots": source["snapshots"], "credentialScan": credential_scan,
               "selectionConditioned": True, "attempts": len(selection["attempts"]),
               "selectedAttempt": selection["selectedAttempt"], "checksPassed": validation["allChecksPassed"],
               "bundleSha256": sha(output / "transcript-evidence.tar.gz")}
    write(output / "export-receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("case-dir", "validation", "explainer", "files-json", "selection", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    receipt = export_review(args.case_dir, args.validation, args.explainer, args.files_json,
                            args.selection, args.output)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
