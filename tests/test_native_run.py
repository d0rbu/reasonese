from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from reasonese.axes import Author, Channel, Framing, Instruction
from reasonese.matchup import prompt_spec_to_dict
from reasonese.native_run import main, run_native
from reasonese.openrouter import JsonObject
from reasonese.planning import PromptSpec


def _spec(instruction: str) -> dict[str, str]:
    return prompt_spec_to_dict(
        PromptSpec(
            Instruction.parse(instruction),
            Framing.NORMAL,
            Channel.USER,
            Author.INKLING,
        )
    )


def _plan(path: Path, *, historical: bool = False) -> dict[str, Any]:
    inputs = [
        {"spec": _spec("Use UTC."), "content": "Please implement UTC handling."},
        {"spec": _spec("Use local time."), "content": "Please implement local time handling."},
    ]
    plan = {
        "kind": "pi_experiment_plan",
        "containsPlaceholders": False,
        "prefix": (
            [{"role": "user", "content": [{"type": "text", "text": inputs[0]["content"]}]}]
            if historical
            else []
        ),
        "turns": [{"prompt": item["content"]} for item in inputs[int(historical) :]],
        "metadata": {
            "authoringInputs": inputs,
            "historicalPlacements": ([{"position": 1, "message": 0}] if historical else []),
        },
    }
    path.write_text(json.dumps(plan) + "\n")
    return plan


class FakeTransport:
    def __init__(self, *, complies: bool = True, mutate: Path | None = None) -> None:
        self.complies = complies
        self.mutate = mutate
        self.calls: list[tuple[str, JsonObject]] = []

    def post_json(self, path: str, body: JsonObject) -> JsonObject:
        self.calls.append((path, body))
        if self.mutate:
            self.mutate.write_bytes(self.mutate.read_bytes() + b" ")
        return {
            "id": "qa-batch",
            "status": "completed",
            "results": [
                {
                    "custom_id": f"request-{i}",
                    "response": {
                        "status_code": 200,
                        "body": {
                            "id": f"qa-{i}",
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": json.dumps(
                                            {
                                                "complies": self.complies,
                                                "issues": []
                                                if self.complies
                                                else ["Lost the UTC requirement."],
                                            }
                                        ),
                                    }
                                }
                            ],
                        },
                    },
                    "error": None,
                }
                for i in range(len(body["requests"]))
            ],
        }

    def get_json(self, path: str) -> JsonObject:
        raise AssertionError(f"unexpected GET {path}")


@pytest.mark.parametrize("historical", [False, True])
def test_native_run_audits_exact_turns_then_launches_and_reuses_cache(
    tmp_path: Path,
    historical: bool,
) -> None:
    path = tmp_path / "plan.json"
    _plan(path, historical=historical)
    output = tmp_path / "run"
    cache = tmp_path / "qa.yaml"
    transport = FakeTransport()
    launches: list[Sequence[str]] = []

    def factory() -> FakeTransport:
        assert len(launches) == 1 and "--preflight-only" in launches[0]
        return transport

    receipt_path = run_native(
        path, output, cache, allow_paid=True, transport_factory=factory, launcher=launches.append
    )
    receipt = json.loads(receipt_path.read_text())
    assert receipt["fixture"] is True
    assert receipt["passed"] is True
    assert receipt["planSha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert receipt["cacheHits"] == 0
    assert len(receipt["inputs"]) == 2
    assert all(item["judgeResponse"] for item in receipt["inputs"])
    assert not cache.exists()
    assert Path(receipt["qaCache"]).name == "qa.fixture.yaml"
    assert Path(receipt["qaCache"]).is_file()
    assert not output.exists()
    assert len(transport.calls) == 1
    assert launches[1][-1] == "--allow-paid"
    assert launches[1][launches[1].index("--qa-receipt") + 1] == str(receipt_path)
    launches.clear()
    second_receipt = run_native(
        path,
        tmp_path / "second-run",
        cache,
        allow_paid=True,
        transport_factory=lambda: None,
        launcher=launches.append,
    )
    assert json.loads(second_receipt.read_text())["cacheHits"] == 2
    assert json.loads(receipt_path.read_text())["cacheHits"] == 0
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    "change,match",
    [
        ("flag", "placeholders"),
        ("todo", "TODO"),
        ("detached", "exact delivered"),
        ("missing", "authoringInputs"),
        ("bad-channel", "user message"),
        ("bad-prefix", "prefix"),
        ("empty-turns", "pending"),
        ("bad-placement", "consecutive"),
        ("outside", "outside"),
        ("tool-placement", "one user text"),
        ("image-placement", "must be text"),
    ],
)
def test_native_run_rejects_bad_inputs_before_provider(
    tmp_path: Path,
    change: str,
    match: str,
) -> None:
    path = tmp_path / "plan.json"
    plan = _plan(path, historical=True)
    match change:
        case "flag":
            plan["containsPlaceholders"] = True
        case "todo":
            plan["metadata"]["authoringInputs"][1]["content"] = "TODO: write me"
        case "detached":
            plan["turns"][0]["prompt"] = "Something else."
        case "missing":
            plan["metadata"].pop("authoringInputs")
        case "bad-channel":
            plan["metadata"]["authoringInputs"][0]["spec"]["channel"] = "system prompt"
        case "bad-prefix":
            plan["prefix"] = None
        case "empty-turns":
            plan["turns"] = []
        case "bad-placement":
            plan["metadata"]["historicalPlacements"][0]["position"] = 2
        case "outside":
            plan["metadata"]["historicalPlacements"][0]["message"] = 8
        case "tool-placement":
            plan["prefix"][0]["role"] = "toolResult"
        case "image-placement":
            plan["prefix"][0]["content"][0]["type"] = "image"
    path.write_text(json.dumps(plan))

    def forbidden() -> None:
        pytest.fail("provider constructed for invalid inputs")

    with pytest.raises(ValueError, match=match):
        run_native(
            path,
            tmp_path / "run",
            tmp_path / "cache",
            allow_paid=True,
            transport_factory=forbidden,
            launcher=lambda _: pytest.fail("launched"),
        )


def test_native_run_requires_opt_in_and_fresh_output_before_provider(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    _plan(path)
    output = tmp_path / "run"
    for allow_paid in (False, True):
        if allow_paid:
            output.mkdir()
        with pytest.raises(ValueError, match="fresh" if allow_paid else "--allow-paid"):
            run_native(
                path,
                output,
                tmp_path / "cache",
                allow_paid=allow_paid,
                transport_factory=lambda: pytest.fail("provider constructed"),
            )


@pytest.mark.parametrize("mutate", [False, True])
def test_failed_qa_or_mutated_plan_keeps_receipt_without_continuation(
    tmp_path: Path,
    mutate: bool,
) -> None:
    path = tmp_path / "plan.json"
    _plan(path)
    launches: list[Sequence[str]] = []
    transport = FakeTransport(complies=mutate, mutate=path if mutate else None)
    with pytest.raises(ValueError, match="changed during" if mutate else "QA failed"):
        run_native(
            path,
            tmp_path / "run",
            tmp_path / "cache",
            allow_paid=True,
            transport_factory=lambda: transport,
            launcher=launches.append,
        )
    assert len(launches) == 1
    receipt = json.loads(next(tmp_path.glob("run.message-qa.*.json")).read_text())
    assert receipt["passed"] is mutate


def test_native_preflight_failure_prevents_qa(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    _plan(path)

    def rejected(command: Sequence[str]) -> None:
        raise subprocess.CalledProcessError(1, command)

    with pytest.raises(subprocess.CalledProcessError):
        run_native(
            path,
            tmp_path / "run",
            tmp_path / "cache",
            allow_paid=True,
            transport_factory=lambda: pytest.fail("provider constructed"),
            launcher=rejected,
        )


def test_native_run_cli_reports_success_and_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import reasonese.native_run as module

    receipt = tmp_path / "qa.json"
    monkeypatch.setattr(module, "run_native", lambda *args, **kwargs: receipt)
    assert main(["--plan", "plan.json", "--out", str(tmp_path / "run"), "--allow-paid"]) == 0
    assert json.loads(capsys.readouterr().out)["qaReceipt"] == str(receipt)

    def fail(*args: object, **kwargs: object) -> None:
        raise ValueError("invalid plan")

    monkeypatch.setattr(module, "run_native", fail)
    with pytest.raises(SystemExit, match="2"):
        main(["--plan", "plan.json", "--out", "out/run"])
    assert "invalid plan" in capsys.readouterr().err


def test_dangling_failed_output_alias_is_retained_before_provider(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    _plan(path)
    output = tmp_path / "run"
    output.symlink_to(tmp_path / "missing-backing", target_is_directory=True)
    def forbidden(*args):
        pytest.fail("provider or launcher called for existing failed alias")
    with pytest.raises(ValueError, match="fresh"):
        run_native(path, output, tmp_path / "qa.yaml", allow_paid=True,
                   transport_factory=forbidden, launcher=forbidden)
    assert output.is_symlink()


def test_existing_qa_receipt_is_retained_before_any_provider_or_preflight(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    _plan(path, historical=False)
    plan_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = tmp_path / f"run.message-qa.{plan_sha}.json"
    receipt.write_text("failed prior QA evidence")
    with pytest.raises(ValueError, match="QA receipt already exists"):
        run_native(path, tmp_path / "run", tmp_path / "qa.yaml", allow_paid=True,
                   transport_factory=lambda: pytest.fail("provider constructed"),
                   launcher=lambda _: pytest.fail("preflight dispatched"))
    assert receipt.read_text() == "failed prior QA evidence"
