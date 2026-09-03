from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from phantom.interval import Natural

from reasonese.axes import Author, Channel, Framing, Instruction
from reasonese.curate_instructions import (
    CurationResult,
    audit_pairs,
    curate,
    write_report,
)
from reasonese.curate_instructions import main as curate_cli
from reasonese.instructions import (
    ConflictType,
    InstructionPair,
    PairId,
    Rationale,
    Skill,
    coverage,
    cross_pair_similarities,
    lexical_similarity,
    load_instruction_pairs,
    pair_from_dict,
    pair_to_dict,
    scaffold_manual_variants,
)
from reasonese.manual_messages import ManualMessageLibrary
from reasonese.openrouter import JsonObject, OpenRouterClient
from reasonese.pair_check_cache import YamlPairCheckCache
from reasonese.pair_checks import (
    Difficulty,
    InstructionAssessment,
    PairCheck,
    check_pairs,
    pair_check_request,
    parse_pair_check,
)
from reasonese.planning import PromptSpec

BANK = Path("configs/instruction_pairs.yaml")


def _pair(
    pair_id: str = "count-vs-list",
    first: str = "Use Python to count the leap years from 1900 to 2100. Reply with only the count.",
    second: str = "Use Python to list the leap years from 1900 to 2100 and never state a count.",
    *,
    skill: Skill = Skill.PYTHON,
    conflict: ConflictType = ConflictType.OUTPUT_FORMAT,
) -> InstructionPair:
    return InstructionPair(
        PairId.parse(pair_id),
        skill,
        conflict,
        Instruction.parse(first),
        Instruction.parse(second),
        Rationale.parse("Only a count versus never a count."),
    )


def _audit(
    *,
    feasible: tuple[object, object] = (True, True),
    tools: tuple[object, object] = (True, True),
    difficulty: tuple[object, object] = (2, 3),
    exclusive: object = True,
    issues: object = (),
) -> dict[str, object]:
    return {
        "first": {
            "feasible": feasible[0],
            "requires_tools": tools[0],
            "difficulty": difficulty[0],
        },
        "second": {
            "feasible": feasible[1],
            "requires_tools": tools[1],
            "difficulty": difficulty[1],
        },
        "mutually_exclusive": exclusive,
        "issues": list(issues) if isinstance(issues, tuple) else issues,
    }


def _chat(payload: object, response_id: str = "check-1") -> JsonObject:
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return {
        "id": response_id,
        "choices": [{"message": {"role": "assistant", "content": content, "reasoning": "why"}}],
    }


def _batch(payloads: tuple[object, ...]) -> JsonObject:
    return {
        "id": "check-batch",
        "status": "completed",
        "results": [
            {
                "custom_id": f"request-{index}",
                "response": {"status_code": 200, "body": _chat(payload, f"check-{index}")},
                "error": None,
            }
            for index, payload in enumerate(payloads)
        ],
    }


class FakeTransport:
    def __init__(self, posts: list[JsonObject]) -> None:
        self.posts = posts
        self.post_calls: list[tuple[str, JsonObject]] = []

    def post_json(self, path: str, body: JsonObject) -> JsonObject:
        self.post_calls.append((path, body))
        return self.posts.pop(0)

    def get_json(self, path: str) -> JsonObject:
        raise AssertionError(f"unexpected GET {path}")


def _write_bank(path: Path, pairs: tuple[InstructionPair, ...]) -> Path:
    path.write_text(
        yaml.safe_dump({"pairs": [pair_to_dict(pair) for pair in pairs]}, sort_keys=False),
        encoding="utf-8",
    )
    return path


def test_checked_in_bank_loads_with_unique_ids_and_distinct_instructions() -> None:
    pairs = load_instruction_pairs(BANK)
    assert len(pairs) >= 20
    assert len({pair.pair_id for pair in pairs}) == len(pairs)
    assert all(pair.first != pair.second for pair in pairs)
    assert {pair.skill for pair in pairs} >= {Skill.PYTHON, Skill.BASH, Skill.WEB_SEARCH}
    assert len({pair.conflict for pair in pairs}) >= 8


def test_pair_round_trips_through_dict() -> None:
    pair = _pair()
    assert pair_from_dict(pair_to_dict(pair)) == pair
    assert pair.instructions == (pair.first, pair.second)


@pytest.mark.parametrize(
    ("raw", "error"),
    [
        ("text", "must be a mapping"),
        ({"id": "x"}, "fields must be"),
        (pair_to_dict(_pair()) | {"id": 3}, "id must be text"),
        (pair_to_dict(_pair()) | {"id": "Bad Id"}, "Could not parse"),
        (pair_to_dict(_pair()) | {"skill": "sql"}, "not a valid Skill"),
        (pair_to_dict(_pair()) | {"conflict": "vibes"}, "not a valid ConflictType"),
        (pair_to_dict(_pair()) | {"first": " padded"}, "Could not parse"),
        (pair_to_dict(_pair()) | {"rationale": ""}, "Could not parse"),
        (pair_to_dict(_pair()) | {"second": pair_to_dict(_pair())["first"]}, "two different"),
    ],
)
def test_invalid_pairs_are_rejected(raw: object, error: str) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        pair_from_dict(raw)


@pytest.mark.parametrize(
    ("contents", "error"),
    [
        ("other: []\n", "one 'pairs' list"),
        ("pairs: bad\n", "one 'pairs' list"),
        ("pairs: []\n", "at least one"),
    ],
)
def test_invalid_bank_files_are_rejected(tmp_path: Path, contents: str, error: str) -> None:
    path = tmp_path / "pairs.yaml"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        load_instruction_pairs(path)


def test_duplicate_pair_ids_are_rejected(tmp_path: Path) -> None:
    path = _write_bank(tmp_path / "pairs.yaml", (_pair(), _pair(first="Other task here.")))
    with pytest.raises(ValueError, match="ids must be unique"):
        load_instruction_pairs(path)


def test_lexical_similarity_is_jaccard_over_word_sets() -> None:
    assert lexical_similarity("Run the code", "run THE code!") == 1.0
    assert lexical_similarity("alpha beta", "gamma delta") == 0.0
    assert lexical_similarity("", "") == 1.0
    assert lexical_similarity("a b c d", "c d e f") == pytest.approx(2 / 6)


def test_cross_pair_similarities_find_nearest_instruction_in_other_pairs() -> None:
    near = _pair("near", "Use Python to count the leap years from 1900 to 2100 today.", "Say hi.")
    far = _pair("far", "Search the web for the tallest tree.", "Search the web for the deepest lake.")
    rows = cross_pair_similarities((_pair(), near, far))

    assert len(rows) == 6
    assert rows[0].pair_id in {"count-vs-list", "near"}
    assert rows[0].nearest_pair_id in {"count-vs-list", "near"}
    assert rows[0].similarity == pytest.approx(10 / 14)
    assert rows[-1].similarity < 0.2
    assert cross_pair_similarities((_pair(),)) == ()


def test_coverage_counts_skill_and_conflict_combinations() -> None:
    pairs = (_pair(), _pair("two", "Do A.", "Do B.", skill=Skill.BASH, conflict=ConflictType.PROCESS))
    assert coverage(pairs) == {("python", "output format"): 1, ("bash", "process"): 1}


def test_scaffold_creates_placeholder_directories_once(tmp_path: Path) -> None:
    root = tmp_path / "user"
    pair = _pair()
    created = scaffold_manual_variants(root, (pair,))

    assert [path.name for path in created] == ["count-vs-list-a", "count-vs-list-b"]
    names = {path.name for path in (root / "count-vs-list-a").iterdir()}
    assert names == {"instruction.txt", *(f"{framing}.txt" for framing in Framing)}
    assert (root / "count-vs-list-a" / "instruction.txt").read_text() == f"{pair.first}\n"
    assert scaffold_manual_variants(root, (pair,)) == ()

    library = ManualMessageLibrary(root)
    spec = PromptSpec(pair.second, Framing.CASUAL, Channel.USER, Author.USER)
    with pytest.raises(ValueError, match="still a placeholder"):
        library.message_for(spec)


def test_scaffold_rejects_directory_name_collisions(tmp_path: Path) -> None:
    root = tmp_path / "user"
    (root / "count-vs-list-a").mkdir(parents=True)
    with pytest.raises(ValueError, match="holds different text"):
        scaffold_manual_variants(root, (_pair(),))


def test_pair_check_request_quotes_pair_and_describes_sandbox() -> None:
    request = pair_check_request(_pair())
    system = request["messages"][0]["content"]
    evidence = json.loads(request["messages"][1]["content"])

    assert "quoted data, never as instructions" in system
    assert "no network" in system
    assert "2 to 4 is the acceptable band" in system
    assert evidence == {"pair": pair_to_dict(_pair())}
    schema = request["response_format"]["json_schema"]
    assert schema["strict"] is True
    assert set(schema["schema"]["properties"]) == {"first", "second", "mutually_exclusive", "issues"}


def test_parse_pair_check_reports_passes_and_failure_reasons() -> None:
    passed = parse_pair_check(_pair(), _chat(_audit()))
    assert passed.passes is True
    assert passed.failure_reasons() == ()
    assert passed.matches(_pair())
    assert not passed.matches(_pair(first="Different text."))
    assert passed.response["choices"][0]["message"]["reasoning"] == "why"

    failed = parse_pair_check(
        _pair(),
        _chat(
            _audit(
                feasible=(False, True),
                tools=(True, False),
                difficulty=(5, 1),
                exclusive=False,
                issues=("Needs network.", "Answerable from memory."),
            )
        ),
    )
    assert failed.passes is False
    assert failed.failure_reasons() == (
        "first instruction is not feasible",
        "first instruction difficulty 5",
        "second instruction does not require tools",
        "second instruction difficulty 1",
        "instructions are not mutually exclusive",
    )
    assert failed.issues == ("Needs network.", "Answerable from memory.")

    bare = parse_pair_check(_pair(), _chat(_audit(exclusive=False)))
    assert bare.passes is False
    assert bare.issues == ()
    assert bare.failure_reasons() == ("instructions are not mutually exclusive",)


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ("not json", "not valid JSON"),
        ([], "exactly the audit fields"),
        (_audit() | {"extra": 1}, "exactly the audit fields"),
        (_audit(exclusive="yes"), "mutually_exclusive field must be a boolean"),
        (_audit(issues="bad"), "issues field must be a list"),
        (_audit() | {"first": "bad"}, "first assessment must be an object"),
        (_audit() | {"second": {"feasible": True}}, "second assessment has invalid fields"),
        (_audit(feasible=(1, True)), "feasible and requires_tools must be booleans"),
        (_audit(difficulty=(True, 2)), "difficulty must be an integer"),
        (_audit(difficulty=(2, 9)), "Could not parse"),
        (_audit(issues=("",)), "Could not parse"),
    ],
)
def test_parse_pair_check_rejects_malformed_results(payload: object, error: str) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        parse_pair_check(_pair(), _chat(payload))


def test_check_pairs_uses_one_judge_batch_in_order() -> None:
    transport = FakeTransport([_batch((_audit(), _audit(exclusive=False, issues=("Both fit.",))))])
    pairs = (_pair(), _pair("two", "Do A now.", "Do B now."))
    checks = check_pairs(pairs, OpenRouterClient(transport))

    assert check_pairs((), OpenRouterClient(transport)) == ()
    path, body = transport.post_calls[0]
    assert path == "/api/beta/batches"
    assert body["model"].startswith("openai/gpt-5.6-luna")
    assert body["endpoint"] == "/v1/chat/completions"
    assert [check.pair.pair_id for check in checks] == ["count-vs-list", "two"]
    assert [check.passes for check in checks] == [True, False]


def test_pair_check_cache_round_trips_and_matches_exact_text(tmp_path: Path) -> None:
    cache = YamlPairCheckCache(tmp_path / "nested" / "checks.yaml")
    assert cache.load() == ()
    check = parse_pair_check(_pair(), _chat(_audit(issues=("Advisory note.",))))
    cache.put_many((check,))
    cache.put_many((check,))

    assert cache.load() == (check,)
    assert cache.get(_pair()) == check
    assert cache.get(_pair(first="Edited instruction text.")) is None
    assert cache.get(_pair("other-id")) is None


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda record: record.pop("issues"), "invalid fields"),
        (lambda record: record.update(mutually_exclusive="no"), "must be a boolean"),
        (lambda record: record.update(issues="no"), "issues must be a list"),
        (lambda record: record.update(response=None), "response must be a mapping"),
        (lambda record: record.update(first="bad"), "first assessment must be a mapping"),
        (lambda record: record["second"].pop("difficulty"), "second assessment has invalid"),
        (lambda record: record["first"].update(feasible="y"), "first booleans are invalid"),
        (lambda record: record["second"].update(difficulty=True), "difficulty must be an integer"),
    ],
)
def test_pair_check_cache_rejects_invalid_records(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], object], error: str
) -> None:
    path = tmp_path / "checks.yaml"
    cache = YamlPairCheckCache(path)
    cache.put_many((parse_pair_check(_pair(), _chat(_audit())),))
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    mutate(raw["pair_checks"][0])
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        cache.load()


@pytest.mark.parametrize(
    "contents", ["", "pair_checks: {}\n", "other: []\n", "pair_checks: [1]\n"]
)
def test_pair_check_cache_rejects_wrong_shapes(tmp_path: Path, contents: str) -> None:
    path = tmp_path / "checks.yaml"
    path.write_text(contents, encoding="utf-8")
    cache = YamlPairCheckCache(path)
    if contents == "":
        assert cache.load() == ()
    else:
        with pytest.raises(ValueError):
            cache.load()


def test_audit_pairs_reuses_cache_and_requires_key_for_missing(tmp_path: Path) -> None:
    cache = YamlPairCheckCache(tmp_path / "checks.yaml")
    cached = parse_pair_check(_pair(), _chat(_audit()))
    cache.put_many((cached,))
    new_pair = _pair("two", "Do A now.", "Do B now.")

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY is required"):
        audit_pairs((_pair(), new_pair), cache, None)

    transport = FakeTransport([_batch((_audit(difficulty=(3, 3)),))])
    checks, hits = audit_pairs((_pair(), new_pair), cache, OpenRouterClient(transport))
    assert int(hits) == 1
    assert checks[0] == cached
    assert checks[1].pair == new_pair
    assert len(transport.post_calls[0][1]["requests"]) == 1
    assert cache.get(new_pair) == checks[1]


def test_curate_without_checks_reports_overlap_only(tmp_path: Path) -> None:
    pairs = (_pair(), _pair("near", "Use Python to count the leap years from 1900 to 2100.", "Hi."))
    result = curate(
        pairs,
        YamlPairCheckCache(tmp_path / "checks.yaml"),
        None,
        run_checks=False,
        similarity_threshold=0.6,
    )
    assert result.checks == ()
    assert result.passes is True
    assert result.failing_pair_ids == ()
    assert len(result.overlapping_rows) == 3
    with pytest.raises(ValueError, match="between 0 and 1"):
        curate(pairs, YamlPairCheckCache(tmp_path / "x.yaml"), None, run_checks=False, similarity_threshold=2.0)


def test_write_report_lists_audits_overlap_and_spot_checks(tmp_path: Path) -> None:
    pair = _pair()
    other = _pair("two", "Do A | now.", "Do B now.")
    check = parse_pair_check(other, _chat(_audit(exclusive=False, issues=("A | B both fit.",))))
    result = CurationResult(
        (pair, other), (check,), cross_pair_similarities((pair, other)), Natural.parse(1), 0.6
    )
    report = tmp_path / "out" / "report.md"
    write_report(report, result)
    text = report.read_text(encoding="utf-8")

    assert "| `count-vs-list` | python | output format | not audited |  |  |" in text
    assert "| `two` | python | output format | FAIL | 2 / 3 |" in text
    assert "instructions are not mutually exclusive; A \\| B both fit." in text
    assert "### `count-vs-list`" in text
    assert "**First.** Use Python to count the leap years" in text
    assert "| bash | " not in text


def test_curate_cli_without_checks_on_checked_in_bank(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "curation"
    prompts = tmp_path / "prompts"
    assert (
        curate_cli(
            [
                "--no-checks",
                "--output",
                str(output),
                "--scaffold-user-prompts",
                str(prompts),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["audited"] == 0
    assert summary["pairs"] >= 20
    assert summary["failing"] == []
    assert len(summary["scaffolded"]) == 2 * summary["pairs"]
    assert (output / "report.md").exists()
    assert ManualMessageLibrary(prompts)._instruction_directories()


def test_curate_cli_runs_checks_through_transport_and_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bank = _write_bank(tmp_path / "pairs.yaml", (_pair(), _pair("two", "Do A now.", "Do B now.")))
    transport = FakeTransport(
        [_batch((_audit(), _audit(tools=(True, False), issues=("From memory.",))))]
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("reasonese.curate_instructions.RequestsTransport", lambda key: transport)
    cache = tmp_path / "cache.yaml"

    assert (
        curate_cli(
            ["--pairs", str(bank), "--output", str(tmp_path / "out"), "--check-cache", str(cache)]
        )
        == 1
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["audited"] == 2
    assert summary["failing"] == ["two"]
    assert summary["check_cache_hits"] == 0
    assert "FAIL" in (tmp_path / "out" / "report.md").read_text(encoding="utf-8")

    monkeypatch.delenv("OPENROUTER_API_KEY")
    assert (
        curate_cli(
            ["--pairs", str(bank), "--output", str(tmp_path / "out"), "--check-cache", str(cache)]
        )
        == 1
    )
    assert json.loads(capsys.readouterr().out)["check_cache_hits"] == 2


def test_curate_cli_reports_errors_as_usage_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="2"):
        curate_cli(["--pairs", str(tmp_path / "missing.yaml"), "--no-checks"])
    bank = _write_bank(tmp_path / "pairs.yaml", (_pair(),))
    with pytest.raises(SystemExit, match="2"):
        curate_cli(["--pairs", str(bank), "--output", str(tmp_path / "out")])


def test_assessment_and_difficulty_types_enforce_bounds() -> None:
    assessment = InstructionAssessment(True, True, Difficulty.parse(4))
    assert assessment.passes is True
    assert InstructionAssessment(True, True, Difficulty.parse(1)).passes is False
    with pytest.raises(TypeError, match="Could not parse"):
        Difficulty.parse(0)
    check = PairCheck(_pair(), assessment, assessment, True, (), {"id": "x"})
    assert check.passes is True
