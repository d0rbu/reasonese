"""Tests for the blinded manual-variant editor.

The blinding is the point of the tool, so most of these assert what the
interface refuses to reveal rather than what it renders.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Iterator
from functools import cache
from pathlib import Path

import pytest
from phantom.interval import Natural

from reasonese.axes import Author, Channel, Framing, Instruction, author_framings
from reasonese.conversation import AUTHORING_RULE, authoring_instructions, framing_guidance
from reasonese.instructions import (
    ConflictType,
    InstructionPair,
    PairId,
    Rationale,
    Skill,
    load_instruction_pairs,
    scaffold_manual_variants,
)
from reasonese.planning import PromptSpec
from reasonese.write_variants import (
    PLACEHOLDER_PREFIX,
    AuthoringSession,
    VariantTask,
    build_queue,
    build_server,
    guidance_for,
    is_written,
    main,
    new_token,
    open_tunnel,
    variant_path,
)

BANK = Path("configs/instruction_pairs.yaml")
MANUAL_ROOT = Path("prompts/user")


@cache
def _pairs() -> tuple[InstructionPair, ...]:
    return load_instruction_pairs(BANK)


def _small_pairs() -> tuple[InstructionPair, ...]:
    return tuple(
        InstructionPair(
            PairId.parse(f"probe-{number}"),
            Skill.PYTHON,
            ConflictType.OUTPUT_FORMAT,
            Instruction.parse(f"Do thing {number} one way."),
            Instruction.parse(f"Do thing {number} the other way."),
            Rationale.parse("The two requests cannot both be satisfied."),
        )
        for number in range(4)
    )


@pytest.fixture
def scaffolded(tmp_path: Path) -> tuple[Path, tuple[InstructionPair, ...]]:
    pairs = _small_pairs()
    root = tmp_path / "user"
    scaffold_manual_variants(root, pairs)
    return root, pairs


def _session(root: Path, pairs: tuple[InstructionPair, ...]) -> AuthoringSession:
    return AuthoringSession(root, build_queue(root, pairs, Natural.parse(0)), set())


# --------------------------------------------------------------------------
# Queue construction and anti-anchoring layout
# --------------------------------------------------------------------------


def test_queue_covers_every_instruction_and_framing_once() -> None:
    queue = build_queue(MANUAL_ROOT, _pairs(), Natural.parse(0))
    manual = author_framings(Author.USER)

    assert len(queue) == 48 * len(manual) == 144
    assert len({(task.slug, task.framing) for task in queue}) == len(queue)
    assert dict(Counter(task.framing for task in queue)) == dict.fromkeys(manual, 48)
    assert {task.instruction for task in queue} == {
        instruction for pair in _pairs() for instruction in pair.instructions
    }


@pytest.mark.parametrize("seed", [0, 1, 7, 99])
def test_an_instruction_reappears_exactly_one_round_later(seed: int) -> None:
    """A plain shuffle once put two framings five apart; this must not."""
    queue = build_queue(MANUAL_ROOT, _pairs(), Natural.parse(seed))
    instructions = len({task.instruction for task in queue})

    positions: dict[str, list[int]] = {}
    for index, task in enumerate(queue):
        positions.setdefault(str(task.instruction), []).append(index)

    for places in positions.values():
        assert len(places) == len(author_framings(Author.USER))
        gaps = [second - first for first, second in zip(places, places[1:], strict=False)]
        assert set(gaps) == {instructions}


def test_every_round_mixes_framings_rather_than_blocking_one() -> None:
    queue = build_queue(MANUAL_ROOT, _pairs(), Natural.parse(0))
    instructions = len({task.instruction for task in queue})

    for start in range(0, len(queue), instructions):
        framings = Counter(task.framing for task in queue[start : start + instructions])
        assert len(framings) == len(author_framings(Author.USER))
        # No framing may dominate a round.
        assert max(framings.values()) < instructions


def test_queue_is_reproducible_and_seed_sensitive(
    scaffolded: tuple[Path, tuple[InstructionPair, ...]],
) -> None:
    root, pairs = scaffolded
    baseline = build_queue(root, pairs, Natural.parse(3))

    assert build_queue(root, pairs, Natural.parse(3)) == baseline
    assert build_queue(root, pairs, Natural.parse(4)) != baseline


def test_queue_rejects_a_missing_or_ambiguous_directory(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no manual variant directory exists"):
        build_queue(empty, _small_pairs(), Natural.parse(0))

    with pytest.raises(ValueError, match="does not exist"):
        build_queue(tmp_path / "absent", _small_pairs(), Natural.parse(0))

    pairs = _small_pairs()
    duplicated = tmp_path / "duplicated"
    scaffold_manual_variants(duplicated, pairs)
    extra = duplicated / "copy"
    extra.mkdir()
    (extra / "instruction.txt").write_text(f"{pairs[0].first}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="same instruction"):
        build_queue(duplicated, pairs, Natural.parse(0))


# --------------------------------------------------------------------------
# Blinding
# --------------------------------------------------------------------------


def test_state_never_reveals_the_pair_side_or_conflict(
    scaffolded: tuple[Path, tuple[InstructionPair, ...]],
) -> None:
    root, pairs = scaffolded
    session = _session(root, pairs)
    state = session.state()

    assert set(state) == {
        "done",
        "index",
        "written",
        "total",
        "framing",
        "guidance",
        "instruction",
    }
    task = session.queue[state["index"]]
    rendered = json.dumps(state)
    # The slug carries the pair id and the side, so it must not appear at all.
    assert task.slug not in rendered
    assert str(task.instruction) in rendered
    for pair in pairs:
        assert str(pair.pair_id) not in rendered
        assert str(pair.conflict) not in rendered
        assert str(pair.rationale) not in rendered
    partner = next(
        pair.second if pair.first == task.instruction else pair.first
        for pair in pairs
        if task.instruction in pair.instructions
    )
    assert str(partner) not in rendered


def test_guidance_is_exactly_what_a_model_author_receives() -> None:
    for framing in author_framings(Author.USER):
        guidance = guidance_for(framing)
        assert framing_guidance(framing) in guidance
        assert AUTHORING_RULE in guidance
        # The same two blocks appear verbatim in the model-author prompt.
        spec = PromptSpec(
            Instruction.parse("Do the thing."), framing, Channel.USER, Author.INKLING
        )
        prompt = authoring_instructions(spec)
        assert framing_guidance(framing) in prompt
        assert AUTHORING_RULE in prompt


# --------------------------------------------------------------------------
# Reading and writing variants
# --------------------------------------------------------------------------


def test_placeholders_count_as_unwritten_until_saved(
    scaffolded: tuple[Path, tuple[InstructionPair, ...]],
) -> None:
    root, pairs = scaffolded
    session = _session(root, pairs)
    total = len(session.queue)

    assert session.written() == 0
    assert variant_path(root, session.queue[0]).read_text().startswith(PLACEHOLDER_PREFIX)

    index = session.current()
    assert index is not None
    path = session.save(index, "  please do the thing  ")

    assert path.read_text(encoding="utf-8") == "please do the thing\n"
    assert is_written(root, session.queue[index])
    assert session.written() == 1
    assert session.state()["total"] == total
    assert session.current() != index


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("", "cannot be empty"),
        ("   \n ", "cannot be empty"),
        (f"{PLACEHOLDER_PREFIX} write it later", "placeholder prefix"),
    ],
)
def test_save_rejects_empty_and_placeholder_text(
    scaffolded: tuple[Path, tuple[InstructionPair, ...]], text: str, error: str
) -> None:
    root, pairs = scaffolded
    session = _session(root, pairs)
    with pytest.raises(ValueError, match=error):
        session.save(0, text)


@pytest.mark.parametrize("index", [-1, 10_000])
def test_save_and_skip_reject_an_unknown_index(
    scaffolded: tuple[Path, tuple[InstructionPair, ...]], index: int
) -> None:
    root, pairs = scaffolded
    session = _session(root, pairs)
    with pytest.raises(ValueError, match="unknown task"):
        session.save(index, "text")
    with pytest.raises(ValueError, match="unknown task"):
        session.skip(index)


def test_skipping_defers_a_task_and_saving_restores_it(
    scaffolded: tuple[Path, tuple[InstructionPair, ...]],
) -> None:
    root, pairs = scaffolded
    session = _session(root, pairs)
    first = session.current()
    assert first is not None

    session.skip(first)
    second = session.current()
    assert second is not None and second != first

    session.save(first, "written after all")
    assert first not in session.skipped
    assert session.current() == second


def test_finishing_every_task_reports_done(
    scaffolded: tuple[Path, tuple[InstructionPair, ...]],
) -> None:
    root, pairs = scaffolded
    session = _session(root, pairs)
    for index in range(len(session.queue)):
        session.save(index, f"variant number {index}")

    assert session.current() is None
    state = session.state()
    assert state == {
        "done": True,
        "written": len(session.queue),
        "total": len(session.queue),
        "skipped": 0,
    }


def test_a_skipped_but_unwritten_task_is_reported_as_outstanding(
    scaffolded: tuple[Path, tuple[InstructionPair, ...]],
) -> None:
    root, pairs = scaffolded
    session = _session(root, pairs)
    for index in range(1, len(session.queue)):
        session.save(index, f"variant number {index}")
    session.skip(0)

    state = session.state()
    assert state["done"] is True
    assert state["skipped"] == 1
    assert state["written"] == len(session.queue) - 1


# --------------------------------------------------------------------------
# The HTTP surface
# --------------------------------------------------------------------------


@pytest.fixture
def served(
    scaffolded: tuple[Path, tuple[InstructionPair, ...]],
) -> Iterator[tuple[str, str, AuthoringSession]]:
    root, pairs = scaffolded
    session = _session(root, pairs)
    token = new_token()
    server = build_server(session, token, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", token, session
    finally:
        server.shutdown()
        server.server_close()


def _request(url: str, token: str | None, payload: dict[str, object] | None = None):
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=body, method="POST" if body else "GET")
    if token is not None:
        request.add_header("X-Auth-Token", token)
    request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
        return response.status, response.read()


def test_the_page_and_api_require_the_run_token(
    served: tuple[str, str, AuthoringSession],
) -> None:
    base, token, _ = served

    status, body = _request(f"{base}/?t={token}", None)
    assert status == 200
    assert b"<title>reasonese variants</title>" in body
    # The page carries the token so its own calls authenticate.
    assert token.encode() in body

    for url, supplied in ((f"{base}/", None), (f"{base}/?t=wrong", None)):
        with pytest.raises(urllib.error.HTTPError) as caught:
            _request(url, supplied)
        assert caught.value.code == 403

    with pytest.raises(urllib.error.HTTPError) as caught:
        _request(f"{base}/api/state", "wrong-token")
    assert caught.value.code == 403

    with pytest.raises(urllib.error.HTTPError) as caught:
        _request(f"{base}/api/save", None, {"index": 0, "text": "x"})
    assert caught.value.code == 403


def test_saving_through_the_api_writes_the_file(
    served: tuple[str, str, AuthoringSession],
) -> None:
    base, token, session = served
    status, body = _request(f"{base}/api/state", token)
    assert status == 200
    state = json.loads(body)
    index = state["index"]

    status, body = _request(f"{base}/api/save", token, {"index": index, "text": "hand written"})
    assert status == 200
    assert json.loads(body)["written"] == 1
    assert variant_path(session.root, session.queue[index]).read_text() == "hand written\n"

    status, body = _request(f"{base}/api/skip", token, {"index": json.loads(body)["index"]})
    assert status == 200


@pytest.mark.parametrize(
    ("path", "payload", "expected"),
    [
        ("/api/save", {"index": "zero", "text": "x"}, 400),
        ("/api/save", {"index": True, "text": "x"}, 400),
        ("/api/save", {"index": 0, "text": 5}, 400),
        ("/api/save", {"index": 0}, 400),
        ("/api/save", {"index": 0, "text": ""}, 400),
        ("/api/save", {"index": 9999, "text": "x"}, 400),
        ("/api/unknown", {"index": 0}, 404),
    ],
)
def test_the_api_rejects_malformed_requests(
    served: tuple[str, str, AuthoringSession],
    path: str,
    payload: dict[str, object],
    expected: int,
) -> None:
    base, token, _ = served
    with pytest.raises(urllib.error.HTTPError) as caught:
        _request(f"{base}{path}", token, payload)
    assert caught.value.code == expected


def test_unknown_paths_are_not_found(served: tuple[str, str, AuthoringSession]) -> None:
    base, token, _ = served
    with pytest.raises(urllib.error.HTTPError) as caught:
        _request(f"{base}/api/elsewhere?t={token}", token)
    assert caught.value.code == 404


def test_the_server_listens_only_on_loopback(
    scaffolded: tuple[Path, tuple[InstructionPair, ...]],
) -> None:
    root, pairs = scaffolded
    server = build_server(_session(root, pairs), new_token(), 0)
    try:
        assert server.server_address[0] == "127.0.0.1"
    finally:
        server.server_close()


# --------------------------------------------------------------------------
# Tunnel and command line
# --------------------------------------------------------------------------


def test_the_tunnel_reports_a_clear_error_when_cloudflared_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("reasonese.write_variants.shutil.which", lambda name: None)
    with pytest.raises(ValueError, match="cloudflared is not installed"):
        open_tunnel(8000)


def _stub_cloudflared(monkeypatch: pytest.MonkeyPatch, script: str) -> None:
    """Replace cloudflared with a real process writing to a real stderr pipe.

    Using an actual `subprocess.Popen` keeps the readline-and-match path under
    test rather than stubbing it out.
    """
    # `subprocess` is one shared module object, so capture the real constructor
    # before replacing it or the replacement calls itself.
    real_popen = subprocess.Popen
    monkeypatch.setattr("reasonese.write_variants.shutil.which", lambda name: "/usr/bin/true")
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *args, **keywords: real_popen(
            [sys.executable, "-c", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        ),
    )


def test_the_tunnel_returns_the_url_cloudflared_reports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_cloudflared(
        monkeypatch,
        "import sys, time\n"
        "sys.stderr.write('opening a banner line\\n')\n"
        "sys.stderr.write('|  https://brave-otter-1234.trycloudflare.com  |\\n')\n"
        "sys.stderr.flush()\n"
        "time.sleep(30)\n",
    )

    process, url = open_tunnel(8000)
    try:
        assert url == "https://brave-otter-1234.trycloudflare.com"
        assert process.poll() is None
    finally:
        process.terminate()
        process.wait(timeout=10)


def test_the_tunnel_gives_up_when_no_url_arrives(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_cloudflared(
        monkeypatch,
        "import sys, time\nsys.stderr.write('nothing useful\\n')\nsys.stderr.flush()\n"
        "time.sleep(30)\n",
    )

    with pytest.raises(ValueError, match="did not report a tunnel URL"):
        open_tunnel(8000, timeout=1.5)


def test_the_cli_serves_and_reports_a_tokenized_local_url(
    scaffolded: tuple[Path, tuple[InstructionPair, ...]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    reached: list[object] = []

    def capture(tunnel: object) -> None:
        reached.append(tunnel)

    monkeypatch.setattr("reasonese.write_variants.wait_for_exit", capture)
    assert main(["--pairs", str(BANK), "--user-messages", str(MANUAL_ROOT), "--port", "0"]) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["variants"] == 144
    assert summary["instructions"] == 48
    assert summary["tunnel"] is False
    assert summary["written"] + summary["remaining"] == 144
    assert summary["url"].startswith("http://127.0.0.1:")
    assert "?t=" in summary["url"]
    # The token is minted per run, so it must not be a fixed string.
    assert len(summary["url"].split("?t=")[1]) >= 20
    assert reached == [None]


def test_the_cli_reports_a_tunnel_failure_without_leaving_a_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("reasonese.write_variants.shutil.which", lambda name: None)
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "--pairs",
                str(BANK),
                "--user-messages",
                str(MANUAL_ROOT),
                "--port",
                "0",
                "--tunnel",
            ]
        )


def test_the_cli_reports_a_missing_bank(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="2"):
        main(["--pairs", str(tmp_path / "absent.yaml"), "--user-messages", str(tmp_path)])


def test_the_cli_reports_an_unscaffolded_directory(tmp_path: Path) -> None:
    empty = tmp_path / "user"
    empty.mkdir()
    with pytest.raises(SystemExit, match="2"):
        main(["--pairs", str(BANK), "--user-messages", str(empty)])


def test_variant_path_stays_inside_the_manual_root(
    scaffolded: tuple[Path, tuple[InstructionPair, ...]],
) -> None:
    root, pairs = scaffolded
    for task in build_queue(root, pairs, Natural.parse(0)):
        path = variant_path(root, task)
        assert path.resolve().is_relative_to(root.resolve())
        assert path.name in {f"{framing}.txt" for framing in author_framings(Author.USER)}


def test_a_task_outside_the_queue_cannot_be_written(
    scaffolded: tuple[Path, tuple[InstructionPair, ...]],
) -> None:
    """Only queued tasks are addressable, so a slug can never be supplied."""
    root, pairs = scaffolded
    session = _session(root, pairs)
    forged = VariantTask("../../escape", Framing.NORMAL, pairs[0].first)

    assert forged not in session.queue
    # `save` addresses tasks by queue index, so there is no path to reach one.
    with pytest.raises(ValueError, match="unknown task"):
        session.save(len(session.queue), "text")


def test_server_and_queue_survive_a_partially_written_directory(
    scaffolded: tuple[Path, tuple[InstructionPair, ...]],
) -> None:
    root, pairs = scaffolded
    session = _session(root, pairs)
    session.save(0, "already done")

    resumed = _session(root, pairs)
    assert resumed.written() == 1
    assert resumed.current() != 0
