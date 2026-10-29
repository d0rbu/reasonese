"""HTTP authoring flow, paid-check boundaries and exact-text persistence."""

import importlib.util
import io
import json
import shutil
import subprocess
import threading
import time
import zipfile
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pi_authoring", ROOT / "tools/pi-authoring/server.py")
assert spec and spec.loader
workshop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workshop)
KEY = "langchain-json-recovery/implementation/a/normal"
OTHER = "langchain-json-recovery/implementation/b/normal"


def response(request, *, passed=True):
    evidence = json.loads(request["messages"][1]["content"])
    return {
        "model": workshop.MODEL,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(
                        {
                            "summary": "Complete" if passed else "State the requested behavior",
                            "checks": [
                                {"id": check["id"], "satisfied": passed, "feedback": "Reviewed"}
                                for check in evidence["requirements"]
                            ],
                        }
                    )
                },
            }
        ],
    }


@pytest.fixture
def factory(tmp_path):
    site = tmp_path / "site"
    (site / "kimi-transcripts").mkdir(parents=True)
    rubrics = json.loads((ROOT / "configs/pi_authoring_rubrics.json").read_text())
    rows = rubrics["messages"]
    bank = tmp_path / "seed-bank"
    shutil.copytree(ROOT / "prompts/scenario_turns", bank)
    cases = sorted({r["case"] for r in rows})
    messages = [
        {"role": "user", "content": "Investigate first"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Private reasoning"},
                {"type": "text", "text": "These are the findings"},
            ],
        },
        {"role": "user", "content": "Implement A"},
        {"role": "assistant", "content": "Done"},
    ]
    catalog = {"scenarios": []}
    for case in cases:
        folder = site / case
        folder.mkdir()
        (folder / "data.json").write_text(json.dumps({"messages": messages, "selection": {}}))
        (folder / "evidence").mkdir()
        native = (json.dumps(messages) + "\n").encode()
        (folder / "evidence/messages.json").write_bytes(native)
        binding_file = bank / case / "source-binding.json"
        binding = json.loads(binding_file.read_text())
        binding["target"]["messagesSha256"] = workshop.sha(native)
        binding_file.write_text(json.dumps(binding))
        for row in rows:
            if row["case"] == case:
                row["sourceBindingSha256"] = workshop.sha(binding_file.read_bytes())
        catalog["scenarios"].append(
            {
                "id": case,
                "title": case,
                "project": "Project",
                "path": f"/{case}/",
                "messagesSha256": workshop.sha(native),
            }
        )
    (site / "kimi-transcripts/catalog.json").write_text(json.dumps(catalog))
    (tmp_path / "rubrics.json").write_text(json.dumps(rubrics))
    apps = []

    def make(judge=response, **kwargs):
        app = workshop.App(
            bank,
            tmp_path / "rubrics.json",
            site,
            tmp_path / f"state-{len(apps)}",
            judge,
            **kwargs,
        )
        apps.append(app)
        return app

    yield make
    for app in apps:
        app.close()


def wait(app, job):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        result = app.job(job)
        if result["status"] not in {"queued", "running"}:
            return result
        time.sleep(0.01)
    pytest.fail("Judge job did not finish")


@contextmanager
def http(app):
    server = ThreadingHTTPServer(("127.0.0.1", 0), workshop.handler(app))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", {"Authorization": "Bearer " + app.token}
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_bespoke_bound_rubrics_and_prior_context(factory):
    app = factory()
    assert len(app.messages) == 108
    assert len(app.cases) == 12
    for message in app.messages.values():
        assert message["reference"].strip()
        assert message["checks"][0]["id"] == "task-1"
        request = workshop.judge_request(message, "A user's complete message")
        assert request["model"] == "openai/gpt-6.1-sol"
        assert request["reasoning"] == {"effort": "high"}
        assert "tools" not in request
        context = json.loads(request["messages"][1]["content"])["prior_context"]
        assert bool(context) == (message["slot"] == "implementation")
        assert "Private reasoning" not in json.dumps(context)
        assert "Implement A" not in json.dumps(context)
    assert app.messages[KEY]["checks"][0] != app.messages[OTHER]["checks"][0]
    rubrics = json.loads((ROOT / "configs/pi_authoring_rubrics.json").read_text())
    rubrics["messages"][0]["authoringSourceSha256"] = "0" * 64
    changed = app.state.parent / "stale.json"
    changed.write_text(json.dumps(rubrics))
    with pytest.raises(ValueError, match="Stale rubric"):
        workshop.App(app.source_bank, changed, app.site, app.state.parent / "stale-state")


def test_http_exact_text_private_drafts_and_safe_static_paths(factory):
    app = factory()
    raw = 'hi\n\n  {"name": "Ada"}\n<script>globalThis.injected = true</script>'
    with http(app) as (url, auth):
        assert requests.get(url + "/api/catalog").json()["canWrite"] is False
        assert requests.get(url + "/api/catalog", headers=auth).json()["canWrite"] is True
        body = {"key": KEY, "text": raw, "author": "Max"}
        assert requests.post(url + "/api/draft", json=body).status_code == 403
        assert (
            requests.post(
                url + "/api/submit", json=body, headers=auth | {"Origin": "https://other.example"}
            ).status_code
            == 403
        )
        assert requests.post(url + "/api/draft", json=body, headers=auth).status_code == 200
        assert requests.get(url + "/api/message/" + KEY).json()["draft"] == ""
        assert requests.get(url + "/api/message/" + KEY, headers=auth).json()["draft"] == raw
        submitted = requests.post(url + "/api/submit", json=body, headers=auth)
        assert submitted.status_code == 202
        job = wait(app, submitted.json()["id"])
        assert job["status"] == "passed"
        assert requests.get(url + "/api/job/" + job["id"]).status_code == 403
        assert requests.get(url + "/api/export.patch").status_code == 403
        assert requests.get(url + "/api/message/" + KEY).json()["acceptedText"] is None
        assert requests.get(url + "/api/message/" + KEY, headers=auth).json()["acceptedText"] == raw
        for path in ["/", "/kimi-json-recovery/", "/kimi-transcripts/index.html"] + [
            c["path"] for c in app.cases.values()
        ]:
            page = requests.get(url + path)
            assert page.status_code == 200 and "Message workshop" in page.text
            assert "no-store" in page.headers["Cache-Control"]
        assert requests.get(url + "/%2e%2e/access-token").status_code == 404
        assert requests.get(url + "/api/unknown").status_code == 404
        native_url = url + app.cases["langchain-json-recovery"]["path"] + "evidence/messages.json"
        native_head = requests.head(native_url)
        native_get = requests.get(native_url)
        assert native_head.status_code == 200 and native_head.content == b""
        assert int(native_head.headers["Content-Length"]) == len(native_get.content)
        (app.site / "escape").symlink_to(app.state / "access-token")
        assert requests.get(url + "/escape").status_code == 404
        assert (
            requests.post(
                url + "/api/submit", json={"key": KEY, "text": "x" * 25000}, headers=auth
            ).status_code
            == 400
        )
    assert (app.bank / app.messages[KEY]["path"]).read_bytes() == raw.encode()
    assert (app.source_bank / app.messages[KEY]["path"]).read_text().startswith("TODO:")
    saved = json.loads((app.state / "checks" / job["id"] / "request.json").read_text())
    assert json.loads(saved["messages"][1]["content"])["candidate_text"] == raw
    exported = app.export(True)
    result = subprocess.run(
        ["git", "apply", "--check", "-"], input=exported, cwd=ROOT, capture_output=True
    )
    assert result.returncode == 0, result.stderr.decode()
    with zipfile.ZipFile(io.BytesIO(app.export(False))) as archive:
        assert archive.read(app.messages[KEY]["path"]) == raw.encode()
        provenance = json.loads(archive.read("authoring-submissions.json"))
        assert provenance["accepted"][0]["author"] == "Max"
        assert provenance["model"] == workshop.MODEL


def test_cache_reselect_and_failed_checks_keep_last_accepted(factory):
    calls = []

    def judge(request):
        calls.append(request)
        text = json.loads(request["messages"][1]["content"])["candidate_text"]
        if text == "transport failure":
            raise requests.ConnectionError("offline")
        return response(request, passed=text != "incomplete")

    app = factory(judge)
    first = app.submit(KEY, "first exact text", "Max")
    assert wait(app, first)["status"] == "passed"
    assert app.submit(KEY, "first exact text", "Max") == first
    second = app.submit(KEY, "second exact text\n", "Max")
    assert wait(app, second)["status"] == "passed"
    # Re-selecting an already judged version must restore that version without another paid call.
    assert app.submit(KEY, "first exact text", "Max") == first
    assert app.message(KEY, True)["acceptedText"] == "first exact text"
    assert wait(app, app.submit(KEY, "incomplete", "Max"))["status"] == "rejected"
    failed = app.submit(KEY, "transport failure", "Max")
    assert wait(app, failed)["status"] == "error"
    assert (app.state / "checks" / failed / "error.json").is_file()
    assert app.message(KEY, True)["acceptedText"] == "first exact text"
    assert app.message(KEY, True)["draft"] == "transport failure"
    assert len(calls) == 4


@pytest.mark.parametrize("fault", ["wrong-model", "cutoff", "missing", "duplicate", "invalid-bool"])
def test_incomplete_or_wrong_model_reviews_never_accept(factory, fault):
    def judge(request):
        item = response(request)
        choice = item["choices"][0]
        verdict = json.loads(choice["message"]["content"])
        if fault == "wrong-model":
            item["model"] = "another-model"
        elif fault == "cutoff":
            choice["finish_reason"] = "length"
        elif fault == "missing":
            verdict["checks"].pop()
        elif fault == "duplicate":
            verdict["checks"].append(verdict["checks"][0])
        else:
            verdict["checks"][0]["satisfied"] = "true"
        choice["message"]["content"] = json.dumps(verdict)
        return item

    app = factory(judge)
    assert wait(app, app.submit(KEY, "A complete candidate", "Max"))["status"] == "error"
    assert app.catalog()["accepted"] == []
    assert (app.bank / app.messages[KEY]["path"]).read_text().startswith("TODO:")


def test_same_message_concurrency_and_budget_boundaries(factory):
    started, release = threading.Event(), threading.Event()

    def judge(request):
        started.set()
        assert release.wait(5)
        return response(request)

    app = factory(judge, max_checks=1)
    try:
        job = app.submit(KEY, "original", "Max")
        assert started.wait(5)
        assert app.submit(KEY, "original", "Max") == job
        with pytest.raises(ValueError, match="running"):
            app.submit(KEY, "newer", "Max")
        with pytest.raises(ValueError, match="limit"):
            app.submit(OTHER, "another task", "Max")
    finally:
        release.set()
    assert wait(app, job)["status"] == "passed"
    assert app.message(KEY, True)["draft"] == "newer"
    assert app.message(KEY, True)["acceptedText"] == "original"


def test_restart_repairs_bank_and_recovers_interrupted_drafts(factory):
    app = factory()
    job = app.submit(KEY, "accepted exact bytes", "Max")
    assert wait(app, job)["status"] == "passed"
    (app.bank / app.messages[KEY]["path"]).write_text("interrupted file write")
    app.save_draft(OTHER, "not yet checked", "Max")
    with app.db() as conn:
        conn.execute(
            "INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?)",
            ("interrupted", OTHER, "not yet checked", "Max", "fingerprint", "running", None, 0),
        )
    app.close()
    restarted = workshop.App(
        app.source_bank, app.state.parent / "rubrics.json", app.site, app.state, response
    )
    try:
        assert (
            restarted.bank / restarted.messages[KEY]["path"]
        ).read_text() == "accepted exact bytes"
        assert restarted.job("interrupted")["status"] == "error"
        assert restarted.message(OTHER, True)["draft"] == "not yet checked"
    finally:
        restarted.close()
    (app.site / "langchain-json-recovery/data.json").write_text('{"messages": []}')
    with pytest.raises(ValueError, match="source binding"):
        workshop.App(app.source_bank, app.state.parent / "rubrics.json", app.site, app.state)


def test_read_only_and_todo_never_make_a_paid_call(factory):
    app = factory(None)
    with pytest.raises(ValueError, match="not enabled"):
        app.submit(KEY, "draft only", "Max")
    assert app.message(KEY, True)["draft"] == "draft only"
    assert app.catalog()["writable"] is False
    app = factory()
    with pytest.raises(ValueError, match="TODO"):
        app.submit(KEY, "TODO: write this", "Max")
    assert app.catalog()["accepted"] == []
    with app.db() as conn:
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


def test_private_state_cannot_be_served_from_public_site(factory):
    app = factory()
    with pytest.raises(ValueError, match="outside the public"):
        workshop.App(
            app.source_bank, app.state.parent / "rubrics.json", app.site, app.site / "private-state"
        )
    assert not (app.site / "private-state").exists()
