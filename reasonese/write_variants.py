"""A blinded local editor for hand-written user-authored variants.

Writing 144 variants by hand invites several biases, and the interface is built
to remove the ones it can:

- one task at a time, so nothing is written by comparison with a neighbour;
- the whole queue shuffled by seed, so the framings of one instruction are far
  apart and no house style carries between them;
- pair identifier, side, skill, and conflict type never leave the server, since
  a directory name like `prime-1234-bare-vs-table-a` would otherwise reveal all
  four;
- the partner instruction is never shown, so neither side of a pair can be
  written to beat the other.

What it cannot remove is stated in the docs: a manual variant is one file reused
across all three channels, while a model author writes one per channel.
"""

from __future__ import annotations

import argparse
import json
import re
import secrets
import shutil
import subprocess
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from random import Random
from typing import Any, cast

from beartype import beartype
from phantom.interval import Natural

from reasonese.axes import Author, Framing, Instruction, author_framings
from reasonese.conversation import AUTHORING_RULE, framing_guidance
from reasonese.instructions import InstructionPair, load_instruction_pairs

PLACEHOLDER_PREFIX = "TODO:"
SOURCE_FILE = "instruction.txt"
_MAX_BODY_BYTES = 256 * 1024


@beartype
@dataclass(frozen=True, slots=True)
class VariantTask:
    """One instruction and the framing still to be written for it.

    `slug` is the directory name and stays server-side: it encodes the pair
    identifier and which side of the pair this is.
    """

    slug: str
    framing: Framing
    instruction: Instruction


def _instruction_directories(root: Path) -> dict[Instruction, str]:
    if not root.is_dir():
        raise ValueError(f"manual message directory does not exist: {root}")
    found: dict[Instruction, str] = {}
    for directory in sorted(entry for entry in root.iterdir() if entry.is_dir()):
        source = directory / SOURCE_FILE
        if not source.is_file():
            continue
        instruction = Instruction.parse(source.read_text(encoding="utf-8").strip())
        if instruction in found:
            raise ValueError(f"two directories hold the same instruction: {directory}")
        found[instruction] = directory.name
    return found


@beartype
def build_queue(
    root: Path,
    pairs: tuple[InstructionPair, ...],
    seed: Natural,
) -> tuple[VariantTask, ...]:
    """Enumerate every manual variant the bank needs, ordered to prevent anchoring.

    A plain shuffle leaves it to luck how close an instruction's framings land;
    over the real bank the closest pair came out five apart, near enough to write
    the second from memory of the first. So the queue is laid out as one round
    per framing instead. Each instruction keeps the same slot in every round,
    which puts its appearances exactly `len(instructions)` apart, and the framing
    each instruction takes in each round is rotated independently so a round is
    still a mix of framings rather than a block of one.
    """
    directories = _instruction_directories(root)
    instructions: list[tuple[str, Instruction]] = []
    for pair in pairs:
        for instruction in pair.instructions:
            slug = directories.get(instruction)
            if slug is None:
                raise ValueError(
                    "no manual variant directory exists for an instruction in the bank; "
                    f"run reasonese-curate-instructions --scaffold-user-prompts: {instruction}"
                )
            instructions.append((slug, instruction))

    framings = author_framings(Author.USER)
    random = Random(int(seed))
    slots = list(range(len(instructions)))
    random.shuffle(slots)
    rotations = [random.randrange(len(framings)) for _ in instructions]

    queue: list[VariantTask | None] = [None] * (len(instructions) * len(framings))
    for index, slot in enumerate(slots):
        slug, instruction = instructions[index]
        for round_number in range(len(framings)):
            framing = framings[(rotations[index] + round_number) % len(framings)]
            queue[round_number * len(instructions) + slot] = VariantTask(
                slug, framing, instruction
            )
    return tuple(cast(list[VariantTask], queue))


@beartype
def variant_path(root: Path, task: VariantTask) -> Path:
    """Return the file one task writes to."""
    return root / task.slug / f"{task.framing}.txt"


@beartype
def is_written(root: Path, task: VariantTask) -> bool:
    """Return whether a variant has real text rather than its placeholder."""
    path = variant_path(root, task)
    if not path.is_file():
        return False
    content = path.read_text(encoding="utf-8").strip()
    return bool(content) and not content.startswith(PLACEHOLDER_PREFIX)


@beartype
def guidance_for(framing: Framing) -> str:
    """Return the same briefing a model author receives for this framing."""
    return f"{framing_guidance(framing)}\n\n{AUTHORING_RULE}"


@beartype
@dataclass(slots=True)
class AuthoringSession:
    """The queue, the filesystem, and which tasks were skipped this run."""

    root: Path
    queue: tuple[VariantTask, ...]
    skipped: set[int]

    @beartype
    def written(self) -> int:
        """Count variants already written."""
        return sum(is_written(self.root, task) for task in self.queue)

    @beartype
    def current(self) -> int | None:
        """Return the next unwritten, unskipped index, or None when finished."""
        for index, task in enumerate(self.queue):
            if index not in self.skipped and not is_written(self.root, task):
                return index
        return None

    @beartype
    def state(self) -> dict[str, Any]:
        """Return everything the page may see, and nothing that would bias it."""
        index = self.current()
        written = self.written()
        total = len(self.queue)
        if index is None:
            remaining = total - written
            return {
                "done": True,
                "written": written,
                "total": total,
                "skipped": remaining,
            }
        task = self.queue[index]
        return {
            "done": False,
            "index": index,
            "written": written,
            "total": total,
            "framing": str(task.framing),
            "guidance": guidance_for(task.framing),
            "instruction": str(task.instruction),
        }

    @beartype
    def save(self, index: int, text: str) -> Path:
        """Write one variant after checking the index and the text."""
        if not 0 <= index < len(self.queue):
            raise ValueError("unknown task")
        stripped = text.strip()
        if not stripped:
            raise ValueError("a variant cannot be empty")
        if stripped.startswith(PLACEHOLDER_PREFIX):
            raise ValueError("a variant cannot start with the placeholder prefix")
        path = variant_path(self.root, self.queue[index])
        path.write_text(f"{stripped}\n", encoding="utf-8")
        self.skipped.discard(index)
        return path

    @beartype
    def skip(self, index: int) -> None:
        """Defer one task until every other unwritten task has been offered."""
        if not 0 <= index < len(self.queue):
            raise ValueError("unknown task")
        self.skipped.add(index)


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>reasonese variants</title>
<style>
:root{color-scheme:light dark;--bg:#fbfbfa;--fg:#1a1a19;--dim:#6b6b66;--line:#e0e0dc;--accent:#2f6f4f;--card:#fff}
@media(prefers-color-scheme:dark){:root{--bg:#16171a;--fg:#e9e9e6;--dim:#94948e;--line:#2c2e33;--accent:#7fc9a2;--card:#1d1f23}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:44rem;margin:0 auto;padding:1.5rem 1.25rem 4rem}
.bar{display:flex;justify-content:space-between;align-items:baseline;gap:1rem;
font-size:.82rem;color:var(--dim);margin-bottom:.5rem}
.track{height:3px;background:var(--line);border-radius:2px;overflow:hidden;margin-bottom:1.75rem}
.fill{height:100%;background:var(--accent);transition:width .25s}
h1{font-size:1.05rem;margin:0 0 .6rem;letter-spacing:.01em}
h1 em{font-style:normal;color:var(--accent);text-transform:uppercase;letter-spacing:.06em}
.guide{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--accent);
border-radius:6px;padding:.8rem .95rem;font-size:.9rem;color:var(--dim);white-space:pre-wrap;margin-bottom:1.5rem}
.label{font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;color:var(--dim);margin-bottom:.4rem}
.request{background:var(--card);border:1px solid var(--line);border-radius:6px;
padding:.9rem 1rem;white-space:pre-wrap;margin-bottom:1.5rem}
textarea{width:100%;min-height:11rem;padding:.9rem 1rem;border:1px solid var(--line);border-radius:6px;
background:var(--card);color:var(--fg);font:inherit;resize:vertical}
textarea:focus{outline:2px solid var(--accent);outline-offset:-1px}
.row{display:flex;gap:.6rem;align-items:center;margin-top:.9rem;flex-wrap:wrap}
button{font:inherit;padding:.5rem 1.1rem;border-radius:6px;border:1px solid var(--line);
background:var(--card);color:var(--fg);cursor:pointer}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:600}
@media(prefers-color-scheme:dark){button.primary{color:#10231a}}
button:disabled{opacity:.5;cursor:default}
.hint{margin-left:auto;font-size:.78rem;color:var(--dim)}
.msg{min-height:1.2rem;font-size:.82rem;color:var(--accent);margin-top:.6rem}
.msg.bad{color:#c2410c}
.done{text-align:center;padding:4rem 1rem}
.done h2{font-size:1.4rem;margin:0 0 .5rem}
.done p{color:var(--dim);margin:.2rem 0}
</style></head><body><main id="app"></main>
<script>
const TOKEN = "__TOKEN__";
const app = document.getElementById("app");
const esc = (s) => s.replace(/[&<>]/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));

async function call(path, body) {
  const res = await fetch(path, {
    method: body ? "POST" : "GET",
    headers: {"X-Auth-Token": TOKEN, "Content-Type": "application/json"},
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error || res.statusText);
  return res.json();
}

function render(state, note, bad) {
  if (state.done) {
    app.innerHTML = `<div class="done"><h2>All done</h2>
      <p>${state.written} of ${state.total} variants written.</p>
      ${state.skipped ? `<p>${state.skipped} skipped — reload to revisit them.</p>` : ""}</div>`;
    return;
  }
  const pct = state.total ? (state.written / state.total) * 100 : 0;
  app.innerHTML = `
    <div class="bar"><span>${state.written} / ${state.total} written</span><span>blinded</span></div>
    <div class="track"><div class="fill" style="width:${pct}%"></div></div>
    <h1>Rewrite this request in the <em>${esc(state.framing)}</em> framing.</h1>
    <div class="guide">${esc(state.guidance)}</div>
    <div class="label">Request</div>
    <div class="request">${esc(state.instruction)}</div>
    <div class="label">Your version</div>
    <textarea id="text" autofocus spellcheck="true"></textarea>
    <div class="row">
      <button class="primary" id="save">Save and next</button>
      <button id="skip">Skip</button>
      ${state.framing === "normal"
        ? '<button id="verbatim">Use the request verbatim</button>' : ""}
      <span class="hint">Ctrl/Cmd + Enter to save</span>
    </div>
    <div class="msg${bad ? " bad" : ""}">${note ? esc(note) : ""}</div>`;

  const text = document.getElementById("text");
  const save = async () => {
    try {
      render(await call("/api/save", {index: state.index, text: text.value}), "Saved.");
    } catch (error) { render(state, error.message, true); }
  };
  document.getElementById("save").onclick = save;
  document.getElementById("skip").onclick = async () => {
    render(await call("/api/skip", {index: state.index}), "Skipped.");
  };
  const verbatim = document.getElementById("verbatim");
  if (verbatim) verbatim.onclick = () => { text.value = state.instruction; text.focus(); };
  text.onkeydown = (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") { event.preventDefault(); save(); }
  };
  text.focus();
}

call("/api/state").then((state) => render(state)).catch((error) => {
  app.innerHTML = `<div class="done"><h2>Cannot load</h2><p>${esc(error.message)}</p></div>`;
});
</script></body></html>
"""


def _handler(session: AuthoringSession, token: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            """Keep the terminal free for the tunnel URL."""

        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def _authorized(self) -> bool:
            supplied = self.headers.get("X-Auth-Token", "")
            return secrets.compare_digest(supplied, token)

        def do_GET(self) -> None:  # noqa: N802
            path, _, query = self.path.partition("?")
            if path == "/":
                supplied = ""
                for part in query.split("&"):
                    key, _, value = part.partition("=")
                    if key == "t":
                        supplied = value
                if not secrets.compare_digest(supplied, token):
                    self._send(HTTPStatus.FORBIDDEN, b"forbidden", "text/plain")
                    return
                page = _PAGE.replace("__TOKEN__", token)
                self._send(HTTPStatus.OK, page.encode(), "text/html; charset=utf-8")
                return
            if path == "/api/state":
                if not self._authorized():
                    self._json(HTTPStatus.FORBIDDEN, {"error": "forbidden"})
                    return
                self._json(HTTPStatus.OK, session.state())
                return
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

        def do_POST(self) -> None:  # noqa: N802
            path, _, _ = self.path.partition("?")
            if not self._authorized():
                self._json(HTTPStatus.FORBIDDEN, {"error": "forbidden"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length > _MAX_BODY_BYTES:
                self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "body too large"})
                return
            try:
                payload = cast(dict[str, Any], json.loads(self.rfile.read(length) or b"{}"))
                index = payload["index"]
                if not isinstance(index, int) or isinstance(index, bool):
                    raise ValueError("index must be an integer")
                if path == "/api/save":
                    text = payload["text"]
                    if not isinstance(text, str):
                        raise ValueError("text must be a string")
                    session.save(index, text)
                elif path == "/api/skip":
                    session.skip(index)
                else:
                    self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
                    return
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            self._json(HTTPStatus.OK, session.state())

    return Handler


@beartype
def build_server(
    session: AuthoringSession,
    token: str,
    port: int = 0,
) -> ThreadingHTTPServer:
    """Bind a loopback-only server. Nothing reaches it except through a tunnel."""
    return ThreadingHTTPServer(("127.0.0.1", port), _handler(session, token))


@beartype
def new_token() -> str:
    """Return an unguessable token for one run."""
    return secrets.token_urlsafe(32)


_TUNNEL_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


@beartype
def open_tunnel(port: int, timeout: float = 45.0) -> tuple[subprocess.Popen[str], str]:
    """Start a Cloudflare quick tunnel and return it with its public URL.

    A quick tunnel is public, so the server behind it stays bound to loopback
    and every request must carry the run's token.
    """
    if shutil.which("cloudflared") is None:
        raise ValueError("cloudflared is not installed; install it or omit --tunnel")
    process = subprocess.Popen(  # noqa: S603
        [
            "cloudflared",
            "tunnel",
            "--no-autoupdate",
            "--url",
            f"http://127.0.0.1:{port}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stderr is not None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = process.stderr.readline()
        if not line:
            break
        found = _TUNNEL_URL.search(line)
        if found:
            return process, found.group(0)
    process.terminate()
    raise ValueError("cloudflared did not report a tunnel URL in time")


@beartype
def wait_for_exit(tunnel: subprocess.Popen[str] | None) -> None:
    """Block until Ctrl-C, or until the tunnel process goes away."""
    try:
        while True:
            time.sleep(0.5)
            if tunnel is not None and tunnel.poll() is not None:
                break
    except KeyboardInterrupt:
        pass


@beartype
def main(argv: Sequence[str] | None = None) -> int:
    """Serve the blinded variant editor, optionally behind a Cloudflare tunnel."""
    parser = argparse.ArgumentParser(prog="reasonese-write-variants")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--user-messages", type=Path, default=Path("prompts/user"))
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tunnel",
        action="store_true",
        help="expose the editor on a public Cloudflare quick tunnel URL",
    )
    args = parser.parse_args(argv)

    tunnel: subprocess.Popen[str] | None = None
    try:
        pairs = load_instruction_pairs(args.pairs)
        queue = build_queue(args.user_messages, pairs, Natural.parse(args.seed))
        session = AuthoringSession(args.user_messages, queue, set())
        token = new_token()
        server = build_server(session, token, args.port)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))

    port = server.server_address[1]
    base = f"http://127.0.0.1:{port}"
    try:
        if args.tunnel:
            tunnel, base = open_tunnel(int(port))
    except (OSError, ValueError) as error:
        server.server_close()
        parser.error(str(error))

    print(
        json.dumps(
            {
                "instructions": 2 * len(pairs),
                "port": int(port),
                "remaining": len(queue) - session.written(),
                "seed": args.seed,
                "tunnel": bool(args.tunnel),
                "url": f"{base}/?t={token}",
                "variants": len(queue),
                "written": session.written(),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    # `shutdown` waits for `serve_forever` to acknowledge it, so the loop has to
    # be running before the finally block can ask it to stop.
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        wait_for_exit(tunnel)
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        if tunnel is not None:
            tunnel.terminate()
    return 0
