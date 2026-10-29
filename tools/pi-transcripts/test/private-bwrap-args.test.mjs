import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { randomUUID } from "node:crypto";
import test from "node:test";
import { encodeBwrapArgs } from "../bwrap-args.mjs";
import { createNeutralCase } from "../backing.mjs";
import { projectDir, sandboxTools } from "../sandbox.mjs";

const text = result => result.content.map(block => block.text ?? "").join("");
const descendants = token => readdirSync("/proc").filter(name => /^\d+$/.test(name)).filter(name => {
  try { return readFileSync(`/proc/${name}/cmdline`).toString().split("\0").includes(token); }
  catch { return false; }
});
async function fixture(t, timeout = 10) {
  const dir = mkdtempSync(join(projectDir, "out/private-argv-"));
  const physical = createNeutralCase(join(dir, "case"));
  t.after(() => { rmSync(dir, { recursive: true, force: true }); rmSync(dirname(physical), { recursive: true, force: true }); });
  const cwd = join(physical, "workspace"), venv = join(physical, "python-env");
  mkdirSync(cwd);
  execFileSync("uv", ["venv", "--python", join(projectDir, ".venv/bin/python"), venv], { stdio: "pipe" });
  const environment = { env: { DEVELOPMENT_NOTE: "space 'quote'\nλ", EMPTY_VALUE: "" } };
  return { physical, bash: sandboxTools(cwd, join(physical, "tool-state"), timeout, venv, environment).find(tool => tool.name === "bash") };
}

test("private argument serialization preserves empty, whitespace, Unicode and shell-literal arguments", () => {
  const args = ["--bind", "/tmp/private path", "", "é\nline", "'\"$()\\", "--"];
  assert.deepEqual(encodeBwrapArgs(args), Buffer.from("--bind\0/tmp/private path\0\0é\nline\0'\"$()\\\0--\0"));
  assert.deepEqual(encodeBwrapArgs([]), Buffer.alloc(0));
  assert.throws(() => encodeBwrapArgs(["bad\0argument"]), /without NUL/);
  assert.throws(() => encodeBwrapArgs([3]), /must be strings/);
});

test("PID1 retains its reaper without exposing mount argv or an inherited setup descriptor", { timeout: 180_000 }, async t => {
  const { bash, physical } = await fixture(t);
  const code = `import json,os,time\nfrom pathlib import Path\nassert os.environ['DEVELOPMENT_NOTE'].splitlines()==["space 'quote'",chr(955)]\nassert os.environ['EMPTY_VALUE']==''\nargs=Path('/proc/1/cmdline').read_bytes().split(b'\\0')[:-1]\nassert args==[b'bwrap',b'--args',b'3',b'/opt/bin/node',b'/opt/pi/tool-worker.mjs'],args\nfds={}\ntry:\n for fd in Path('/proc/1/fd').iterdir():\n  try:fds[fd.name]=os.readlink(fd)\n  except PermissionError:fds[fd.name]='permission denied'\nexcept PermissionError:fds['directory']='permission denied'\nassert '3' not in fds,fds\nassert not any(${JSON.stringify(physical)} in value for value in fds.values()),fds\ncache=Path.home()/'.cache'/'private-argv';cache.mkdir(parents=True);(cache/'write').write_text('works')\nassert (cache/'write').read_text()=='works'\nreadfd,writefd=os.pipe()\npid=os.fork()\nif pid==0:\n os.close(readfd)\n orphan=os.fork()\n if orphan==0:\n  os.close(writefd);time.sleep(.1);os._exit(0)\n os.write(writefd,str(orphan).encode());os._exit(0)\nos.close(writefd);orphan=int(os.read(readfd,100));os.close(readfd);os.waitpid(pid,0)\nfor _ in range(200):\n if not Path('/proc',str(orphan)).exists():break\n time.sleep(.01)\nelse:raise AssertionError('PID1 failed to reap orphan')\nprint(json.dumps({'pid1Argv':[x.decode() for x in args],'pid1Descriptors':fds,'writableCache':True,'orphanReaped':True}))`;
  const result = await bash.execute("private-argv", { command: `set -e\npython - <<'PY'\n${code}\nPY\nps -ef | rg '[u]v sync|[u]v ' || true` });
  assert.equal(result.isError ?? false, false, text(result));
  assert.match(text(result), /"orphanReaped": true/);
  assert.doesNotMatch(text(result), /\/home\/d0rb|reasonese|\.codex|--ro-bind/);
  await assert.rejects(bash.execute("exit-code", { command: "exit 37" }), /code 37/);
});

test("tool timeout still tears down namespace descendants with private argument transport", { timeout: 30_000 }, async t => {
  const { bash } = await fixture(t, 2);
  const token = `private-argv-${randomUUID()}`;
  await assert.rejects(bash.execute("timeout-child", { command: `python -c 'import time; time.sleep(30)' ${token} & wait` }), /time limit/);
  for (let i = 0; i < 100 && descendants(token).length; i++) await new Promise(resolve => setTimeout(resolve, 20));
  assert.deepEqual(descendants(token), [], "Timed-out namespace left a live descendant");
});

test("tool cancellation still tears down namespace descendants", { timeout: 30_000 }, async t => {
  const { bash } = await fixture(t);
  const token = `private-argv-${randomUUID()}`, controller = new AbortController();
  const stopped = assert.rejects(bash.execute("cancel-child", { command: `python -c 'import time; time.sleep(30)' ${token} & wait` }, controller.signal), /Tool aborted/);
  try {
    for (let i = 0; i < 100 && !descendants(token).length; i++) await new Promise(resolve => setTimeout(resolve, 20));
    assert(descendants(token).length, "Cancellation fixture did not launch its child");
  } finally { controller.abort(); await stopped; }
  for (let i = 0; i < 100 && descendants(token).length; i++) await new Promise(resolve => setTimeout(resolve, 20));
  assert.deepEqual(descendants(token), [], "Cancelled namespace left a live descendant");
});
