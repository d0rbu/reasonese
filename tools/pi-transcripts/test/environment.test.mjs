import assert from "node:assert/strict";
import { createServer } from "node:http";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import test from "node:test";
import { neutralCwd, projectDir, sandboxTools } from "../sandbox.mjs";
import { snapshotWorkspace } from "../checkpoints.mjs";

const root = () => mkdtempSync(join(projectDir, "out/environment-fixture-"));
test("neutral tools retain private networking, deny metadata/private addresses and hide collector files", { timeout: 180_000 }, async t => {
  const dir = root(); t.after(() => rmSync(dir, { recursive: true, force: true }));
  const cwd = join(dir, "workspace"); mkdirSync(cwd);
  const http = createServer((_, res) => res.end("host-only secret"));
  await new Promise(resolve => http.listen(0, "127.0.0.1", resolve));
  t.after(() => new Promise(resolve => http.close(resolve)));
  const port = http.address().port;
  const tools = sandboxTools(cwd, join(dir, "tool-state"), 15);
  const bash = tools.find(tool => tool.name === "bash");
  const code = `import os,socket,subprocess,sys\nfrom pathlib import Path\nassert str(Path.cwd()) == ${JSON.stringify(neutralCwd())}\nassert os.environ['HOME'] == '/home/developer'\nassert not any(k.startswith('PI_') for k in os.environ)\nassert not os.environ.get('OPENROUTER_API_KEY')\nassert not Path('/opt/pi/collect.mjs').exists()\nassert not Path('/opt/validation').exists()\nfor host in ['127.0.0.1','10.0.2.2','169.254.169.254','192.168.1.1']:\n s=socket.socket(); s.settimeout(1); assert s.connect_ex((host,${port})) != 0, host; s.close()\nassert subprocess.run(['/usr/sbin/nft','flush','ruleset'],capture_output=True).returncode != 0\nassert 'reasonese' not in sys.executable\nprint('neutral paths; loopback/private/metadata rejected; firewall protected')`;
  const result = await bash.execute("network-check", { command: `python - <<'PY'\n${code}\nPY\nrg --version | head -1\nuv --version\nnpm --version\npython -m pytest --version\nruff --version` });
  assert.match(result.content[0].text, /firewall protected/);
  assert.doesNotMatch(result.content[0].text, /command not found|No module named|reasonese|\.codex/);
});

test("task dependency mounts are writable and complete checkpoint includes ignored/untracked state", { timeout: 180_000 }, async t => {
  const dir = root(); t.after(() => rmSync(dir, { recursive: true, force: true }));
  const cwd = join(dir, "workspace"), modules = join(dir, "go-modules");
  mkdirSync(cwd); mkdirSync(modules);
  writeFileSync(join(cwd, ".env"), "FIXTURE=local\n");
  const bash = sandboxTools(cwd, join(dir, "tool-state"), 10, undefined,
    { mounts: [{ source: modules, target: "/home/developer/go/pkg/mod", writable: true }], env: { GOMODCACHE: modules } }).find(t => t.name === "bash");
  const result = await bash.execute("cache-check", { command: "python - <<'PY'\nimport os\nfrom pathlib import Path\np=Path(os.environ['GOMODCACHE'])/'download.lock'\np.write_text('writable')\nassert p.read_text()=='writable'\nPath('untracked.txt').write_text('task change')\nprint('writable cache')\nPY" });
  assert.match(result.content[0].text, /writable cache/);
  assert.equal(readFileSync(join(modules, "download.lock"), "utf8"), "writable");
  const snapshot = await snapshotWorkspace(dir, join(dir, "checkpoint"));
  assert(snapshot.roots.includes("go-modules") && snapshot.roots.includes("tool-state"));
  assert(snapshot.bytes > 0 && /^[a-f0-9]{64}$/.test(snapshot.sha256));
});
