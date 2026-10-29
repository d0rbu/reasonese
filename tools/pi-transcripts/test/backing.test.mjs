import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import test from "node:test";
import { createNeutralCase, isNeutralCase, neutralRuntime } from "../backing.mjs";
import { restoreCheckpoint, snapshotWorkspace } from "../checkpoints.mjs";
import { git } from "../workspaces.mjs";
import { projectDir, sandboxTools } from "../sandbox.mjs";

test("fresh physical backing hides research paths in kernel mount metadata, Python metadata, Git and environment", { timeout: 180_000 }, async t => {
  const dir = mkdtempSync(join(projectDir, "out/backing-audit-")), alias = join(dir, "case");
  const physical = createNeutralCase(alias);
  t.after(() => { rmSync(dir, { recursive: true, force: true }); rmSync(dirname(physical), { recursive: true, force: true }); });
  assert.equal(realpathSync(alias), physical); assert(isNeutralCase(physical));
  assert.doesNotMatch(neutralRuntime().root, /reasonese|\.codex/);
  const cwd = join(physical, "workspace"), venv = join(physical, "python-env"); mkdirSync(cwd);
  execFileSync("uv", ["venv", "--python", join(projectDir, ".venv/bin/python"), venv], { stdio: "pipe" });
  const bash = sandboxTools(cwd, join(physical, "tool-state"), 30, venv).find(tool => tool.name === "bash");
  const input = { command: `python - <<'PY'\nimport os,sys\nfrom pathlib import Path\nfor p in ['/proc/self/mountinfo','/proc/self/mounts']:\n text=Path(p).read_text()\n assert 'reasonese' not in text and '.codex' not in text, text\nassert not any(x.startswith('PI_') for x in os.environ)\nassert not any('reasonese' in x or '.codex' in x for x in sys.path)\nassert not Path('/opt/validation').exists()\nassert not Path('/opt/pi/collect.mjs').exists()\nassert not list(Path.home().rglob('*baseline*'))\nassert os.environ['UV_LINK_MODE']=='copy'\nprint('neutral kernel mounts, sys.path, environment, SDK and model caches')\nPY` };
  const result = await bash.execute("physical-audit", input);
  assert.match(result.content[0].text, /neutral kernel mounts/);
  const receipt = await snapshotWorkspace(physical, join(dir, "checkpoint"));
  const resumed = await restoreCheckpoint(join(dir, receipt.archive), join(dir, "resumed"), {
    sha256: receipt.sha256, sourceCaseDir: alias,
    prepared: { pythonEnvironment: { path: venv }, toolEnvironment: { mounts: [], path: [], env: {} } },
  });
  t.after(() => rmSync(dirname(resumed.destination), { recursive: true, force: true }));
  assert.equal(resumed.prepared.pythonEnvironment.path, join(resumed.destination, "python-env"));
  const resumedBash = sandboxTools(resumed.workspace, join(resumed.destination, "tool-state"), 30,
    resumed.prepared.pythonEnvironment.path, resumed.prepared.toolEnvironment).find(tool => tool.name === "bash");
  const resumedResult = await resumedBash.execute("restored-physical-audit", input);
  assert.match(resumedResult.content[0].text, /neutral kernel mounts/);
});

test("host verification cannot execute repository fsmonitor commands", async t => {
  const dir = mkdtempSync(join(projectDir, "out/git-monitor-audit-"));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  await git(dir, "init");
  const sentinel = join(dir, "host-monitor-ran"), monitor = join(dir, "monitor.sh");
  writeFileSync(monitor, `#!/bin/sh\ntouch '${sentinel}'\n`, { mode: 0o755 });
  await git(dir, "config", "core.fsmonitor", monitor);
  await git(dir, "status", "--porcelain");
  assert.equal(existsSync(sentinel), false);
  assert.match(readFileSync(monitor, "utf8"), /touch/);
});
