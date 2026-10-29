import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readlinkSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import test from "node:test";
import { restoreCheckpoint, snapshotWorkspace } from "../checkpoints.mjs";
import { projectDir, sandboxTools } from "../sandbox.mjs";
import { git } from "../workspaces.mjs";

const python = join(projectDir, ".venv/bin/python");
const fixture = t => {
  const dir = mkdtempSync(join(projectDir, "out/checkpoint-fixture-"));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  return dir;
};
test("checkpoint restores Git, ignored files, dependencies, home/tmp and installed console scripts at a new host path", { timeout: 180_000 }, async t => {
  const dir = fixture(t), source = join(dir, "source"), destination = join(dir, "restored");
  const cwd = join(source, "workspace"), venv = join(source, "python-env");
  mkdirSync(cwd, { recursive: true });
  // A commit hook supplies Git environment variables for the parent repository.
  // Use the isolated Git wrapper for every fixture operation, including restore.
  await git(cwd, "init", "-q");
  await git(cwd, "config", "user.name", "Developer");
  await git(cwd, "config", "user.email", "developer@localhost");
  writeFileSync(join(cwd, ".gitignore"), "ignored.txt\n");
  await git(cwd, "add", ".gitignore");
  await git(cwd, "commit", "-qm", "Initialize project");
  writeFileSync(join(cwd, "ignored.txt"), "retained");
  writeFileSync(join(cwd, "module.py"), "value = 42\n");
  symlinkSync("/home/developer/.local/share/node/node_modules", join(cwd, "node_modules"));
  execFileSync("uv", ["venv", "--python", python, venv], { stdio: "pipe" });
  const version = execFileSync(python, ["-c", "import sys;print(f'python{sys.version_info.major}.{sys.version_info.minor}')"], { encoding: "utf8" }).trim();
  writeFileSync(join(venv, "lib", version, "site-packages", "project.pth"), cwd + "\n");
  const secret = join(dir, "host-secret");
  writeFileSync(secret, cwd + "\nPRIVATE_VALUE=not-for-tools\n");
  symlinkSync(secret, join(venv, "lib", version, "site-packages", "host.pth"));
  const prepared = { pythonEnvironment: { path: venv }, toolEnvironment: { mounts: [], path: [], env: {} } };
  const bash = sandboxTools(cwd, join(source, "tool-state"), 10, venv, prepared.toolEnvironment).find(tool => tool.name === "bash");
  const result = await bash.execute("initial", { command: "python -c 'import module;assert module.value == 42'\nprintf '#!/bin/sh\\necho installed\\n' > \"$VIRTUAL_ENV/bin/task-command\"\nchmod +x \"$VIRTUAL_ENV/bin/task-command\"\necho temporary > /tmp/task.txt\necho home > ~/notes.txt" });
  assert.equal(result.isError, undefined);
  assert.equal(existsSync(join(source, "tool-state/runtime/python/task/metadata", version, "host.pth")), false);
  const status = await git(cwd, "status", "--porcelain");
  const receipt = await snapshotWorkspace(source, join(dir, "turn"), { status });
  assert.deepEqual(receipt.compression, { program: "gzip", level: 1 });
  assert.equal(receipt.archiveTimeoutSeconds, 1800);
  assert.equal(readFileSync(join(dir, receipt.archive))[8], 4); // gzip's fastest-compression header flag.
  const restored = await restoreCheckpoint(join(dir, receipt.archive), destination, { sha256: receipt.sha256, sourceCaseDir: source, prepared });
  t.after(() => rmSync(dirname(restored.destination), { recursive: true, force: true }));
  assert.doesNotMatch(restored.destination, /reasonese|\.codex/);
  assert.equal(readFileSync(join(restored.workspace, "ignored.txt"), "utf8"), "retained");
  assert.equal(readlinkSync(join(restored.workspace, "node_modules")), "/home/developer/.local/share/node/node_modules");
  assert.equal(await git(restored.workspace, "status", "--porcelain"), status);
  // Original host paths no longer exist. Editable imports and generated console
  // scripts must work using only the restored mutable environment.
  rmSync(source, { recursive: true, force: true });
  const resumed = sandboxTools(restored.workspace, join(destination, "tool-state"), 10, restored.prepared.pythonEnvironment.path, restored.prepared.toolEnvironment).find(tool => tool.name === "bash");
  const output = await resumed.execute("restored", { command: "python -c 'import module;print(module.value)'\ntask-command\ncat /tmp/task.txt ~/notes.txt" });
  assert.match(output.content[0].text, /42\ninstalled\ntemporary\nhome/);
  assert.equal(existsSync(source), false);
});

test("restore rejects corrupted archives and unsafe tar members without writing outside its private root", async t => {
  const dir = fixture(t), archive = join(dir, "unsafe.tar.gz");
  for (const mode of ["traversal", "symlink_parent", "duplicate", "device"]) {
    execFileSync(python, ["-c", `import io,sys,tarfile\np,mode=sys.argv[1:]\nwith tarfile.open(p,'w:gz') as t:\n a=tarfile.TarInfo('workspace');a.type=tarfile.DIRTYPE;t.addfile(a)\n if mode=='symlink_parent':\n  a=tarfile.TarInfo('workspace/link');a.type=tarfile.SYMTYPE;a.linkname='../../outside';t.addfile(a)\n names=['workspace/../../outside'] if mode=='traversal' else ['workspace/link/file'] if mode=='symlink_parent' else ['workspace/file']*2 if mode=='duplicate' else ['workspace/device']\n for n in names:\n  a=tarfile.TarInfo(n)\n  if mode=='device':a.type=tarfile.CHRTYPE\n  else:a.size=1\n  t.addfile(a,io.BytesIO(b'x'))`, archive, mode]);
    const sha256 = createHash("sha256").update(readFileSync(archive)).digest("hex");
    await assert.rejects(restoreCheckpoint(archive, join(dir, mode), { sha256 }));
    if (existsSync(join(dir, mode))) {
      const { realpathSync } = await import("node:fs");
      const physical = realpathSync(join(dir, mode));
      t.after(() => rmSync(dirname(physical), { recursive: true, force: true }));
    }
    assert.equal(existsSync(join(dir, "outside")), false);
  }
  await assert.rejects(restoreCheckpoint(archive, join(dir, "corrupt"), { sha256: "0".repeat(64) }), /hash mismatch/);
  assert.equal(existsSync(join(dir, "corrupt")), false);
});
