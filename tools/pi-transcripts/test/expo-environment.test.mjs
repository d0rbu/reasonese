import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import test from "node:test";
import { expoWorkspaceEnvironment, EXPO_NODE_VERSION, EXPO_PNPM_VERSION, validateEnvironment, validateNodeIdentity } from "../environments.mjs";
import { restoreCheckpoint, snapshotWorkspace } from "../checkpoints.mjs";
import { projectDir, sandboxArgs } from "../sandbox.mjs";

const hash = bytes => createHash("sha256").update(bytes).digest("hex");
function fixture(t) {
  mkdirSync(join(projectDir, "out"), { recursive: true });
  const root = mkdtempSync(join(projectDir, "out/expo-environment-test-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const cwd = join(root, "workspace"), path = join(root, "node-env/runtime/node"), binary = join(path, "bin/node");
  mkdirSync(cwd, { recursive: true }); mkdirSync(dirname(binary), { recursive: true });
  // Identity/restoration fixture only; it is never executed as a Node process.
  writeFileSync(binary, "offline pinned runtime bytes", { mode: 0o755 });
  const runtime = { path, version: EXPO_NODE_VERSION, executableSha256: hash(readFileSync(binary)) };
  return { root, cwd, runtime, binary };
}

test("Expo uses upstream workspace tools and native Jest config while keeping the controller runtime explicit", t => {
  const f = fixture(t), environment = expoWorkspaceEnvironment(f.root, f.cwd, f.runtime);
  validateEnvironment({ nodeProject: "packages/@expo/env" });
  assert.throws(() => validateEnvironment({ nodeProject: "packages/other" }), /required together/);
  assert.equal(environment.dependencies.node.version, EXPO_NODE_VERSION);
  assert.equal(environment.dependencies.node.controllerVersion, process.version);
  assert.equal(environment.dependencies.node.packageManager.version, EXPO_PNPM_VERSION);
  assert.equal(environment.env.JEST_CONFIG, join(f.cwd, "packages/@expo/env/jest.config.js"));
  assert.equal(environment.baselineEnv.PI_NODE_MODULES, join(f.cwd, "node_modules"));
  assert.equal(environment.dependencies.node.packages, undefined);
  assert.match(readFileSync(join(f.root, "node-env/bin/npm"), "utf8"), /\/opt\/task-node\/lib\/node_modules\/npm\/bin\/npm-cli.js/);
  const args = sandboxArgs(f.cwd, join(f.root, "scratch"), undefined, environment);
  const target = args.lastIndexOf("/opt/bin/node");
  assert.equal(args[target + 1], "/opt/pi/tool-worker.mjs");
  assert.ok(args.some((value, i) => value === f.binary && args[i + 1] === "/opt/bin/node"));
  assert.equal(validateNodeIdentity(environment, f.root), environment.dependencies.node);
});

test("task runtime identity rejects changed bytes, escaped binaries and controller drift; old sources keep their contract", t => {
  const f = fixture(t), environment = expoWorkspaceEnvironment(f.root, f.cwd, f.runtime);
  assert.equal(validateNodeIdentity({ dependencies: { node: { version: process.version } } }, f.root), null);
  assert.throws(() => validateNodeIdentity({ dependencies: { node: { version: "v0.0.0" } } }, f.root), /Node runtime changed/);
  const changed = structuredClone(environment);
  changed.dependencies.node.controllerVersion = "v0.0.0";
  assert.throws(() => validateNodeIdentity(changed, f.root), /task Node runtime changed/);
  changed.dependencies.node.controllerVersion = process.version;
  changed.mounts.find(m => m.target === "/opt/bin/node").source = process.execPath;
  assert.throws(() => validateNodeIdentity(changed, f.root), /task Node runtime changed/);
  const manager = environment.dependencies.node.packageManager;
  manager.executable = join(manager.path, "pnpm");
  mkdirSync(manager.path, { recursive: true }); writeFileSync(manager.executable, "pinned package manager");
  manager.executableSha256 = hash(readFileSync(manager.executable));
  validateNodeIdentity(environment, f.root);
  writeFileSync(manager.executable, "changed package manager");
  assert.throws(() => validateNodeIdentity(environment, f.root), /task package manager changed/);
  writeFileSync(manager.executable, "pinned package manager");
  writeFileSync(f.binary, "changed runtime bytes");
  assert.throws(() => validateNodeIdentity(environment, f.root), /task Node runtime changed/);
});

test("full checkpoints retain and relocate task Node, pnpm, and workspace dependency state without the source case", async t => {
  const f = fixture(t), environment = expoWorkspaceEnvironment(f.root, f.cwd, f.runtime);
  const manager = environment.dependencies.node.packageManager.path;
  mkdirSync(manager, { recursive: true }); writeFileSync(join(manager, "package-lock.json"), "retained pnpm lock");
  writeFileSync(join(manager, "pnpm"), "retained package-manager binary");
  Object.assign(environment.dependencies.node.packageManager, {
    executable: join(manager, "pnpm"), executableSha256: hash(readFileSync(join(manager, "pnpm"))),
  });
  mkdirSync(join(f.cwd, "node_modules")); writeFileSync(join(f.cwd, "node_modules/package.txt"), "retained workspace dependency");
  const prepared = { toolEnvironment: environment };
  const checkpoint = await snapshotWorkspace(f.root, join(f.root, "checkpoint"));
  const restored = await restoreCheckpoint(join(f.root, checkpoint.archive), join(f.root, "restored"), {
    sha256: checkpoint.sha256, sourceCaseDir: f.root, prepared,
  });
  t.after(() => rmSync(dirname(restored.destination), { recursive: true, force: true }));
  const moved = restored.prepared.toolEnvironment;
  assert.equal(moved.dependencies.node.runtime.path, join(restored.destination, "node-env/runtime/node"));
  assert.equal(moved.dependencies.node.packageManager.path, join(restored.destination, "node-env/package-manager"));
  assert.equal(moved.dependencies.node.packageManager.executable, join(restored.destination, "node-env/package-manager/pnpm"));
  assert.equal(realpathSync(moved.mounts.find(m => m.target === "/opt/bin/node").source), join(restored.destination, "node-env/runtime/node/bin/node"));
  rmSync(join(f.root, "node-env"), { recursive: true }); rmSync(f.cwd, { recursive: true });
  assert.equal(validateNodeIdentity(moved, restored.destination), moved.dependencies.node);
  assert.equal(readFileSync(join(restored.workspace, "node_modules/package.txt"), "utf8"), "retained workspace dependency");
  assert.equal(readFileSync(join(moved.dependencies.node.packageManager.path, "package-lock.json"), "utf8"), "retained pnpm lock");
});
