import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { closeSync, existsSync, mkdirSync, openSync, readFileSync, readSync, readdirSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { dirname, isAbsolute, join, relative, sep } from "node:path";
import { promisify } from "node:util";
import { projectDir, sandboxGit, sandboxTools, toolDir } from "./sandbox.mjs";
import { neutralRuntime, preparedCase } from "./backing.mjs";
import { prepareEnvironment, validateEnvironment } from "./environments.mjs";

const exec = promisify(execFile);
export const MODEL = "deepseek/deepseek-v4.1-flash-1017";
export const KIMI_MODEL = "moonshotai/kimi-k3.1-pro";
export const PI_VERSION = "0.73.1";
export const sha256 = (value) => createHash("sha256").update(value).digest("hex");
export function sha256File(filename) {
  const hash = createHash("sha256"), buffer = Buffer.alloc(1024 * 1024), fd = openSync(filename, "r");
  try { let size; while ((size = readSync(fd, buffer, 0, buffer.length, null))) hash.update(buffer.subarray(0, size)); }
  finally { closeSync(fd); }
  return hash.digest("hex");
}
export const readJson = (filename) => JSON.parse(readFileSync(filename, "utf8"));
export const writeJson = (filename, value) => writeFileSync(filename, JSON.stringify(value, null, 2) + "\n");

export async function git(cwd, ...args) {
  const result = await exec("git", ["-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false", "-c", "core.autocrlf=false", ...args], {
    cwd, maxBuffer: 32 * 1024 * 1024, timeout: 1_800_000,
    env: { PATH: "/usr/bin:/bin", LANG: "C.UTF-8", GIT_TERMINAL_PROMPT: "0",
      GIT_CONFIG_NOSYSTEM: "1", GIT_CONFIG_GLOBAL: "/dev/null" },
  });
  return result.stdout;
}

export function validateBaseCollection(manifest, scenarios = manifest.scenarios) {
  if (manifest.collectionPurpose !== "base_transcript") {
    throw new Error("Use a base_transcript manifest; historical conflict manifests are retained for provenance only");
  }
  for (const scenario of scenarios) {
    if (typeof scenario.baseInstruction !== "string" || !scenario.baseInstruction.trim()) {
      throw new Error("Each base transcript needs one baseInstruction");
    }
    for (const key of ["conflictInstructions", "intendedFraming", "decisiveCase"]) {
      if (key in scenario) throw new Error(`${key} belongs to later experiments, not base collection`);
    }
  }
}

export function loadManifest(filename) {
  const manifest = readJson(filename);
  if (![MODEL, KIMI_MODEL].includes(manifest.model) || manifest.thinking !== "high") throw new Error("Unexpected model or thinking setting");
  for (const key of ["maxModelCalls", "maxOutputTokens", "maxRequestBytes", "timeoutSeconds", "toolTimeoutSeconds"]) {
    if (!Number.isSafeInteger(manifest[key]) || manifest[key] <= 0) throw new Error(`Invalid ${key}`);
  }
  if (!Array.isArray(manifest.scenarios) || !manifest.scenarios.length) throw new Error("No scenarios");
  validateBaseCollection(manifest);
  const ids = new Set();
  for (const scenario of manifest.scenarios) {
    if (typeof scenario.id !== "string" || !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(scenario.id) || ids.has(scenario.id)) throw new Error("Invalid or duplicate scenario id");
    ids.add(scenario.id);
    if (!/^https:\/\/github\.com\/[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+\.git$/.test(scenario.repository)) throw new Error("Expected a public GitHub repository");
    if (!/^[a-f0-9]{40}$/.test(scenario.revision)) throw new Error("Repository revision must be a full commit SHA");
    if (typeof scenario.prompt !== "string" || !scenario.prompt.trim()) throw new Error("Empty user prompt");
    if (typeof scenario.ruleFile !== "string" || isAbsolute(scenario.ruleFile) || scenario.ruleFile.split(/[\\/]/).includes("..")) throw new Error("Invalid rule file");
    if (scenario.pythonProject !== undefined && scenario.pythonProject !== "libs/core") throw new Error("Unsupported Python project");
    validateEnvironment(scenario);
    if (scenario.maxUserTurns !== undefined && (!Number.isSafeInteger(scenario.maxUserTurns) || scenario.maxUserTurns < 1 || scenario.maxUserTurns > 8)) throw new Error("Invalid maxUserTurns");
  }
  return manifest;
}

export function collectorIdentity() {
  for (const name of ["pi-coding-agent", "pi-ai", "pi-agent-core", "pi-tui"]) {
    const installed = readJson(join(toolDir, "node_modules/@mariozechner", name, "package.json"));
    if (installed.version !== PI_VERSION) throw new Error(`Unexpected installed ${name}; run npm ci`);
  }
  return {
    pi: PI_VERSION, node: process.version, runtime: neutralRuntime(),
    files: Object.fromEntries(["package.json", "package-lock.json", "collect.mjs", "workspaces.mjs", "environments.mjs", "trajectory.mjs", "sandbox.mjs", "tool-worker.mjs", "network-worker.mjs", "bwrap-args.mjs", "backing.mjs", "checkpoints.mjs", "restore-checkpoint.py", ...readdirSync(toolDir).filter(n => n.startsWith("baseline_")).sort()].map((name) =>
      [name, sha256(readFileSync(join(toolDir, name)))])),
    pythonLock: sha256(readFileSync(join(projectDir, "uv.lock"))),
  };
}

export function contextFile(cwd, scenario) {
  const filename = join(cwd, scenario.ruleFile);
  const actual = realpathSync(filename);
  if (!actual.startsWith(cwd + sep)) throw new Error("Rule file escapes checkout");
  const content = readFileSync(actual, "utf8");
  // Match Pi's root AGENTS/CLAUDE discovery, including real Git symlinks.
  const entry = ["AGENTS.md", "AGENTS.MD", "CLAUDE.md", "CLAUDE.MD"]
    .map((name) => join(cwd, name)).find(existsSync);
  if (!entry || realpathSync(entry) !== actual) throw new Error("Pinned rule file differs from Pi's project context");
  return { path: entry, resolvedPath: relative(cwd, actual), content, sha256: sha256(content) };
}

export async function verifyWorkspace(cwd, scenario) {
  const check = (...args) => sandboxGit(cwd, join(dirname(cwd), "tool-state"), ...args);
  if ((await check("rev-parse", "HEAD")).trim() !== scenario.revision) throw new Error("Checkout revision changed");
  if ((await check("status", "--porcelain=v1", "--untracked-files=all")).trim()) throw new Error("Checkout is not pristine");
  const context = contextFile(cwd, scenario);
  const original = await check("show", `${scenario.revision}:${scenario.ruleFile}`);
  if (context.content !== original) throw new Error("Repository instructions changed");
  return context;
}

export async function fetchSource(scenario, sourceDirectory) {
  const cache = join(sourceDirectory, scenario.repository.split("/").slice(-2).join("-") + "-" + scenario.revision);
  if (!existsSync(cache)) {
    mkdirSync(cache, { recursive: true });
    await git(cache, "init", "--bare");
    await git(cache, "remote", "add", "origin", scenario.repository);
  }
  const fetched = await git(cache, "rev-parse", "--verify", "FETCH_HEAD").catch(() => "");
  if (fetched.trim() !== scenario.revision) {
    console.log(`Fetching ${scenario.repository} at ${scenario.revision}`);
    await git(cache, "fetch", "--depth=1", "--no-tags", "origin", scenario.revision);
  }
  // A FETCH_HEAD alone is not advertised to local clones.
  await git(cache, "update-ref", "refs/heads/source", scenario.revision);
  await git(cache, "symbolic-ref", "HEAD", "refs/heads/source");
  return cache;
}

export async function prepare(manifest, output, sourceDirectory = join(dirname(output), "pi-transcript-sources")) {
  validateBaseCollection(manifest);
  mkdirSync(output, { recursive: true });
  const planFile = join(output, "plan.json");
  const plan = { manifest, collector: collectorIdentity() };
  if (manifest.scenarios.some((scenario) => existsSync(join(output, scenario.id, "trajectory")))) {
    throw new Error("Collection already started; prepare a fresh run");
  }
  if (existsSync(planFile)) {
    const previous = readJson(planFile);
    if (JSON.stringify(previous) !== JSON.stringify(plan)) {
      if (JSON.stringify(previous.manifest) !== JSON.stringify(manifest)) throw new Error("Plan changed; use a fresh output directory");
      // Preparation can refresh an unstarted run after code review; keep the earlier plan.
      writeJson(join(output, `plan-${sha256(JSON.stringify(previous))}.json`), previous);
      writeJson(planFile, plan);
    }
  } else writeJson(planFile, plan);
  const unique = new Map(manifest.scenarios.map((scenario) => [scenario.repository + scenario.revision, scenario]));
  const queue = [...unique.values()];
  const caches = new Map();
  const fetched = await Promise.allSettled(Array.from({ length: Math.min(3, queue.length) }, async () => {
    while (queue.length) {
      const scenario = queue.shift();
      caches.set(scenario.repository + scenario.revision, await fetchSource(scenario, sourceDirectory));
    }
  }));
  const failure = fetched.find((result) => result.status === "rejected");
  if (failure) throw failure.reason;
  for (const scenario of manifest.scenarios) {
    const caseDir = preparedCase(join(output, scenario.id));
    const cwd = join(caseDir, "workspace");
    const cache = caches.get(scenario.repository + scenario.revision);
    mkdirSync(caseDir, { recursive: true });
    if (!existsSync(cwd)) {
      console.log(`Preparing ${scenario.id} at ${scenario.revision}`);
      // Independent Git storage: later agent commits cannot mutate another scenario or cache.
      await git(caseDir, "clone", "--no-hardlinks", "--no-checkout", cache, cwd);
      await git(cwd, "checkout", "--detach", scenario.revision);
      await git(cwd, "remote", "set-url", "origin", scenario.repository);
      await git(cwd, "config", "commit.gpgSign", "false");
      await git(cwd, "config", "core.hooksPath", "/dev/null");
    }
    await git(cwd, "config", "user.name", "Developer");
    await git(cwd, "config", "user.email", "developer@localhost");
    // Local-cache cloning records private host paths/identity in setup reflogs.
    // No trajectory has started here; subsequent assistant Git work gets normal
    // reflogs with the developer identity and neutral tool-visible paths.
    rmSync(join(cwd, ".git/logs"), { recursive: true, force: true });
    await git(cwd, "config", "core.logAllRefUpdates", "true");
    const context = await verifyWorkspace(cwd, scenario);
    for (const source of scenario.sourceFiles ?? []) {
      const path = realpathSync(join(cwd, source));
      if (!path.startsWith(cwd + sep)) throw new Error(`Source escapes checkout: ${source}`);
    }
    const { pythonEnvironment, toolEnvironment } = await prepareEnvironment(scenario, caseDir, cwd);
    const command = scenario.baselineCommand ?? (scenario.goPackage ? `go test ${scenario.goPackage}` :
      "python --version && python -m pytest --version && git --version");
    let check;
    const baselineFile = join(caseDir, "baseline.json");
    if (existsSync(baselineFile) && readJson(baselineFile).error) {
      const failed = readFileSync(baselineFile);
      writeFileSync(join(caseDir, `baseline-failed-${sha256(failed)}.json`), failed);
    }
    try {
      check = await sandboxTools(cwd, join(caseDir, "setup-tool-state"), Math.max(600, manifest.toolTimeoutSeconds), pythonEnvironment?.path, { ...toolEnvironment, preparation: true })
        .find((tool) => tool.name === "bash").execute("preflight", { command });
      writeJson(join(caseDir, "baseline.json"), { command, result: check });
    } catch (error) {
      writeJson(join(caseDir, "baseline.json"), { command, error: error.message });
      throw new Error(`Baseline failed for ${scenario.id}: ${error.message}`);
    }
    await verifyWorkspace(cwd, scenario);
    writeJson(join(caseDir, "prepared.json"), {
      scenario, context, tree: (await git(cwd, "rev-parse", "HEAD^{tree}")).trim(),
      preflight: check, pythonEnvironment, toolEnvironment, preparedAt: new Date().toISOString(),
      checkout: scenario.nodeProject === "packages/@expo/env" ?
        "complete shallow checkout; locked pnpm workspace install and native @expo/env package gate; submodules and native apps not bootstrapped" :
        "complete shallow checkout; scoped dependencies as recorded; submodules and full application builds not installed",
    });
    console.log(`Baseline passed: ${scenario.id}`);
  }
  console.log(`Prepared ${manifest.scenarios.length} workspaces. No model requests made.`);
}
