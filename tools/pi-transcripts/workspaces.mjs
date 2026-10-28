import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, realpathSync, writeFileSync } from "node:fs";
import { dirname, isAbsolute, join, relative, sep } from "node:path";
import { promisify } from "node:util";
import { projectDir, sandboxGit, sandboxTools, toolDir } from "./sandbox.mjs";

const exec = promisify(execFile);
export const MODEL = "deepseek/deepseek-v4.1-flash-1017";
export const PI_VERSION = "0.73.1";
export const sha256 = (value) => createHash("sha256").update(value).digest("hex");
export const readJson = (filename) => JSON.parse(readFileSync(filename, "utf8"));
export const writeJson = (filename, value) => writeFileSync(filename, JSON.stringify(value, null, 2) + "\n");

export async function git(cwd, ...args) {
  const result = await exec("git", ["-c", "core.hooksPath=/dev/null", "-c", "core.autocrlf=false", ...args], {
    cwd, maxBuffer: 32 * 1024 * 1024, timeout: 1_800_000,
    env: { PATH: "/usr/bin:/bin", LANG: "C.UTF-8", GIT_TERMINAL_PROMPT: "0",
      GIT_CONFIG_NOSYSTEM: "1", GIT_CONFIG_GLOBAL: "/dev/null" },
  });
  return result.stdout;
}

export function loadManifest(filename) {
  const manifest = readJson(filename);
  if (manifest.model !== MODEL || manifest.thinking !== "high") throw new Error("Unexpected model or thinking setting");
  for (const key of ["maxModelCalls", "maxOutputTokens", "maxRequestBytes", "timeoutSeconds", "toolTimeoutSeconds"]) {
    if (!Number.isSafeInteger(manifest[key]) || manifest[key] <= 0) throw new Error(`Invalid ${key}`);
  }
  if (!Array.isArray(manifest.scenarios) || !manifest.scenarios.length) throw new Error("No scenarios");
  const ids = new Set();
  for (const scenario of manifest.scenarios) {
    if (typeof scenario.id !== "string" || !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(scenario.id) || ids.has(scenario.id)) throw new Error("Invalid or duplicate scenario id");
    ids.add(scenario.id);
    if (!/^https:\/\/github\.com\/[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+\.git$/.test(scenario.repository)) throw new Error("Expected a public GitHub repository");
    if (!/^[a-f0-9]{40}$/.test(scenario.revision)) throw new Error("Repository revision must be a full commit SHA");
    if (typeof scenario.prompt !== "string" || !scenario.prompt.trim()) throw new Error("Empty user prompt");
    if (typeof scenario.ruleFile !== "string" || isAbsolute(scenario.ruleFile) || scenario.ruleFile.split(/[\\/]/).includes("..")) throw new Error("Invalid rule file");
    if (scenario.pythonProject !== undefined && scenario.pythonProject !== "libs/core") throw new Error("Unsupported Python project");
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
    pi: PI_VERSION, node: process.version,
    files: Object.fromEntries(["package-lock.json", "collect.mjs", "workspaces.mjs", "trajectory.mjs", "sandbox.mjs", "tool-worker.mjs"].map((name) =>
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
    const caseDir = join(output, scenario.id);
    const cwd = join(caseDir, "workspace");
    const cache = caches.get(scenario.repository + scenario.revision);
    mkdirSync(caseDir, { recursive: true });
    if (!existsSync(cwd)) {
      console.log(`Preparing ${scenario.id} at ${scenario.revision}`);
      // Independent Git storage: later agent commits cannot mutate another scenario or cache.
      await git(caseDir, "clone", "--no-hardlinks", "--no-checkout", cache, cwd);
      await git(cwd, "checkout", "--detach", scenario.revision);
      await git(cwd, "remote", "set-url", "origin", scenario.repository);
      await git(cwd, "config", "user.name", "Transcript Collector");
      await git(cwd, "config", "user.email", "collector@example.invalid");
      await git(cwd, "config", "commit.gpgSign", "false");
      await git(cwd, "config", "core.hooksPath", "/dev/null");
    }
    const context = await verifyWorkspace(cwd, scenario);
    let pythonEnvironment;
    if (scenario.pythonProject) {
      const environment = join(caseDir, "python-env");
      const project = join(cwd, scenario.pythonProject);
      const installed = await exec("uv", ["sync", "--locked", "--no-default-groups", "--group", "test", "--project", project], {
        env: { ...process.env, UV_PROJECT_ENVIRONMENT: environment }, timeout: 1_800_000, maxBuffer: 16 * 1024 * 1024,
      });
      writeFileSync(join(caseDir, "python-setup.log"), installed.stdout + installed.stderr);
      const frozen = await exec("uv", ["pip", "freeze", "--python", join(environment, "bin/python")]);
      pythonEnvironment = { path: environment, project: scenario.pythonProject,
        lockSha256: sha256(readFileSync(join(project, "uv.lock"))), packages: frozen.stdout };
      await verifyWorkspace(cwd, scenario);
    }
    const check = await sandboxTools(cwd, join(caseDir, "tool-state"), manifest.toolTimeoutSeconds, pythonEnvironment?.path)
      .find((tool) => tool.name === "bash").execute("preflight", {
        command: "python --version && python -m pytest --version && git --version",
      });
    writeJson(join(caseDir, "prepared.json"), {
      scenario, context, tree: (await git(cwd, "rev-parse", "HEAD^{tree}")).trim(),
      preflight: check, pythonEnvironment, preparedAt: new Date().toISOString(),
      checkout: scenario.pythonProject ? "complete shallow checkout; locked core test environment installed; submodules not installed" : "complete shallow checkout; submodules and repository build dependencies are not installed",
    });
  }
  console.log(`Prepared ${manifest.scenarios.length} workspaces. No model requests made.`);
}
