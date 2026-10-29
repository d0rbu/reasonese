// Dependency setup is explicit and runs before collection; tools retain public network access.
import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, linkSync, lstatSync, mkdirSync, mkdtempSync, readFileSync, readlinkSync, readdirSync, realpathSync, rmSync, statfsSync, symlinkSync, unlinkSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, isAbsolute, join } from "node:path";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";
import { backingRoot } from "./backing.mjs";
import { sandboxTools } from "./sandbox.mjs";

const exec = promisify(execFile);
const hash = (value) => createHash("sha256").update(value).digest("hex");
const GO_VERSION = "1.26.9";
const GO_SHA256 = "42d158b4d8f7b61ac0a830567c940a86098fb7aac52e467a5ebec03ef5cc2f8d";
export const UV_VERSION = "0.13.6";
export const UV_WHEEL_URL = "https://files.pythonhosted.org/packages/5c/43/0688ff5142934238f2812e348f4a24e09b57d48bfb38dd8f850325a0d9de/uv-0.13.6-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl";
export const UV_WHEEL_SHA256 = "22759b8895e1103578de6fb4cd27e94b4399d021023378d29f01333717751fba";
export const UV_BINARY_SHA256 = "23803452c0fbe125033054cd9ff245f1d649fb65b4b73f68429f4abda432436d";
export const EXPO_NODE_VERSION = "v24.22.2";
export const EXPO_NODE_URL = `https://nodejs.org/dist/${EXPO_NODE_VERSION}/node-${EXPO_NODE_VERSION}-linux-x64.tar.xz`;
export const EXPO_NODE_SHA256 = "48cc0e21e048790e54c0cf5749c2725005313d652651c40b7a21feb9697aaf75";
export const EXPO_PNPM_VERSION = "12.2.1";
const MIN_FREE_BYTES = 40 * 1024 ** 3;
const relativePath = (value) => typeof value === "string" && value.length > 0 &&
  !isAbsolute(value) && !value.split(/[\\/]/).includes("..");

export function validateEnvironment(scenario) {
  for (const key of ["pythonEditable", "nodeProject", "goPackage"]) {
    if (scenario[key] !== undefined && !relativePath(scenario[key])) throw new Error(`Invalid ${key}`);
  }
  if (scenario.goPackage !== undefined && !/^\.\/[A-Za-z0-9_/-]+$/.test(scenario.goPackage)) throw new Error("Expected a relative Go package directory");
  for (const key of ["sourceFiles", "pythonPath"]) {
    if (scenario[key] !== undefined && (!Array.isArray(scenario[key]) || !scenario[key].every(relativePath))) throw new Error(`Invalid ${key}`);
  }
  if (scenario.pythonProject && scenario.pythonEditable) throw new Error("Choose one Python environment");
  for (const key of ["pythonPackages", "nodePackages"]) {
    const pattern = key === "pythonPackages" ? /^[A-Za-z0-9_.-]+==[0-9][A-Za-z0-9.+-]*$/ : /^(?:@[a-z0-9_.-]+\/)?[a-z0-9_.-]+@[0-9]+\.[0-9]+\.[0-9]+$/;
    if (scenario[key] !== undefined && (!Array.isArray(scenario[key]) || !scenario[key].length || !scenario[key].every(v => typeof v === "string" && pattern.test(v)))) throw new Error(`Pin exact ${key}`);
  }
  if (scenario.pythonPackages && !scenario.pythonEditable) throw new Error("pythonPackages requires pythonEditable");
  if (scenario.nodeProject !== "packages/@expo/env" && Boolean(scenario.nodePackages) !== Boolean(scenario.nodeProject)) throw new Error("nodePackages and nodeProject are required together");
  if (scenario.bunVersion !== undefined && scenario.bunVersion !== "1.3.12") throw new Error("Unsupported Bun version");
  if (scenario.baselineCommand !== undefined && (typeof scenario.baselineCommand !== "string" || !scenario.baselineCommand.trim())) throw new Error("Empty baselineCommand");
}

function verifiedFile(path, expected, label) {
  let stat;
  try { stat = lstatSync(path); } catch (error) { if (error.code === "ENOENT") return false; throw error; }
  if (!stat.isFile() || hash(readFileSync(path)) !== expected) throw new Error(`${label} checksum mismatch: ${path}`);
  return true;
}
function publishVerifiedFile(path, bytes, expected, label, mode = 0o644) {
  if (hash(bytes) !== expected) throw new Error(`${label} checksum mismatch`);
  mkdirSync(dirname(path), { recursive: true });
  const stage = mkdtempSync(join(dirname(path), ".verified-"));
  try {
    const candidate = join(stage, "artifact");
    writeFileSync(candidate, bytes, { mode });
    // Publish a complete file without overwriting a concurrent preparation.
    try { linkSync(candidate, path); } catch (error) { if (error.code !== "EEXIST") throw error; }
    verifiedFile(path, expected, label);
  } finally { rmSync(stage, { recursive: true, force: true }); }
}
export async function downloadVerifiedFile(path, url, expected, fetcher = fetch) {
  if (verifiedFile(path, expected, "Cached download")) return path;
  const response = await fetcher(url, { signal: AbortSignal.timeout(120_000) });
  if (!response.ok) throw new Error(`Dependency download: HTTP ${response.status}`);
  publishVerifiedFile(path, Buffer.from(await response.arrayBuffer()), expected, "Downloaded artifact");
  return path;
}
export async function prepareUv({ cacheRoot = join(backingRoot, "runtimes"), fetcher = fetch } = {}) {
  if (process.platform !== "linux" || process.arch !== "x64") throw new Error("Prepared uv runtime requires Linux amd64");
  const root = join(cacheRoot, `uv-${UV_VERSION}`), binary = join(root, "bin/uv");
  const hasBinary = verifiedFile(binary, UV_BINARY_SHA256, "Cached uv binary");
  const wheel = join(root, UV_WHEEL_URL.split("/").at(-1));
  await downloadVerifiedFile(wheel, UV_WHEEL_URL, UV_WHEEL_SHA256, fetcher);
  if (!hasBinary) {
    const script = `import sys,zipfile\nwith zipfile.ZipFile(sys.argv[1]) as wheel: sys.stdout.buffer.write(wheel.read('uv-${UV_VERSION}.data/scripts/uv'))`;
    const extracted = await exec("/usr/bin/python3", ["-c", script, wheel], {
      encoding: "buffer", maxBuffer: 128 * 1024 * 1024, timeout: 120_000,
      env: { PATH: "/usr/bin:/bin", LANG: "C.UTF-8" },
    });
    publishVerifiedFile(binary, extracted.stdout, UV_BINARY_SHA256, "Extracted uv binary", 0o755);
  }
  const { stdout } = await exec(binary, ["--version"], { timeout: 10_000,
    env: { PATH: "/usr/bin:/bin", LANG: "C.UTF-8" } });
  const observedVersion = stdout.trim();
  if (observedVersion !== `uv ${UV_VERSION} (x86_64-unknown-linux-gnu)`) throw new Error(`Unexpected uv version: ${observedVersion}`);
  return { path: binary, version: UV_VERSION, observedVersion, wheelUrl: UV_WHEEL_URL,
    wheelSha256: UV_WHEEL_SHA256, executableSha256: UV_BINARY_SHA256,
    modelAccess: "read-only runtime mount; preserve wheel and executable with source evidence" };
}

function requireFreeSpace(path) {
  const space = statfsSync(path);
  if (space.bavail * space.bsize < MIN_FREE_BYTES) throw new Error("Expo preparation requires at least 40 GiB free disk space");
}

// A task-owned Node can differ from the controller SDK runtime. Older captures
// without a task runtime keep their original controller-version contract.
export function validateNodeIdentity(environment, caseDir) {
  const identity = environment?.dependencies?.node;
  if (!identity?.runtime) {
    if (identity?.version && identity.version !== process.version) throw new Error("Prepared Node runtime changed");
    return null;
  }
  const runtime = identity.runtime;
  const root = realpathSync(runtime.path), task = realpathSync(join(caseDir, "node-env"));
  const binary = realpathSync(join(root, "bin/node"));
  const mount = environment.mounts?.find(item => item.target === "/opt/bin/node");
  if (!root.startsWith(task + "/") || !binary.startsWith(root + "/") ||
      !mount || realpathSync(mount.source) !== binary ||
      !runtime.executableSha256 || hash(readFileSync(binary)) !== runtime.executableSha256 ||
      runtime.version !== identity.version || identity.controllerVersion !== process.version) throw new Error("Prepared task Node runtime changed");
  const manager = identity.packageManager;
  if (manager?.executable && (!realpathSync(manager.executable).startsWith(task + "/") ||
      hash(readFileSync(manager.executable)) !== manager.executableSha256)) throw new Error("Prepared task package manager changed");
  return identity;
}

export function expoWorkspaceEnvironment(caseDir, cwd, runtime) {
  const prefix = join(caseDir, "node-env"), visible = "/home/developer/.local/share/node";
  const bin = join(prefix, "bin"), manager = join(prefix, "package-manager");
  mkdirSync(bin, { recursive: true });
  const mounts = [{ source: prefix, target: visible, writable: true },
    { source: runtime.path, target: "/opt/task-node" },
    { source: join(runtime.path, "bin/node"), target: "/opt/bin/node" }];
  for (const [name, entry] of Object.entries({
    npm: "/opt/task-node/lib/node_modules/npm/bin/npm-cli.js",
    npx: "/opt/task-node/lib/node_modules/npm/bin/npx-cli.js",
    pnpm: `${visible}/package-manager/node_modules/pnpm/bin/pnpm.mjs`,
  })) {
    const script = join(bin, name);
    writeFileSync(script, `#!/bin/sh\nexec /opt/bin/node ${entry} "$@"\n`, { mode: 0o755 });
    mounts.push({ source: script, target: `/opt/bin/${name}` });
  }
  const config = join(cwd, "packages/@expo/env/jest.config.js");
  return { mounts, path: [join(cwd, "node_modules/.bin")],
    env: { JEST_CONFIG: config, npm_config_cache: join(prefix, "cache/npm"), PNPM_HOME: join(prefix, "pnpm-home") },
    baselineEnv: { PI_NODE_MODULES: join(cwd, "node_modules"), PI_JEST_CONFIG: config },
    dependencies: { node: { version: EXPO_NODE_VERSION, controllerVersion: process.version, runtime,
      packageManager: { name: "pnpm", version: EXPO_PNPM_VERSION, path: manager },
      resolution: "pinned upstream pnpm workspace lock and native package scripts; no custom Jest transform" } } };
}

export async function prepareExpoWorkspace(caseDir, cwd, pythonEnvironment, environment,
  { fetcher = fetch, cacheRoot = join(backingRoot, "downloads/node") } = {}) {
  if (process.platform !== "linux" || process.arch !== "x64") throw new Error("Expo task runtime requires Linux amd64");
  requireFreeSpace(caseDir);
  const prefix = join(caseDir, "node-env"), runtimePath = join(prefix, "runtime", `node-${EXPO_NODE_VERSION}`);
  if (existsSync(join(prefix, "package.json")) || existsSync(join(cwd, "packages/@expo/env/node_modules"))) throw new Error("Expo workspace setup needs a fresh case without the former focused Node environment");
  const archive = join(cacheRoot, EXPO_NODE_URL.split("/").at(-1));
  await downloadVerifiedFile(archive, EXPO_NODE_URL, EXPO_NODE_SHA256, fetcher);
  if (existsSync(runtimePath)) throw new Error("Expo task runtime already exists; retain the failed preparation and use a fresh case");
  mkdirSync(runtimePath, { recursive: true });
  await exec("tar", ["-xJf", archive, "--strip-components=1", "-C", runtimePath], {
    timeout: 120_000, env: { PATH: "/usr/bin:/bin", LANG: "C.UTF-8" } });
  const node = join(runtimePath, "bin/node"), version = (await exec(node, ["--version"])).stdout.trim();
  if (version !== EXPO_NODE_VERSION) throw new Error("Downloaded Expo Node version differs from its pin");
  const runtime = { path: runtimePath, version, executableSha256: hash(readFileSync(node)),
    archiveUrl: EXPO_NODE_URL, archiveSha256: EXPO_NODE_SHA256,
    modelAccess: "task-owned runtime included in full checkpoints; controller SDK runs separately" };
  const tools = expoWorkspaceEnvironment(caseDir, cwd, runtime);
  environment.mounts.push(...tools.mounts); environment.path.push(...tools.path);
  for (const key of ["env", "baselineEnv", "dependencies"]) Object.assign(environment[key], tools[key]);
  const bash = sandboxTools(cwd, join(caseDir, "setup-tool-state"), 1800, pythonEnvironment.path, environment).find(tool => tool.name === "bash");
  const run = async (label, command) => {
    requireFreeSpace(caseDir);
    const abort = new AbortController();
    let diskError;
    const timer = setInterval(() => { try { requireFreeSpace(caseDir); } catch (error) { diskError = error; abort.abort(); } }, 5000);
    const log = join(prefix, "setup-logs", `${label}.log`);
    mkdirSync(dirname(log), { recursive: true });
    const visibleLog = `/home/developer/.local/share/node/setup-logs/${label}.log`;
    try {
      const wrapped = `(${command}) > ${visibleLog} 2>&1; status=$?; tail -n 80 ${visibleLog}; exit "$status"`;
      const result = await bash.execute(label, { command: wrapped, timeout: 1800 }, abort.signal);
      const text = readFileSync(log, "utf8");
      if (diskError || result.isError) throw diskError ?? new Error(`${label} failed; see its retained setup log`);
      requireFreeSpace(caseDir);
      return text;
    } finally {
      clearInterval(timer);
      if (existsSync(log)) writeFileSync(join(caseDir, `${label}.log`), readFileSync(log));
    }
  };
  await run("expo-package-manager", `npm install --prefix /home/developer/.local/share/node/package-manager --ignore-scripts --no-audit --no-fund --save-exact pnpm@${EXPO_PNPM_VERSION}`);
  const managerVersion = (await run("expo-pnpm-version", "pnpm --version")).trim();
  if (managerVersion !== EXPO_PNPM_VERSION) throw new Error("Expo pnpm version differs from its pin");
  writeFileSync(join(caseDir, "expo-runtime.json"), JSON.stringify(environment.dependencies.node, null, 2) + "\n");
  await run("expo-workspace-install", "SKIP_SKIA_DOWNLOAD=1 pnpm install --frozen-lockfile --store-dir /home/developer/.local/share/node/cache/pnpm");
  await run("expo-package-check", "node ./tools/bin/expotools.js check-packages @expo/env");
  const identity = environment.dependencies.node;
  identity.packageManager.lockSha256 = hash(readFileSync(join(prefix, "package-manager/package-lock.json")));
  identity.packageManager.executable = join(prefix, "package-manager/node_modules/@pnpm/exe.linux-x64/pnpm");
  identity.packageManager.executableSha256 = hash(readFileSync(identity.packageManager.executable));
  identity.npmVersion = JSON.parse(readFileSync(join(runtimePath, "lib/node_modules/npm/package.json"), "utf8")).version;
  identity.lockSha256 = hash(readFileSync(join(cwd, "pnpm-lock.yaml")));
  identity.workspaceConfigSha256 = hash(readFileSync(join(cwd, "pnpm-workspace.yaml")));
  identity.jestConfigSha256 = hash(readFileSync(join(cwd, "packages/@expo/env/jest.config.js")));
  identity.packageCheckLogSha256 = hash(readFileSync(join(caseDir, "expo-package-check.log")));
  validateNodeIdentity(environment, caseDir);
  return environment;
}

export async function prepareEnvironment(scenario, caseDir, cwd) {
  const environment = { mounts: [], path: [], env: {}, baselineEnv: {}, dependencies: {} };
  const uv = await prepareUv();
  environment.mounts.push({ source: uv.path, target: "/opt/bin/uv" });
  environment.path.push("/opt/bin");
  environment.dependencies.uv = uv;
  environment.baselineEnv.PI_BASELINE_PROBE = fileURLToPath(new URL("baseline_probe.py", import.meta.url));
  environment.baselineEnv.PI_BASELINE_DIR = dirname(environment.baselineEnv.PI_BASELINE_PROBE);
  const setupHome = join(caseDir, "setup-home");
  mkdirSync(setupHome, { recursive: true });
  // Public dependency installation does not inherit provider credentials or user config.
  const setupEnv = { PATH: `${dirname(uv.path)}:${process.env.PATH}`, HOME: setupHome, LANG: "C.UTF-8",
    UV_CACHE_DIR: join(backingRoot, "downloads/uv"), UV_LINK_MODE: "copy",
    UV_PYTHON_INSTALL_DIR: join(backingRoot, "runtimes/python"),
    npm_config_cache: join(backingRoot, "downloads/npm") };
  const run = async (label, command, args, options = {}) => {
    try {
      const result = await exec(command, args, { cwd, env: setupEnv, timeout: 1_800_000,
        maxBuffer: 16 * 1024 * 1024, ...options });
      writeFileSync(join(caseDir, `${label}.log`), result.stdout + result.stderr);
      return result.stdout;
    } catch (error) {
      const log = (error.stdout ?? "") + (error.stderr ?? "") + error.message;
      writeFileSync(join(caseDir, `${label}.log`), log);
      writeFileSync(join(caseDir, `${label}-failed-${hash(log)}.log`), log);
      throw error;
    }
  };
  let pythonEnvironment;
  {
    const path = join(caseDir, "python-env");
    const project = join(cwd, scenario.pythonProject ?? scenario.pythonEditable ?? ".");
    if (scenario.pythonProject) {
      await run("python-setup", uv.path, ["sync", "--locked", "--no-default-groups", "--group", "test", "--project", project],
        { env: { ...setupEnv, UV_PROJECT_ENVIRONMENT: path } });
    } else {
      if (!existsSync(join(path, "bin/python"))) await run("python-venv", uv.path, ["venv", "--python", "3.13", path]);
      if (scenario.pythonEditable) await run("python-setup", uv.path, ["pip", "install", "--python", join(path, "bin/python"), "-e", project, ...(scenario.pythonPackages ?? [])]);
    }
    await run("python-development-tools", uv.path, ["pip", "install", "--python", join(path, "bin/python"), "pip==25.3", "ruff==0.15.2", ...(!scenario.pythonProject && !scenario.pythonEditable ? ["pytest==9.1.1"] : [])]);
    const packages = await run("python-freeze", uv.path, ["pip", "freeze", "--python", join(path, "bin/python")]);
    const executable = realpathSync(join(path, "bin/python"));
    const runtime = dirname(dirname(executable));
    const version = readFileSync(join(path, "pyvenv.cfg"), "utf8").match(/^version(?:_info)? = (.+)$/m)?.[1];
    pythonEnvironment = { path, runtime: { path: runtime, version,
      executableSha256: hash(readFileSync(executable)),
      sharedLibraries: readdirSync(join(runtime, "lib")).filter(name => /^libpython.*\.so(?:\..*)?$/.test(name))
        .map(name => ({ file: name, sha256: hash(readFileSync(join(runtime, "lib", name))) })),
      modelAccess: "read-only runtime mount; preserve this immutable runtime with source evidence" }, project: scenario.pythonProject ?? scenario.pythonEditable,
      lockSha256: scenario.pythonProject ? hash(readFileSync(join(project, "uv.lock"))) : undefined,
      packages, packagesSha256: hash(packages),
      resolution: scenario.pythonProject ? "upstream uv.lock plus pinned pip/ruff development tools" : scenario.pythonEditable ? "editable upstream package and exact test/tool package pins; transitive versions recorded in freeze" : "pinned development tools; transitive versions recorded in freeze" };
  }
  if (scenario.pythonPath) environment.env.PYTHONPATH = scenario.pythonPath.map(p => join(cwd, p)).join(":");

  if (scenario.goPackage) {
    if (process.platform !== "linux" || process.arch !== "x64") throw new Error("Prepared Go runtime requires Linux amd64");
    const runtime = join(backingRoot, "runtimes", `go-${GO_VERSION}`);
    const archive = runtime + ".tar.gz";
    mkdirSync(dirname(runtime), { recursive: true });
    if (!existsSync(archive)) {
      const response = await fetch(`https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz`);
      if (!response.ok) throw new Error(`Go download: ${response.status}`);
      writeFileSync(archive, Buffer.from(await response.arrayBuffer()));
    }
    if (hash(readFileSync(archive)) !== GO_SHA256) throw new Error("Go archive checksum mismatch");
    if (!existsSync(join(runtime, "go/bin/go"))) {
      mkdirSync(runtime, { recursive: true });
      await run("go-extract", "tar", ["-xzf", archive, "-C", runtime]);
    }
    const root = join(runtime, "go");
    const moduleCache = join(caseDir, "go-modules");
    mkdirSync(moduleCache, { recursive: true });
    const goEnv = { GOROOT: root, GOTOOLCHAIN: "local", GOMODCACHE: moduleCache,
      GOPATH: "/tmp/go-path", GOCACHE: "/tmp/go-build", GOFLAGS: "-mod=readonly -p=2", GOMAXPROCS: "2", CGO_ENABLED: "0" };
    // Resolve only this package's test dependency graph; do not run upstream tests on the host.
    await run("go-dependencies", join(root, "bin/go"), ["list", "-deps", "-test", scenario.goPackage],
      { env: { ...setupEnv, ...goEnv, GOCACHE: join(caseDir, "setup-go-cache"), GOPATH: join(caseDir, "setup-go-path") } });
    await run("go-cache-permissions", "chmod", ["-R", "u+w", moduleCache]);
    Object.assign(environment.env, goEnv, { GOPROXY: "https://proxy.golang.org,direct", GOSUMDB: "sum.golang.org", GOFLAGS: "-mod=readonly -modcacherw -p=2" });
    environment.mounts.push({ source: root, target: "/opt/go" }, { source: moduleCache, target: "/home/developer/go/pkg/mod", writable: true });
    environment.path.push(join(root, "bin"));
    environment.dependencies.go = { version: GO_VERSION, archiveSha256: GO_SHA256,
      executableSha256: hash(readFileSync(join(root, "bin/go"))),
      goModSha256: hash(readFileSync(join(cwd, "go.mod"))), goSumSha256: hash(readFileSync(join(cwd, "go.sum"))) };
  }
  if (scenario.bunVersion) {
    const bun = realpathSync(process.env.PI_BUN_PATH ?? join(homedir(), ".bun/bin/bun"));
    const version = (await run("bun-version", bun, ["--version"])).trim();
    if (version !== scenario.bunVersion) throw new Error(`Expected Bun ${scenario.bunVersion}, found ${version}`);
    environment.mounts.push({ source: bun, target: "/opt/bin/bun" });
    environment.path.push("/opt/bin");
    environment.dependencies.bun = { version, executableSha256: hash(readFileSync(bun)) };
  }
  if (scenario.nodeProject === "packages/@expo/env") {
    await prepareExpoWorkspace(caseDir, cwd, pythonEnvironment, environment);
  } else if (scenario.nodeProject) {
    const prefix = join(caseDir, "node-env");
    mkdirSync(prefix, { recursive: true });
    if (!existsSync(join(prefix, "package.json"))) writeFileSync(join(prefix, "package.json"), '{"private":true}\n');
    const locked = existsSync(join(prefix, "package-lock.json"));
    await run("node-setup", "npm", [locked ? "ci" : "install", "--prefix", prefix,
      "--ignore-scripts", "--no-audit", "--no-fund", ...(locked ? [] : ["--save-exact", ...scenario.nodePackages])]);
    const modules = join(prefix, "node_modules");
    environment.baselineEnv.PI_NODE_MODULES = modules;
    const link = join(cwd, scenario.nodeProject, "node_modules");
    if (!existsSync(link)) {
      try { unlinkSync(link); } catch (error) { if (error.code !== "ENOENT") throw error; }
      symlinkSync("/home/developer/.local/share/node/node_modules", link);
    }
    if (readlinkSync(link) !== "/home/developer/.local/share/node/node_modules") throw new Error("Unexpected node_modules directory");
    environment.mounts.push({ source: prefix, target: "/home/developer/.local/share/node", writable: true });
    environment.path.push(join(modules, ".bin"));
    environment.dependencies.node = { version: process.version, packages: scenario.nodePackages,
      lockSha256: hash(readFileSync(join(prefix, "package-lock.json"))) };
  }
  return { pythonEnvironment, toolEnvironment: environment };
}
