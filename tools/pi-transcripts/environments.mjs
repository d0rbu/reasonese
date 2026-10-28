// Dependency setup is explicit and runs before collection. Model tools remain offline.
import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, realpathSync, symlinkSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, isAbsolute, join } from "node:path";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";

const exec = promisify(execFile);
const hash = (value) => createHash("sha256").update(value).digest("hex");
const GO_VERSION = "1.26.9";
const GO_SHA256 = "42d158b4d8f7b61ac0a830567c940a86098fb7aac52e467a5ebec03ef5cc2f8d";
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
  if (Boolean(scenario.nodePackages) !== Boolean(scenario.nodeProject)) throw new Error("nodePackages and nodeProject are required together");
  if (scenario.bunVersion !== undefined && scenario.bunVersion !== "1.3.12") throw new Error("Unsupported Bun version");
  if (scenario.baselineCommand !== undefined && (typeof scenario.baselineCommand !== "string" || !scenario.baselineCommand.trim())) throw new Error("Empty baselineCommand");
}

export async function prepareEnvironment(scenario, caseDir, cwd) {
  const environment = { readOnlyPaths: [], path: [], env: {}, dependencies: {} };
  environment.env.PI_BASELINE_PROBE = fileURLToPath(new URL("baseline_probe.py", import.meta.url));
  environment.env.PI_BASELINE_DIR = dirname(environment.env.PI_BASELINE_PROBE);
  const setupHome = join(caseDir, "setup-home");
  mkdirSync(setupHome, { recursive: true });
  // Public dependency installation does not inherit provider credentials or user config.
  const setupEnv = { PATH: process.env.PATH, HOME: setupHome, LANG: "C.UTF-8",
    UV_CACHE_DIR: join(dirname(caseDir), "dependency-cache/uv"),
    npm_config_cache: join(dirname(caseDir), "dependency-cache/npm") };
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
  if (scenario.pythonProject || scenario.pythonEditable) {
    const path = join(caseDir, "python-env");
    const project = join(cwd, scenario.pythonProject ?? scenario.pythonEditable);
    if (scenario.pythonProject) {
      await run("python-setup", "uv", ["sync", "--locked", "--no-default-groups", "--group", "test", "--project", project],
        { env: { ...setupEnv, UV_PROJECT_ENVIRONMENT: path } });
    } else {
      if (!existsSync(join(path, "bin/python"))) await run("python-venv", "uv", ["venv", "--python", "3.13", path]);
      await run("python-setup", "uv", ["pip", "install", "--python", join(path, "bin/python"), "-e", project, ...(scenario.pythonPackages ?? [])]);
    }
    const packages = await run("python-freeze", "uv", ["pip", "freeze", "--python", join(path, "bin/python")]);
    pythonEnvironment = { path, project: scenario.pythonProject ?? scenario.pythonEditable,
      lockSha256: scenario.pythonProject ? hash(readFileSync(join(project, "uv.lock"))) : undefined,
      packages, packagesSha256: hash(packages),
      resolution: scenario.pythonProject ? "upstream uv.lock" : "editable upstream package and exact test package pins; transitive versions recorded in freeze" };
  }
  if (scenario.pythonPath) environment.env.PYTHONPATH = scenario.pythonPath.map(p => join(cwd, p)).join(":");

  if (scenario.goPackage) {
    if (process.platform !== "linux" || process.arch !== "x64") throw new Error("Prepared Go runtime requires Linux amd64");
    const runtime = join(dirname(caseDir), "runtimes", `go-${GO_VERSION}`);
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
    Object.assign(environment.env, goEnv, { GOPROXY: "off", GOSUMDB: "off" });
    environment.readOnlyPaths.push(root, moduleCache);
    environment.path.push(join(root, "bin"));
    environment.dependencies.go = { version: GO_VERSION, archiveSha256: GO_SHA256,
      executableSha256: hash(readFileSync(join(root, "bin/go"))),
      goModSha256: hash(readFileSync(join(cwd, "go.mod"))), goSumSha256: hash(readFileSync(join(cwd, "go.sum"))) };
  }
  if (scenario.bunVersion) {
    const bun = realpathSync(process.env.PI_BUN_PATH ?? join(homedir(), ".bun/bin/bun"));
    const version = (await run("bun-version", bun, ["--version"])).trim();
    if (version !== scenario.bunVersion) throw new Error(`Expected Bun ${scenario.bunVersion}, found ${version}`);
    environment.readOnlyPaths.push(bun);
    environment.path.push(dirname(bun));
    environment.dependencies.bun = { version, executableSha256: hash(readFileSync(bun)) };
  }
  if (scenario.nodeProject) {
    const prefix = join(caseDir, "node-env");
    mkdirSync(prefix, { recursive: true });
    if (!existsSync(join(prefix, "package.json"))) writeFileSync(join(prefix, "package.json"), '{"private":true}\n');
    const locked = existsSync(join(prefix, "package-lock.json"));
    await run("node-setup", "npm", [locked ? "ci" : "install", "--prefix", prefix,
      "--ignore-scripts", "--no-audit", "--no-fund", ...(locked ? [] : ["--save-exact", ...scenario.nodePackages])]);
    const modules = join(prefix, "node_modules");
    environment.env.PI_NODE_MODULES = modules;
    const link = join(cwd, scenario.nodeProject, "node_modules");
    if (!existsSync(link)) symlinkSync(modules, link);
    if (realpathSync(link) !== modules) throw new Error("Unexpected node_modules directory");
    environment.readOnlyPaths.push(prefix);
    environment.path.push(join(modules, ".bin"));
    environment.dependencies.node = { version: process.version, packages: scenario.nodePackages,
      lockSha256: hash(readFileSync(join(prefix, "package-lock.json"))) };
    if (scenario.nodeProject === "packages/@expo/env") {
      // Run the actual @expo/env tests with their memfs convention, without bootstrapping Expo CLI.
      const config = join(prefix, "jest.config.cjs");
      const setup = join(prefix, "jest.setup.cjs");
      const tsconfig = join(prefix, "tsconfig.json");
      writeFileSync(setup, ["fs", "node:fs"].map(name =>
        `jest.mock(${JSON.stringify(name)}, () => require(${JSON.stringify(join(modules, "memfs"))}).fs);`).join("\n") + "\n");
      writeFileSync(tsconfig, JSON.stringify({ compilerOptions: {
        target: "ES2022", module: "commonjs", esModuleInterop: true, isolatedModules: true,
      } }, null, 2) + "\n");
      writeFileSync(config, "module.exports = " + JSON.stringify({
        rootDir: join(cwd, scenario.nodeProject), testEnvironment: "node", clearMocks: true,
        testMatch: ["<rootDir>/src/__tests__/*.test.ts"], setupFilesAfterEnv: [setup],
        moduleDirectories: [modules, "node_modules"],
        transform: { "^.+\\.tsx?$": [join(modules, "ts-jest"), { diagnostics: false, tsconfig }] },
      }, null, 2) + ";\n");
      environment.env.PI_JEST_CONFIG = config;
      environment.dependencies.node.testConfigSha256 = hash(readFileSync(config));
      environment.dependencies.node.testSetupSha256 = hash(readFileSync(setup));
      environment.dependencies.node.typescriptConfigSha256 = hash(readFileSync(tsconfig));
    }
  }
  return { pythonEnvironment, toolEnvironment: environment };
}
