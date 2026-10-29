// Real tools run in neutral filesystem paths and a private, routed network.
import { spawn, spawnSync } from "node:child_process";
import { cpSync, existsSync, lstatSync, mkdirSync, readFileSync, readdirSync, realpathSync, symlinkSync, writeFileSync } from "node:fs";
import { neutralRuntime } from "./backing.mjs";
import { networkInterfaces } from "node:os";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  createReadToolDefinition, createBashToolDefinition,
  createEditToolDefinition, createWriteToolDefinition,
} from "@mariozechner/pi-coding-agent";

export const toolDir = dirname(fileURLToPath(import.meta.url));
export const projectDir = resolve(toolDir, "../..");
export const neutralCwd = () => "/home/developer/project";
const definitions = [createReadToolDefinition, createBashToolDefinition,
  createEditToolDefinition, createWriteToolDefinition];
const executable = name => {
  const result = spawnSync("/usr/bin/which", [name], { encoding: "utf8" });
  if (result.status !== 0) throw new Error(`Install required development tool: ${name}`);
  return realpathSync(result.stdout.trim());
};

export function environmentPaths(cwd, venv = join(projectDir, ".venv"), environment = {}) {
  const python = dirname(dirname(realpathSync(join(venv, "bin/python"))));
  const runtime = neutralRuntime();
  const mounts = [
    { source: cwd, target: neutralCwd(), writable: true },
    { source: venv, target: "/home/developer/.venv", writable: venv !== join(projectDir, ".venv") },
    { source: python, target: "/opt/python" },
    { source: join(runtime.root, "node_modules"), target: "/opt/pi/node_modules" },
    { source: join(runtime.root, "tool-worker.mjs"), target: "/opt/pi/tool-worker.mjs" },
    ...(environment.mounts ?? []),
  ];
  const replacements = [...mounts.map(m => [m.source, m.target]), ...(environment.pathAliases ?? []).map(m => [m.source, m.target])].sort((a, b) => b[0].length - a[0].length);
  const translate = value => replacements.reduce((text, [source, target]) => text.split(source).join(target), value);
  return { mounts, translate, python };
}

function shadowPython(venv, scratch, translate) {
  // Task-owned metadata may contain symlinks written by the assistant. Never
  // follow those into the controller filesystem while projecting runtime paths.
  const physicalRoot = realpathSync(venv);
  const inside = path => { try { return realpathSync(path).startsWith(physicalRoot + "/"); } catch { return false; } };
  if (!inside(join(venv, "bin"))) throw new Error("Python bin directory leaves task environment");
  const shadow = join(scratch, "runtime", "python", venv === join(projectDir, ".venv") ? "controller" : "task");
  const bin = join(shadow, "bin");
  mkdirSync(bin, { recursive: true });
  for (const entry of readdirSync(join(venv, "bin"), { withFileTypes: true })) {
    const source = join(venv, "bin", entry.name), target = join(bin, entry.name);
    if (existsSync(target) || lstatSyncSafe(target)) continue;
    if (entry.isSymbolicLink()) symlinkSync(translate(realpathSync(source)), target);
    else {
      cpSync(source, target);
      const content = readFileSync(source);
      if (!content.includes(0)) writeFileSync(target, translate(content.toString()));
    }
  }
  const cfg = join(shadow, "pyvenv.cfg");
  if (!existsSync(cfg) && !inside(join(venv, "pyvenv.cfg"))) throw new Error("Python configuration leaves task environment");
  if (!existsSync(cfg)) writeFileSync(cfg, translate(readFileSync(join(venv, "pyvenv.cfg"), "utf8")).replace(/^prompt = .*$/m, "prompt = project"));
  const overrides = [];
  // Editable import paths and direct-url records are environment metadata, not
  // repository source. Compile bytecode at the visible path instead of reusing
  // host-built .pyc files (which retain their original co_filename).
  for (const library of inside(join(venv, "lib")) ? readdirSync(join(venv, "lib")) : []) {
    const packages = join(venv, "lib", library, "site-packages");
    if (!inside(packages)) continue;
    for (const name of readdirSync(packages)) {
      const files = name.endsWith(".dist-info") ? [join(packages, name, "direct_url.json")] :
        name.endsWith(".pth") || name.startsWith("__editable__") && name.endsWith(".py") ? [join(packages, name)] : [];
      for (const source of files) {
        if (!inside(source)) continue;
        const content = readFileSync(source, "utf8"), translated = translate(content);
        if (content === translated) continue;
        const target = join(shadow, "metadata", library, name, basename(source));
        mkdirSync(dirname(target), { recursive: true });
        if (!existsSync(target)) writeFileSync(target, translated);
        overrides.push([target, translate(source)]);
      }
    }
  }
  return { bin, cfg, overrides };
}
const lstatSyncSafe = path => { try { return lstatSync(path); } catch { return null; } };

export function sandboxArgs(cwd, scratch, venv = join(projectDir, ".venv"), environment = {}) {
  const { mounts, translate } = environmentPaths(cwd, venv, environment);
  for (const name of ["tmp", "home", "runtime"]) mkdirSync(join(scratch, name), { recursive: true });
  const runtime = join(scratch, "runtime"), bin = join(runtime, "tools");
  mkdirSync(bin, { recursive: true });
  const tools = { node: realpathSync(process.execPath), rg: join(neutralRuntime().root, "rg"), uv: executable("uv") };
  for (const mount of mounts.filter(m => m.target.startsWith("/opt/bin/"))) tools[basename(mount.target)] = mount.source;
  const npm = dirname(dirname(executable("npm")));
  const args = ["--unshare-all", "--die-with-parent", "--new-session", "--hostname", "workstation",
    "--uid", String(process.getuid()), "--gid", String(process.getgid()),
    "--proc", "/proc", "--dev", "/dev", "--clearenv",
    "--bind", join(scratch, "tmp"), "/tmp", "--bind", join(scratch, "home"), "/home/developer"];
  for (const directory of ["/usr", "/bin", "/lib", "/lib64", "/etc/alternatives", "/etc/ssl"]) {
    if (existsSync(directory)) args.push("--ro-bind", directory, directory);
  }
  for (const filename of ["/etc/localtime", "/etc/hosts", "/etc/nsswitch.conf"]) {
    if (existsSync(filename)) args.push("--ro-bind", filename, filename);
  }
  const system = {
    "resolv.conf": "nameserver 10.0.2.3\n",
    passwd: `root:x:0:0:root:/root:/bin/bash\ndeveloper:x:${process.getuid()}:${process.getgid()}:Developer:/home/developer:/bin/bash\n`,
    group: `root:x:0:\ndeveloper:x:${process.getgid()}:\n`,
  };
  for (const [name, content] of Object.entries(system)) {
    const file = join(runtime, name); writeFileSync(file, content); args.push("--ro-bind", file, `/etc/${name}`);
  }
  for (const mount of mounts.filter(m => !m.target.startsWith("/opt/bin/"))) args.push(mount.writable ? "--bind" : "--ro-bind", mount.source, mount.target);
  const npmLock = join(toolDir, "node_modules/.package-lock.json");
  if (existsSync(npmLock)) {
    const neutralLock = join(runtime, "sdk-package-lock.json");
    writeFileSync(neutralLock, readFileSync(npmLock, "utf8").replaceAll("reasonese-pi-transcripts", "pi"));
    args.push("--ro-bind", neutralLock, "/opt/pi/node_modules/.package-lock.json");
  }
  const shadow = shadowPython(venv, scratch, translate);
  args.push("--bind", shadow.bin, "/home/developer/.venv/bin", "--bind", shadow.cfg, "/home/developer/.venv/pyvenv.cfg");
  for (const [source, target] of shadow.overrides) args.push("--bind", source, target);
  args.push("--ro-bind", bin, "/opt/bin");
  for (const [name, source] of Object.entries(tools)) {
    if (!existsSync(join(bin, name))) writeFileSync(join(bin, name), "");
    args.push("--ro-bind", source, `/opt/bin/${name}`);
  }
  args.push("--ro-bind", npm, "/opt/node/npm");
  for (const name of ["npm", "npx"]) {
    const file = join(bin, name);
    if (!existsSync(file)) writeFileSync(file, `#!/bin/sh\nexec /opt/bin/node /opt/node/npm/bin/${name === "npm" ? "npm-cli" : "npx-cli"}.js "$@"\n`, { mode: 0o755 });
  }
  for (const [key, value] of Object.entries(environment.env ?? {})) {
    if (key.startsWith("PI_")) continue;
    args.push("--setenv", key, translate(value));
  }
  if (environment.preparation) {
    args.push("--ro-bind", toolDir, "/opt/validation");
    for (const [key, value] of Object.entries(environment.baselineEnv ?? {})) args.push("--setenv", key,
      translate(value).replace(toolDir, "/opt/validation"));
  }
  return [...args, "--remount-ro", "/",
    "--setenv", "HOME", "/home/developer", "--setenv", "TMPDIR", "/tmp",
    "--setenv", "PATH", ["/home/developer/.venv/bin", "/opt/bin", ...(environment.path ?? []).map(translate), "/usr/bin", "/bin"].join(":"),
    "--setenv", "VIRTUAL_ENV", "/home/developer/.venv",
    "--setenv", "UV_LINK_MODE", "copy",
    "--setenv", "PYTHONPYCACHEPREFIX", "/home/developer/.cache/pycache",
    "--setenv", "LANG", "C.UTF-8", "--setenv", "TZ", "UTC",
    "--setenv", "GIT_CONFIG_NOSYSTEM", "1", "--setenv", "GIT_CONFIG_GLOBAL", "/dev/null",
    "--chdir", neutralCwd(), "/opt/bin/node", "/opt/pi/tool-worker.mjs"];
}

export function sandboxTools(cwd, scratch, timeoutSeconds, venv, environment) {
  return definitions.map(factory => {
    const definition = factory(neutralCwd());
    return { ...definition, execute: (id, parameters, signal) =>
      runInSandbox(cwd, scratch, { name: definition.name, id, parameters }, timeoutSeconds, signal, venv, environment) };
  });
}
export function sandboxGit(cwd, scratch, ...args) {
  return runInSandbox(cwd, scratch, { git: args }, 120);
}
function runInSandbox(cwd, scratch, request, timeoutSeconds, signal, venv, environment) {
  return new Promise((accept, reject) => {
    const args = sandboxArgs(cwd, scratch, venv, environment);
    const child = spawn("unshare", ["--user", "--map-root-user", "--net", process.execPath, join(toolDir, "network-worker.mjs")], {
      stdio: ["pipe", "pipe", "pipe", "pipe"], env: { PATH: "/usr/bin:/bin" },
    });
    let router, stdout = "", stderr = "", failure;
    const stop = reason => { if (failure) return; failure = reason; child.kill("SIGKILL"); router?.kill("SIGKILL"); };
    const abort = () => stop("Tool aborted");
    const timer = setTimeout(() => stop("Tool time limit reached"), timeoutSeconds * 1000);
    signal?.addEventListener("abort", abort, { once: true });
    if (signal?.aborted) abort();
    child.stdio[3].once("data", () => {
      router = spawn("slirp4netns", ["--configure", "--disable-host-loopback", "--enable-sandbox", "--enable-seccomp", "--ready-fd=3", String(child.pid), "tap0"], {
        stdio: ["ignore", "ignore", "pipe", "pipe"], env: { PATH: "/usr/bin:/bin" },
      });
      router.stderr.on("data", chunk => { stderr += chunk; });
      router.on("error", error => stop(error.message));
      router.on("exit", () => { if (child.exitCode === null) stop("Development network exited"); });
      router.stdio[3].once("data", () => {
        const blockedAddresses = Object.values(networkInterfaces()).flat().filter(a => a.family === "IPv4" && !a.internal).map(a => a.address);
        child.stdin.end(JSON.stringify({ args, request: { cwd: neutralCwd(), ...request }, blockedAddresses }));
      });
    });
    child.stdout.setEncoding("utf8"); child.stderr.setEncoding("utf8");
    child.stdout.on("data", chunk => { stdout += chunk; if (stdout.length > 48 * 1024 * 1024) stop("Tool transport output limit reached"); });
    child.stderr.on("data", chunk => { stderr += chunk; if (stderr.length > 1024 * 1024) stop("Tool stderr limit reached"); });
    child.on("error", error => { clearTimeout(timer); reject(error); });
    child.stdin.on("error", () => {});
    child.on("close", code => {
      clearTimeout(timer); signal?.removeEventListener("abort", abort); router?.kill("SIGTERM");
      if (failure || code !== 0) return reject(new Error(failure || stderr || `Tool exit ${code}`));
      try { const response = JSON.parse(stdout); if (response.error) reject(new Error(response.error)); else accept(response.result); }
      catch (error) { reject(error); }
    });
  });
}
