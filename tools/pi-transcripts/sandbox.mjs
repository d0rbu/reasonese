// Run Pi's own tools in a disposable filesystem and network namespace.
import { spawn } from "node:child_process";
import { existsSync, mkdirSync, realpathSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  createReadToolDefinition, createBashToolDefinition,
  createEditToolDefinition, createWriteToolDefinition,
} from "@mariozechner/pi-coding-agent";

export const toolDir = dirname(fileURLToPath(import.meta.url));
export const projectDir = resolve(toolDir, "../..");
const definitions = [createReadToolDefinition, createBashToolDefinition,
  createEditToolDefinition, createWriteToolDefinition];

export function sandboxArgs(cwd, scratch) {
  const venv = join(projectDir, ".venv");
  const python = dirname(dirname(realpathSync(join(venv, "bin/python"))));
  const node = realpathSync(process.execPath);
  mkdirSync(join(scratch, "tmp"), { recursive: true });
  mkdirSync(join(scratch, "home"), { recursive: true });
  const args = ["--unshare-all", "--die-with-parent", "--new-session",
    "--proc", "/proc", "--dev", "/dev", "--clearenv",
    "--bind", join(scratch, "tmp"), "/tmp",
    "--bind", join(scratch, "home"), "/home/pi"];
  for (const directory of ["/usr", "/bin", "/lib", "/lib64", "/etc/alternatives"]) {
    if (existsSync(directory)) args.push("--ro-bind", directory, directory);
  }
  for (const filename of ["/etc/passwd", "/etc/group", "/etc/localtime"]) {
    if (existsSync(filename)) args.push("--ro-bind", filename, filename);
  }
  // Never mount the host home, credentials, source worktree, or transcript directory.
  for (const resource of [toolDir, venv, python, node]) {
    args.push("--ro-bind", resource, resource);
  }
  return [...args, "--bind", cwd, cwd,
    "--remount-ro", "/",
    "--setenv", "HOME", "/home/pi", "--setenv", "TMPDIR", "/tmp",
    "--setenv", "PATH", `${venv}/bin:${dirname(node)}:/usr/bin:/bin`,
    "--setenv", "LANG", "C.UTF-8", "--setenv", "TZ", "UTC",
    "--setenv", "GIT_CONFIG_NOSYSTEM", "1",
    "--setenv", "GIT_CONFIG_GLOBAL", "/dev/null",
    "--chdir", cwd, node, join(toolDir, "tool-worker.mjs")];
}

export function sandboxTools(cwd, scratch, timeoutSeconds) {
  return definitions.map((factory) => {
    const definition = factory(cwd);
    return { ...definition, execute: (id, parameters, signal) =>
      runInSandbox(cwd, scratch, { name: definition.name, id, parameters }, timeoutSeconds, signal) };
  });
}

export function sandboxGit(cwd, scratch, ...args) {
  // A model can edit .git/config and install filters/hooks. Even evidence collection stays isolated.
  return runInSandbox(cwd, scratch, { git: args }, 120);
}

function runInSandbox(cwd, scratch, request, timeoutSeconds, signal) {
  return new Promise((accept, reject) => {
    const child = spawn("bwrap", sandboxArgs(cwd, scratch), {
      stdio: ["pipe", "pipe", "pipe"], env: { PATH: "/usr/bin:/bin" },
    });
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    let stdout = "", stderr = "";
    let failure;
    const stop = (reason) => { failure = reason; child.kill("SIGKILL"); };
    const abort = () => stop("Tool aborted");
    const timer = setTimeout(() => stop("Tool time limit reached"), timeoutSeconds * 1000);
    signal?.addEventListener("abort", abort, { once: true });
    if (signal?.aborted) abort();
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
      if (stdout.length > 48 * 1024 * 1024) stop("Tool transport output limit reached");
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
      if (stderr.length > 1024 * 1024) stop("Tool stderr limit reached");
    });
    const finish = (error) => {
      clearTimeout(timer);
      signal?.removeEventListener("abort", abort);
      if (error) reject(error);
    };
    child.on("error", finish);
    child.stdin.on("error", () => {}); // The close event records an early worker exit.
    child.on("close", (code) => {
      finish();
      if (failure || code !== 0) return reject(new Error(failure || stderr || `Tool exit ${code}`));
      try {
        const response = JSON.parse(stdout);
        if (response.error) reject(new Error(response.error));
        else accept(response.result);
      } catch (error) { reject(error); }
    });
    child.stdin.end(JSON.stringify({ cwd, ...request }));
  });
}
