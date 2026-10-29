// Complete task state is recorded separately from the model-visible history.
import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { createReadStream, existsSync, realpathSync, statSync } from "node:fs";
import { basename, dirname, isAbsolute, join } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { createNeutralCase, pathExists } from "./backing.mjs";
const exec = promisify(execFile);

export async function snapshotWorkspace(caseDir, prefix, gitState = null) {
  const roots = ["workspace", "python-env", "node-env", "go-modules", "tool-state"].filter(name => existsSync(`${caseDir}/${name}`));
  const archive = `${prefix}.workspace.tar.gz`;
  // GNU tar does not follow symlinks by default. Include independent .git state,
  // ignored/untracked files and mutable dependencies, not just a tracked diff.
  await exec("tar", ["--format=pax", "-czf", archive, "-C", caseDir, ...roots], {
    timeout: 600_000, maxBuffer: 1024 * 1024,
    env: { PATH: "/usr/bin:/bin", LANG: "C.UTF-8" },
  });
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(archive)) hash.update(chunk);
  return { archive: basename(archive), sha256: hash.digest("hex"), bytes: statSync(archive).size,
    roots, gitState, format: "pax tar.gz; symlinks retained without dereferencing", includesGitState: true,
    mutableDependenciesIncluded: true,
    restore: "Extract into a fresh private case directory only. Reject traversal, special files and writes through symlink parents. Retain absolute neutral-path symlinks without dereferencing them. Shared read-only runtime identity is recorded separately in provenance." };
}

export async function restoreCheckpoint(archive, destination, { sha256: expected, sourceCaseDir, prepared } = {}) {
  if (typeof expected !== "string" || !/^[a-f0-9]{64}$/.test(expected)) throw new Error("Expected checkpoint SHA-256");
  if (!isAbsolute(destination)) throw new Error("Expected absolute private restore directory");
  if (prepared && (typeof sourceCaseDir !== "string" || !isAbsolute(sourceCaseDir))) throw new Error("Expected original source case directory");
  if (pathExists(destination)) throw new Error("Restore needs a fresh private case directory");
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(archive)) hash.update(chunk);
  if (hash.digest("hex") !== expected) throw new Error("Checkpoint hash mismatch");
  const alias = destination;
  destination = createNeutralCase(alias, { create: false });
  const toolRoot = dirname(fileURLToPath(import.meta.url));
  await exec(join(toolRoot, "../../.venv/bin/python"), [join(toolRoot, "restore-checkpoint.py"), archive, destination], {
    timeout: 600_000, maxBuffer: 1024 * 1024, env: { PATH: "/usr/bin:/bin", LANG: "C.UTF-8" },
  });
  let relocated;
  if (prepared) {
    relocated = structuredClone(prepared);
    const sourceRoots = [...new Set([sourceCaseDir,
      pathExists(sourceCaseDir) ? realpathSync(sourceCaseDir) : sourceCaseDir,
      relocated.pythonEnvironment?.path ? dirname(relocated.pythonEnvironment.path) : sourceCaseDir,
    ])].sort((a, b) => b.length - a.length);
    const move = value => {
      const source = typeof value === "string" && sourceRoots.find(root => value === root || value.startsWith(root + "/"));
      return source ? destination + value.slice(source.length) : value;
    };
    if (relocated.pythonEnvironment) relocated.pythonEnvironment.path = move(relocated.pythonEnvironment.path);
    if (relocated.context) relocated.context.path = move(relocated.context.path);
    const environment = relocated.toolEnvironment;
    if (environment) {
      environment.mounts = (environment.mounts ?? []).map(m => ({ ...m, source: move(m.source) }));
      environment.path = (environment.path ?? []).map(move);
      for (const key of ["env", "baselineEnv"]) environment[key] = Object.fromEntries(Object.entries(environment[key] ?? {}).map(([name, value]) => [name, sourceRoots.reduce((text, root) => text.split(root + "/").join(destination + "/"), value)]));
      environment.pathAliases = [
        ...sourceRoots.flatMap(root => [
          { source: root + "/workspace", target: "/home/developer/project" },
          { source: root + "/python-env", target: "/home/developer/.venv" },
          { source: root + "/node-env", target: "/home/developer/.local/share/node" },
          { source: root + "/go-modules", target: "/home/developer/go/pkg/mod" },
        ]),
        ...(environment.pathAliases ?? []),
      ];
    }
  }
  return { destination, alias, workspace: join(destination, "workspace"), archive, sha256: expected, prepared: relocated };
}
