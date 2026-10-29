// Private physical backing keeps kernel mount metadata free of research paths.
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { closeSync, cpSync, existsSync, lstatSync, mkdirSync, mkdtempSync, openSync, readFileSync, readlinkSync, readSync, readdirSync, realpathSync, renameSync, rmSync, symlinkSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, isAbsolute, join, sep } from "node:path";
import { fileURLToPath } from "node:url";

export const backingRoot = join(homedir(), ".cache/development");
const toolDir = dirname(fileURLToPath(import.meta.url));
export const pathExists = path => { try { lstatSync(path); return true; } catch (error) { if (error.code === "ENOENT") return false; throw error; } };
export function isNeutralCase(path) {
  const relative = path.slice(backingRoot.length + 1);
  return path.startsWith(backingRoot + sep) && /^work-[^/]+\/state$/.test(relative);
}
export function createNeutralCase(alias, { create = true } = {}) {
  if (!isAbsolute(alias) || pathExists(alias)) throw new Error("Expected fresh absolute case alias");
  mkdirSync(backingRoot, { recursive: true, mode: 0o700 });
  if (realpathSync(backingRoot) !== backingRoot) throw new Error("Development backing root must not be a symlink");
  const physical = join(mkdtempSync(join(backingRoot, "work-")), "state");
  if (create) mkdirSync(physical, { mode: 0o700 });
  mkdirSync(dirname(alias), { recursive: true });
  symlinkSync(physical, alias, "dir");
  return physical;
}
export function preparedCase(alias) {
  if (!pathExists(alias)) return createNeutralCase(alias);
  const physical = realpathSync(alias);
  if (!isNeutralCase(physical)) throw new Error("Case needs fresh neutral physical backing; use a fresh output directory");
  return physical;
}

const buffer = Buffer.alloc(1024 * 1024);
function updateFile(hash, path) {
  const fd = openSync(path, "r");
  try { let bytes; while ((bytes = readSync(fd, buffer, 0, buffer.length, null))) hash.update(buffer.subarray(0, bytes)); }
  finally { closeSync(fd); }
}
function treeHash(root) {
  const hash = createHash("sha256");
  const visit = (directory, prefix = "") => {
    for (const entry of readdirSync(directory, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
      const path = join(directory, entry.name), name = join(prefix, entry.name);
      hash.update(JSON.stringify([name, entry.isDirectory() ? "directory" : entry.isSymbolicLink() ? "symlink" : "file"]));
      if (entry.isDirectory()) visit(path, name);
      else if (entry.isSymbolicLink()) {
        const target = readlinkSync(path);
        if (isAbsolute(target)) throw new Error("SDK must use relative package symlinks; reinstall with npm ci");
        hash.update(target);
      } else if (entry.isFile()) updateFile(hash, path);
      else throw new Error("Unsupported SDK entry");
    }
  };
  visit(root); return hash.digest("hex");
}
let cached;
export function neutralRuntime() {
  if (cached) return cached;
  const sdk = realpathSync(join(toolDir, "node_modules")), worker = join(toolDir, "tool-worker.mjs");
  const rg = realpathSync(execFileSync("/usr/bin/which", ["rg"], { encoding: "utf8" }).trim());
  const sdkSha256 = treeHash(sdk), workerSha256 = createHash("sha256").update(readFileSync(worker)).digest("hex");
  const hash = createHash("sha256"); updateFile(hash, rg); const rgSha256 = hash.digest("hex");
  const identity = { sdkSha256, workerSha256, rgSha256 };
  const key = createHash("sha256").update(JSON.stringify(identity)).digest("hex");
  const parent = join(backingRoot, "runtime"), root = join(parent, key);
  mkdirSync(parent, { recursive: true, mode: 0o700 });
  if (!existsSync(root)) {
    const staging = mkdtempSync(join(parent, "stage-"));
    try {
      cpSync(sdk, join(staging, "node_modules"), { recursive: true, dereference: false, verbatimSymlinks: true });
      cpSync(worker, join(staging, "tool-worker.mjs")); cpSync(rg, join(staging, "rg"));
      try { renameSync(staging, root); }
      catch (error) { if (!["EEXIST", "ENOTEMPTY"].includes(error.code) || !existsSync(root)) throw error; }
    } finally { rmSync(staging, { recursive: true, force: true }); }
  }
  cached = { root, ...identity };
  return cached;
}
