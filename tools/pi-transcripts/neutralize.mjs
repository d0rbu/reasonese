// Offline, path-only derivatives. Never overwrite collection evidence.
import { createHash } from "node:crypto";

export const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const escape = (text) => text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

export function pathReplacements(system, repository) {
  const cwd = system.match(/^Current working directory: (.+)$/m)?.[1];
  if (!cwd || !cwd.endsWith("/workspace")) throw new Error("Expected recorded workspace cwd");
  const caseRoot = cwd.slice(0, -"/workspace".length);
  const projectRoot = caseRoot.split("/out/")[0];
  if (projectRoot === caseRoot) throw new Error("Expected recorded collection root");
  const repo = repository.split("/").at(-1).replace(/\.git$/, "");
  if (!/^[\w.-]+$/.test(repo)) throw new Error("Invalid repository name");
  return [
    // pytest abbreviates module repr paths independently of its traceback paths.
    [cwd.replace("/pi-scenario-transcripts/", "/pi-scen...ipts/"), `/home/developer/projects/${repo}`],
    [cwd, `/home/developer/projects/${repo}`],
    [`${caseRoot}/python-env`, "/home/developer/.venv"],
    [`${caseRoot}/node-env`, "/home/developer/.local/share/node"],
    [`${caseRoot}/go-modules`, "/home/developer/go/pkg/mod"],
    [caseRoot, "/home/developer"],
    [`${projectRoot}/tools/pi-transcripts`, "/opt/pi"],
    [`${projectRoot}/.venv`, "/opt/python"],
    [projectRoot, "/opt/devtools"],
    ["/home/d0rb", "/home/developer"],
    ["/home/pi", "/home/developer"],
  ].sort((a, b) => b[0].length - a[0].length);
}

export function neutralize(value, replacements, edits = [], pointer = "") {
  if (typeof value === "string") {
    const lookup = new Map(replacements);
    const pattern = new RegExp(`(?:${replacements.map(([from]) => escape(from)).join("|")})(?![\\w.-])`, "g");
    return value.replace(pattern, (from, offset) => {
      const to = lookup.get(from);
      edits.push({ pointer, offset, from, to });
      return to;
    });
  }
  if (Array.isArray(value)) return value.map((item, i) => neutralize(item, replacements, edits, `${pointer}/${i}`));
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key,
      neutralize(item, replacements, edits, `${pointer}/${key.replace(/~/g, "~0").replace(/\//g, "~1")}`)]));
  }
  return value;
}

const categories = {
  collectionPath: /reasonese\/|\.codex\/|pi-(?:base|transcript|scenario)[\w/-]*|tool-state|baseline_probe/gi,
  evaluatorAwareness: /\bhidden (?:evaluator|tests?)\b|\bbenchmark task\b|\bcoding agent benchmarks\b|\btask design expects\b/gi,
  environment: /command not found|could not resolve host|read-only file system|module lookup disabled|GOPROXY=off|UV_OFFLINE|isolated test venv|missing script|cannot find module|no module named|docker\.sock/gi,
  contextual: /\b(?:sandbox\w*|bubblewrap|bwrap|harness|pilot|evaluat\w*|benchmark\w*|fixtures?|mocks?|memfs|isolat\w*)\b/gi,
};

// Search every string, including tool args/details, reasoning and error messages.
export function audit(value, pointer = "", hits = []) {
  if (typeof value === "string") {
    for (const [category, pattern] of Object.entries(categories)) {
      for (const match of value.matchAll(pattern)) hits.push({ pointer, category,
        offset: match.index, match: match[0],
        context: value.slice(Math.max(0, match.index - 100), match.index + match[0].length + 180) });
    }
  } else if (Array.isArray(value)) value.forEach((item, i) => audit(item, `${pointer}/${i}`, hits));
  else if (value && typeof value === "object") for (const [key, item] of Object.entries(value)) audit(item, `${pointer}/${key}`, hits);
  return hits;
}
