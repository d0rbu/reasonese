import assert from "node:assert/strict";
import { test } from "node:test";
import { audit, neutralize, pathReplacements } from "../neutralize.mjs";

test("path derivatives preserve non-path observations and consistent references", () => {
  const root = "/home/d0rb/Documents/Github/reasonese/.codex/worktrees/pi-scenario-transcripts";
  const cwd = `${root}/out/pi-base-remaining-20261028/expo-dotenv-precedence/workspace`;
  const system = `Current working directory: ${cwd}`;
  const replacements = pathReplacements(system, "https://github.com/expo/expo.git");
  const input = { system, messages: [{ role: "assistant", content: [
    { type: "thinking", thinking: `Perhaps hidden evaluator expects ${cwd}/package.json` },
    { type: "toolCall", arguments: { command: `cd '${cwd}' && pwd` } },
  ] }, { role: "toolResult", content: [
    { type: "text", text: `${cwd}/package.json: read-only file system` },
  ] }], toolPath: `${root}/tools/pi-transcripts/node_modules/pi`,
  venv: `${cwd.slice(0, -10)}/python-env/bin/python`, sibling: `${cwd}2`, count: 3 };
  const edits = [];
  const result = neutralize(input, replacements, edits);
  assert.equal(result.system, "Current working directory: /home/developer/projects/expo");
  assert.equal(result.toolPath, "/opt/pi/node_modules/pi");
  assert.equal(result.venv, "/home/developer/.venv/bin/python");
  assert(result.sibling.endsWith("/workspace2"));
  assert.equal(result.messages[1].content[0].text, "/home/developer/projects/expo/package.json: read-only file system");
  assert(result.messages[0].content[0].thinking.includes("hidden evaluator"));
  assert.equal(result.count, input.count);
  assert.equal(neutralize(cwd.replace("/pi-scenario-transcripts/", "/pi-scen...ipts/") + "/mlflow/utils.py", replacements), "/home/developer/projects/expo/mlflow/utils.py");
  assert.equal(input.system, system);
  assert(edits.some(e => e.pointer.endsWith("/thinking")));
  assert.deepEqual(neutralize(result, replacements), result);
  assert(audit(result).some(h => h.category === "evaluatorAwareness"));
  assert(!audit(result).some(h => h.category === "collectionPath"));
});

test("audit searches schemas, nested arguments, details, and returned reasoning", () => {
  const hits = audit({ tools: [{ description: "sandbox" }], messages: [
    { content: [{ thinking: "coding agent benchmarks", arguments: { path: "/x/reasonese/out/a" } }], details: { error: "GOPROXY=off" } },
  ] });
  assert.deepEqual(new Set(hits.map(h => h.category)), new Set(["contextual", "evaluatorAwareness", "collectionPath", "environment"]));
  assert.throws(() => pathReplacements("", "https://github.com/expo/expo.git"));
});
