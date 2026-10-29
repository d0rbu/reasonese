import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { authorRequest, checkVariants, FRAMINGS, MODEL, scaffold, tasksFor } from "../author-variants.mjs";

function configuration() {
  const item = (prompt, canonicalInstruction = null) => ({ source: prompt, prompt,
    canonicalInstruction, sourceCheckpointSha256: "checkpoint", adaptation: [] });
  return { model: MODEL, thinking: "high", framings: Object.fromEntries(FRAMINGS.map(f => [f, `Style ${f}`])),
    maxOutputTokens: 8192, maxRequests: 108, requestTimeoutSeconds: 180, concurrency: 3,
    scenarios: [{ id: "parser", source: { sourceMessagesSha256: "source-hash" },
      investigation: item("Inspect before editing; use {literal} syntax."),
      implementation: { a: item("Keep recovery. Run tests.", "Keep recovery."), b: item("Reject incomplete data. Run tests.", "Reject incomplete data.") } }] };
}

test("complete-turn scaffold keeps source separate and exposes human TODO status", () => {
  const root = mkdtempSync(join(tmpdir(), "pi-variant-scaffold-"));
  try {
    const config = configuration();
    assert.equal(scaffold(config, root).length, 9);
    const status = checkVariants(config, root);
    assert(status.every(row => row.human.status === "placeholder" && row.kimi.status === "missing"));
    const human = join(root, "variants/parser/implementation/b/human/casual.txt");
    writeFileSync(human, "please reject incomplete data\n");
    scaffold(config, root);
    assert.equal(readFileSync(human, "utf8"), "please reject incomplete data\n");
    assert.equal(checkVariants(config, root).find(row => row.id === "parser/implementation/b/casual").human.status, "written");
    assert.equal(readFileSync(join(root, "variants/parser/implementation/b/canonical-instruction.txt"), "utf8"), "Reject incomplete data.\n");
    config.scenarios[0].implementation.b.source = "Changed source";
    assert.throws(() => scaffold(config, root), /Changed existing source/);
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test("author request uses selected full source and preserves exact Kimi/high routing", () => {
  const config = configuration(), tasks = tasksFor(config);
  const task = tasks.find(t => t.id === "parser/implementation/b/persuasive");
  const request = authorRequest(task, config);
  assert.equal(request.model, MODEL);
  assert.deepEqual(request.reasoning, { effort: "high", exclude: false });
  assert.equal(request.messages[1].content, "Reject incomplete data. Run tests.");
  assert(!JSON.stringify(request).includes("Keep recovery."));
  assert(request.messages[0].content.includes("Style persuasive"));
  assert.equal(request.max_tokens, 8192);
  assert.equal(request.provider.require_parameters, true);
});

test("invalid model, framing set, paths and workload limits fail offline", () => {
  for (const mutate of [c => { c.model = "other/model"; }, c => { c.framings.extra = "extra"; },
    c => { c.scenarios[0].id = "../escape"; }, c => { c.maxRequests = 1; },
    c => { c.maxOutputTokens = 9000; }, c => { c.concurrency = 0; },
    c => { c.scenarios[0].implementation.a.prompt = "Unrelated"; }]) {
    const config = configuration(); mutate(config); assert.throws(() => tasksFor(config));
  }
});
