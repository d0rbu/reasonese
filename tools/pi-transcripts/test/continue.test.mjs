import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { createServer } from "node:http";
import { lstatSync, mkdtempSync, mkdirSync, readFileSync, readlinkSync, readdirSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import test from "node:test";
import { main, preflightPlan, runContinuation, validateHistory } from "../continue.mjs";
import { runTrajectory } from "../trajectory.mjs";
import { isNeutralCase } from "../backing.mjs";
import { projectDir } from "../sandbox.mjs";
import { git, KIMI_MODEL, loadManifest, readJson, sha256, writeJson } from "../workspaces.mjs";

const metadata = { id: KIMI_MODEL, name: "Explicit offline fixture", context_length: 262144,
  top_provider: { max_completion_tokens: 8192 }, pricing: { prompt: "0.000002", completion: "0.00001", input_cache_read: "0.0000002" } };
async function server(t, reply) {
  const requests = [];
  const http = createServer(async (req, res) => {
    let body = ""; for await (const chunk of req) body += chunk;
    requests.push(JSON.parse(body)); reply(requests.length - 1, res);
  });
  await new Promise(r => http.listen(0, "127.0.0.1", r));
  t.after(() => new Promise(r => { http.closeAllConnections(); http.close(r); }));
  return { requests, fixtureEndpoint: `http://127.0.0.1:${http.address().port}/v1` };
}
function completion(res, index, call = null) {
  res.writeHead(200, { "Content-Type": "text/event-stream" });
  const chunk = (delta, finish_reason = null) => res.write(`data: ${JSON.stringify({ id: `fixture-${index}`, object: "chat.completion.chunk", created: 1,
    model: KIMI_MODEL, choices: [{ index: 0, delta, finish_reason }] })}\n\n`);
  chunk({ role: "assistant", reasoning: "Explicit offline fixture reasoning.", reasoning_details: [{ type: "reasoning.text", text: "Explicit offline fixture reasoning.", index: 0 }] });
  chunk(call ? { tool_calls: [{ index: 0, id: `call-${index}`, type: "function", function: { name: call[0], arguments: JSON.stringify(call[1]) } }] } : { content: "Done with the requested turn." });
  chunk({}, call ? "tool_calls" : "stop"); res.end("data: [DONE]\n\n");
}
async function fixture(t) {
  mkdirSync(join(projectDir, "out"), { recursive: true });
  const root = mkdtempSync(join(projectDir, "out/pi-continuation-fixture-"));
  t.after(() => {
    for (const entry of readdirSync(root)) {
      const path = join(root, entry);
      if (lstatSync(path).isSymbolicLink()) {
        const target = readlinkSync(path);
        if (isNeutralCase(target)) rmSync(dirname(target), { recursive: true, force: true });
      }
    }
    rmSync(root, { recursive: true, force: true });
  });
  const caseDir = join(root, "source"), cwd = join(caseDir, "workspace"); mkdirSync(cwd, { recursive: true });
  writeFileSync(join(cwd, "AGENTS.md"), "Explicit offline fixture context.\n");
  writeFileSync(join(cwd, "README.md"), "Explicit synthetic offline fixture.\n");
  for (const args of [["init"], ["config", "user.name", "Fixture"], ["config", "user.email", "fixture@example.invalid"], ["config", "commit.gpgSign", "false"], ["add", "."], ["commit", "-m", "fixture"]]) await git(cwd, ...args);
  const venv = join(caseDir, "python-env");
  execFileSync("uv", ["venv", "--python", join(projectDir, ".venv/bin/python"), venv], { stdio: "pipe" });
  const python = realpathSync(join(venv, "bin/python")), runtime = dirname(dirname(python));
  const library = readdirSync(join(runtime, "lib")).find(name => /^libpython.*\.so\.[0-9]/.test(name));
  const prepared = { pythonEnvironment: { path: venv, runtime: { path: runtime,
    executableSha256: sha256(readFileSync(python)), sharedLibraries: library ? [{ file: library, sha256: sha256(readFileSync(join(runtime, "lib", library))) }] : [] } },
    toolEnvironment: { mounts: [], path: [], env: {}, dependencies: {} } };
  writeJson(join(caseDir, "prepared.json"), prepared);
  const scenario = { id: "offline-fixture", repository: "https://github.com/example/fixture.git", revision: (await git(cwd, "rev-parse", "HEAD")).trim(), ruleFile: "AGENTS.md", baseInstruction: "Print 43.", prompt: "Investigate README.md.", maxUserTurns: 2 };
  const manifest = { ...loadManifest(join(projectDir, "configs/pi_transcripts.json")), model: KIMI_MODEL, scenarios: [scenario] };
  const sourceServer = await server(t, (i, res) => completion(res, i, i === 0 ? ["read", { path: "README.md" }] : i === 2 ? ["write", { path: "old-implementation.txt", content: "STALE RECORDED IMPLEMENTATION\n" }] : null));
  const outcome = await runTrajectory({ caseDir, scenario, manifest, metadata, apiKey: "fixture", ...sourceServer,
    nextTurn: async () => ({ prompt: "Original implementation request.", rationale: "Offline fixture operator turn" }) });
  assert.equal(outcome.status, "completed", JSON.stringify(outcome));
  const dir = join(caseDir, "trajectory"), messages = readJson(join(dir, "messages.json"));
  const cut = messages.findIndex((m, i) => i > 0 && m.role === "user"), prefix = messages.slice(0, cut);
  const ref = path => ({ path, sha256: sha256(readFileSync(join(dir, path))) });
  const snapshot = readJson(join(dir, "turns/001.workspace.json"));
  const source = { kind: "pi_source_template", captureKind: "offline_test_fixture", scenario: scenario.id, sourceCaseDir: caseDir, prepared, scenarioConfig: scenario, manifest,
    messages: ref("messages.json"), system: ref("system.txt"), tools: ref("tools.json"),
    checkpoints: { "1": { workspace: ref("turns/001.workspace.tar.gz"), prefixSha256: sha256(JSON.stringify(prefix)), gitState: snapshot.gitState } } };
  writeJson(join(dir, "template.json"), source);
  const sourceDescriptor = ref("template.json");
  const spec = { instruction: "Print 43.", author: "Kimi K3.1 Pro", framing: "normal", channel: "user message" };
  const inputs = [{ spec, content: "First authored policy assignment." }, { spec, content: "Second authored policy assignment." }];
  const bank = join(root, "variants"); mkdirSync(bank);
  const file = (name, data) => { const path = join(bank, name); writeFileSync(path, data); return { path, sha256: sha256(readFileSync(path)) }; };
  const authoringSource = file("authoring.txt", "Print 43.\n");
  const sourceProvenance = file("provenance.json", JSON.stringify({ authoringSourceSha256: sha256("Print 43.") }));
  const sourceBinding = file("binding.json", JSON.stringify({ kind: "reviewed_native_source_binding", reviewNote: "Explicit offline fixture binding",
    target: { descriptorSha256: sourceDescriptor.sha256, messagesSha256: source.messages.sha256, checkpoints: { "1": source.checkpoints["1"].workspace.sha256 } },
    slots: Object.fromEntries(["a", "b"].map(side => ["implementation/" + side, { authoringSource: authoringSource.sha256, sourceProvenance: sourceProvenance.sha256, instructionSource: authoringSource.sha256 }])) }));
  const variants = inputs.map((item, i) => ({ ...file(`variant-${i}.txt`, item.content), spec, slot: "implementation", side: i ? "b" : "a", authoringSource, sourceProvenance, instructionSource: authoringSource, sourceBinding }));
  const plan = { kind: "pi_experiment_plan", mode: "conflicting_user_turns", model: KIMI_MODEL, thinking: "high", source, sourceDescriptor, corpusRoot: dir,
    prefix, system: readFileSync(join(dir, "system.txt"), "utf8"), tools: readJson(join(dir, "tools.json")),
    checkpoint: { ...source.checkpoints["1"], workspace: { ...source.checkpoints["1"].workspace, path: join(dir, "turns/001.workspace.tar.gz") } },
    containsPlaceholders: false, turns: inputs.map((item, i) => ({ prompt: item.content, variant: variants[i] })), metadata: { authoringInputs: inputs, authoringVariants: variants, historicalPlacements: [] } };
  const planPath = join(root, "plan.json"), receiptPath = join(root, "qa.json");
  function save() {
    writeJson(planPath, plan);
    writeJson(receiptPath, { kind: "pi_message_qa_receipt", passed: true, judge: "openai/gpt-5.6-luna:batch", planSha256: sha256(readFileSync(planPath)),
      inputs: plan.metadata.authoringInputs.map(i => ({ spec: i.spec, contentSha256: sha256(i.content), complies: true, issues: [] })), fixture: true });
  }
  save(); return { root, plan, planPath, receiptPath, save };
}

test("native continuation restores exact investigation state and executes both authored turns", { timeout: 240_000 }, async t => {
  const f = await fixture(t);
  const calls = [["write", { path: "answer.py", content: "print(42)\n" }], ["edit", { path: "answer.py", oldText: "42", newText: "43" }], ["bash", { command: "python answer.py" }]];
  const http = await server(t, (i, res) => completion(res, 100 + i, calls[i % 4]));
  const output = join(f.root, "continued");
  const result = await runContinuation({ ...f, output, metadata, apiKey: "fixture", ...http });
  assert.equal(result.status, "completed", JSON.stringify(result));
  assert.equal(result.userTurns, 2); assert.equal(result.modelRequests, 8);
  assert.equal(readFileSync(join(output, "workspace/answer.py"), "utf8"), "print(43)\n");
  const messages = readJson(join(output, "trajectory/messages.json"));
  assert.deepEqual(messages.slice(0, f.plan.prefix.length), f.plan.prefix);
  assert.ok(!JSON.stringify(messages).includes("STALE RECORDED IMPLEMENTATION"));
  assert.equal(http.requests[0].messages[0].content, f.plan.system);
  assert.ok(http.requests[0].messages.some(m => m.role === "assistant" && JSON.stringify(m).includes("Explicit offline fixture reasoning")));
  assert.ok(http.requests.at(-1).messages.some(m => m.role === "tool" && m.content.includes("43")));
  const persisted = readFileSync(result.sessionFile, "utf8").trim().split("\n").map(JSON.parse).filter(e => e.type === "message").map(e => e.message);
  assert.deepEqual(persisted, messages);
  assert.equal(readJson(join(output, "trajectory/provenance.json")).kind, "offline_test_fixture");
  assert.equal(readJson(join(output, "trajectory/provenance.json")).continuation.planSha256, sha256(readFileSync(f.planPath)));
  await assert.rejects(runContinuation({ ...f, output, metadata, apiKey: "fixture", ...http }), /fresh/);
});

test("lower momentum delivers consecutive authored messages before first new response", { timeout: 180_000 }, async t => {
  const f = await fixture(t);
  f.plan.metadata.historicalPlacements = [{ position: 1, message: f.plan.prefix.length }];
  f.plan.prefix.push({ role: "user", content: [{ type: "text", text: f.plan.turns.shift().prompt }], timestamp: 0 }); f.save();
  const http = await server(t, (i, res) => completion(res, 100 + i));
  const result = await runContinuation({ ...f, output: join(f.root, "low"), metadata, apiKey: "fixture", ...http });
  assert.equal(result.status, "completed", JSON.stringify(result)); assert.equal(result.modelRequests, 1);
  assert.deepEqual(http.requests[0].messages.filter(m => m.role === "user").slice(-2).map(m => typeof m.content === "string" ? m.content : m.content.map(b => b.text).join("")), ["First authored policy assignment.", "Second authored policy assignment."]);
});

test("preflight rejects placeholders, stale QA, altered source/state and changed delivery before provider access", { timeout: 180_000 }, async t => {
  const f = await fixture(t);
  assert.throws(() => preflightPlan(f.planPath), /live source/);
  assert.ok(preflightPlan(f.planPath, f.receiptPath, { fixture: true }));
  const original = structuredClone(f.plan);
  for (const [change, error] of [
    [p => { p.containsPlaceholders = true; }, /Placeholder/],
    [p => { p.turns[0].prompt = "TODO: human"; }, /Placeholder/],
    [p => { p.prefix[0].content[0].text = "changed"; }, /prefix differs/],
    [p => { p.system += "changed"; }, /system\/tools/],
    [p => { p.source.prepared.environment = { mounts: [{ source: "/", target: "/host" }] }; }, /pinned complete descriptor/],
    [p => { p.source.manifest.maxModelRequests = 1000000; }, /pinned complete descriptor/],
    [p => { p.source.checkpoints["1"].gitState = {}; }, /pinned complete descriptor/],
    [p => { p.checkpoint.workspace.sha256 = "0".repeat(64); }, /checkpoint differs/],
    [p => { p.turns[0].prompt = "unjudged text"; }, /QA text differs/],
    [p => { p.turns[0].prompt = p.metadata.authoringInputs[0].content = "different authored text"; }, /authored variant file/],
    [p => { p.turns[0].variant = null; }, /needs variant provenance/],
  ]) {
    Object.assign(f.plan, structuredClone(original)); change(f.plan); f.save();
    assert.throws(() => preflightPlan(f.planPath, f.receiptPath, { fixture: true }), error);
  }
  Object.assign(f.plan, original); f.save();
  writeFileSync(f.receiptPath, "{}"); assert.throws(() => preflightPlan(f.planPath, f.receiptPath, { fixture: true }), /message QA/);
  f.save(); writeFileSync(f.plan.checkpoint.workspace.path, "changed archive");
  assert.throws(() => preflightPlan(f.planPath, null, { fixture: true }), /archive hash/);
  await assert.rejects(main(["--plan", f.planPath]), /allow-paid/);
  assert.throws(() => validateHistory([{ role: "toolResult", toolCallId: "x", toolName: "read" }]), /Orphan/);
  assert.throws(() => validateHistory([{ role: "assistant", content: [{ type: "toolCall", id: "x", name: "read" }] }]), /unresolved/);
});

// Rewriting a source descriptor requires a new explicit applicability review.
function refreshDescriptor(f, { review = true } = {}) {
  const path = resolve(f.plan.corpusRoot, f.plan.sourceDescriptor.path);
  writeJson(path, f.plan.source);
  f.plan.sourceDescriptor.sha256 = sha256(readFileSync(path));
  if (review) {
    const variants = [...f.plan.turns.map(t => t.variant), ...f.plan.metadata.authoringVariants];
    for (const path of new Set(variants.map(v => v.sourceBinding.path))) {
      const binding = readJson(path); binding.target.descriptorSha256 = f.plan.sourceDescriptor.sha256;
      writeJson(path, binding);
      const hash = sha256(readFileSync(path));
      variants.filter(v => v.sourceBinding.path === path).forEach(v => { v.sourceBinding.sha256 = hash; });
    }
  }
  f.save();
}

test("review binding rejects coherently repinned system or restoration metadata", { timeout: 180_000 }, async t => {
  const f = await fixture(t), original = structuredClone(f.plan);
  for (const mutate of [
    p => { p.source.prepared.toolEnvironment.mounts.push({ source: "/", target: "/host" }); },
    p => {
      const path = join(p.corpusRoot, "changed-system.txt"); writeFileSync(path, "Changed system context");
      p.source.system = { path, sha256: sha256(readFileSync(path)) }; p.system = "Changed system context";
    },
  ]) {
    Object.assign(f.plan, structuredClone(original)); mutate(f.plan);
    refreshDescriptor(f, { review: false });
    assert.throws(() => preflightPlan(f.planPath, f.receiptPath, { fixture: true }), /reviewed binding/);
  }
});

test("SDK tool incompatibility is rejected by the pre-QA preflight", { timeout: 180_000 }, async t => {
  const f = await fixture(t);
  f.plan.tools[0].description += " Different SDK schema.";
  const path = join(f.plan.corpusRoot, "changed-tools.json"); writeJson(path, f.plan.tools);
  f.plan.source.tools = { path, sha256: sha256(readFileSync(path)) };
  refreshDescriptor(f);
  assert.throws(() => preflightPlan(f.planPath, null, { fixture: true }), /incompatible with current Pi SDK/);
});

test("restored Git, context and immutable runtimes are checked before model lookup", { timeout: 240_000 }, async t => {
  const f = await fixture(t), original = structuredClone(f.plan);
  const runtime = join(f.root, "external-runtime"); writeFileSync(runtime, "external binary");
  const originalFetch = globalThis.fetch; let lookups = 0;
  globalThis.fetch = async () => { lookups++; throw new Error("Unexpected provider lookup before local guards"); };
  try {
    for (const [name, mutate, error] of [
      ["git", p => { p.source.checkpoints["1"].gitState.status = "wrong restored state"; }, /Restored Git state differs/],
      ["context", p => { p.source.scenarioConfig.ruleFile = "missing-context.md"; }, /ENOENT/],
      ["python", p => { p.source.prepared.pythonEnvironment.runtime.executableSha256 = "0".repeat(64); }, /Python runtime changed/],
      ["library", p => { p.source.prepared.pythonEnvironment.runtime.sharedLibraries[0].sha256 = "0".repeat(64); }, /Python shared library changed/],
      ["uv", p => {
        p.source.prepared.toolEnvironment.mounts = [{ source: runtime, target: "/opt/bin/uv" }];
        p.source.prepared.toolEnvironment.dependencies = { uv: { executableSha256: "0".repeat(64) } };
      }, /Prepared uv runtime changed/],
    ]) {
      Object.assign(f.plan, structuredClone(original)); mutate(f.plan); refreshDescriptor(f);
      await assert.rejects(runContinuation({ planPath: f.planPath, receiptPath: f.receiptPath,
        output: join(f.root, "rejected-" + name), apiKey: "fixture", fixtureEndpoint: "http://127.0.0.1:1/v1" }), error);
      assert.equal(lookups, 0, name);
      assert.deepEqual(readJson(join(f.root, "rejected-" + name, "plan.json")), f.plan);
      assert.equal(readJson(join(f.root, "rejected-" + name, "message-qa.json")).passed, true);
    }
  } finally { globalThis.fetch = originalFetch; }
});
