import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createServer } from "node:http";
import { mkdtempSync, mkdirSync, readFileSync, readdirSync, realpathSync, rmSync, statSync, symlinkSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import test from "node:test";
import { main } from "../collect.mjs";
import { projectDir, sandboxArgs, sandboxTools } from "../sandbox.mjs";
import { captureTransport, modelMetadata, runTrajectory, verifyCompleted } from "../trajectory.mjs";
import { collectorIdentity, contextFile, git, KIMI_MODEL, loadManifest, MODEL, prepare, readJson, writeJson } from "../workspaces.mjs";

const manifest = loadManifest(join(projectDir, "configs/pi_transcripts.json"));
mkdirSync(join(projectDir, "out"), { recursive: true });
const metadata = {
  id: MODEL, name: "Offline test model", context_length: 1_048_576,
  top_provider: { max_completion_tokens: 393216 },
  pricing: { prompt: "0.0000003", completion: "0.0000012", input_cache_read: "0.000000006" },
};

async function fixture(t) {
  const caseDir = mkdtempSync(join(projectDir, "out/pi-fixture-"));
  t.after(() => rmSync(caseDir, { recursive: true, force: true }));
  const cwd = join(caseDir, "workspace");
  mkdirSync(cwd);
  writeFileSync(join(cwd, "CLAUDE.md"), "Offline fixture: include the observed command output in your reply.\n");
  symlinkSync("CLAUDE.md", join(cwd, "AGENTS.md"));
  writeFileSync(join(cwd, "README.md"), "This is an explicitly synthetic offline integration fixture.\n");
  await git(cwd, "init");
  await git(cwd, "config", "user.name", "Offline Fixture");
  await git(cwd, "config", "user.email", "fixture@example.invalid");
  await git(cwd, "config", "commit.gpgSign", "false");
  await git(cwd, "add", ".");
  await git(cwd, "commit", "-m", "test: create fixture");
  const scenario = { id: "offline-fixture", revision: (await git(cwd, "rev-parse", "HEAD")).trim(),
    repository: "https://github.com/example/fixture.git", ruleFile: "CLAUDE.md",
    prompt: "Read README.md, write and run answer.py, change it to print 43, run it again, and commit it locally." };
  return { caseDir, cwd, scenario };
}

async function server(t, respond) {
  const requests = [];
  const http = createServer(async (req, res) => {
    let body = "";
    for await (const chunk of req) body += chunk;
    requests.push(JSON.parse(body));
    respond(requests.length - 1, res);
  });
  await new Promise((resolve) => http.listen(0, "127.0.0.1", resolve));
  t.after(() => new Promise((resolve) => { http.closeAllConnections(); http.close(resolve); }));
  return { requests, fixtureEndpoint: `http://127.0.0.1:${http.address().port}/v1` };
}

function completion(res, index, call = null, finish = "stop", { model = MODEL, reasoning = null } = {}) {
  res.writeHead(200, { "Content-Type": "text/event-stream", "x-request-id": `fixture-${index}` });
  const chunk = (delta, finish_reason, usage) => res.write(`data: ${JSON.stringify({
    id: `fixture-${index}`, object: "chat.completion.chunk", created: 1, model,
    provider: "explicitly-offline-fixture", choices: [{ index: 0, delta, finish_reason }],
    ...(usage ? { usage } : {}),
  })}\n\n`);
  if (reasoning) chunk({ role: "assistant", reasoning,
    reasoning_details: [{ type: "reasoning.text", text: reasoning, index: 0 }] }, null);
  chunk(call ? { role: "assistant", tool_calls: [{ index: 0, id: `tool-${index}`, type: "function",
    function: { name: call[0], arguments: JSON.stringify(call[1]) } }] } :
    { role: "assistant", content: "Observed output: 43. Local commit complete." }, null);
  chunk({}, call ? "tool_calls" : finish, { prompt_tokens: 200, completion_tokens: 30, total_tokens: 230 });
  res.end("data: [DONE]\n\n");
}

test("real Pi agent loop runs all four real tools and persists the complete trajectory", { timeout: 60_000 }, async (t) => {
  const setup = await fixture(t);
  const calls = [
    ["read", { path: "README.md" }],
    ["write", { path: "answer.py", content: "print(42)\n" }],
    ["bash", { command: "python answer.py" }],
    ["edit", { path: "answer.py", oldText: "42", newText: "43" }],
    ["bash", { command: "python answer.py && git add answer.py && git commit -m 'test: add answer'" }],
  ];
  const mock = await server(t, (index, res) => completion(res, index, calls[index]));
  const result = await runTrajectory({ ...setup, ...mock, manifest, metadata, apiKey: "fixture-secret" });
  assert.equal(result.status, "completed", JSON.stringify(result));
  assert.equal(result.toolResults, 5);
  assert.equal(result.modelRequests, 6);
  assert.notEqual(result.finalGit.head, setup.scenario.revision);
  assert.equal(result.finalGit.status, "");
  assert.equal(readFileSync(join(setup.cwd, "answer.py"), "utf8"), "print(43)\n");
  const evidence = join(setup.caseDir, "trajectory");
  assert.ok(result.artifactHashes["http/001.response.txt"]);
  assert.throws(() => verifyCompleted(setup.caseDir, manifest, setup.scenario), /non-live attempt/);
  assert.equal(readJson(join(evidence, "provenance.json")).kind, "offline_test_fixture");
  const system = readFileSync(join(evidence, "system.txt"), "utf8");
  assert.match(system, /Offline fixture: include the observed/);
  assert.doesNotMatch(system, /Shared GPU spending policy|AGENTS.md - reasonese/);
  const messages = readJson(join(evidence, "messages.json"));
  assert.equal(messages[0].role, "user");
  assert.equal(messages[0].content[0].text, setup.scenario.prompt);
  assert.equal(messages.at(-1).stopReason, "stop");
  assert.ok(messages.some((message) => message.role === "toolResult" && message.content[0].text.includes("43")));
  const persisted = readFileSync(result.sessionFile, "utf8").trim().split("\n").map(JSON.parse)
    .filter((entry) => entry.type === "message").map((entry) => entry.message);
  assert.deepEqual(persisted, messages);
  assert.match(readFileSync(join(evidence, "http/001.response.txt"), "utf8"), /explicitly-offline-fixture/);
  assert.match(readFileSync(join(evidence, "changes.patch"), "utf8"), /print\(43\)/);
  for (const request of mock.requests) {
    assert.equal(request.model, MODEL);
    assert.equal(request.max_tokens, manifest.maxOutputTokens);
    assert.equal(request.reasoning.effort, "high");
  }
  assert.ok(mock.requests.at(-1).messages.some((message) => message.role === "tool" && message.content.includes("43")));
  for (const filename of readdirSync(join(evidence, "http"))) {
    assert.doesNotMatch(readFileSync(join(evidence, "http", filename), "utf8"), /fixture-secret/);
  }
  await assert.rejects(runTrajectory({ ...setup, ...mock, manifest, metadata, apiKey: "fixture-secret" }), /revision changed/);
  assert.equal(mock.requests.length, 6);
});

test("tools cannot read host secrets, inherit credentials, write outside checkout, or reach host network", { timeout: 20_000 }, async (t) => {
  const { caseDir, cwd } = await fixture(t);
  const secret = join(caseDir, "host-secret");
  writeFileSync(secret, "do not expose this");
  symlinkSync(secret, join(cwd, "escape"));
  const tools = sandboxTools(cwd, join(caseDir, "tool-state"), 5);
  const get = (name) => tools.find((tool) => tool.name === name);
  const unicode = "€🙂汉字\n".repeat(3000);
  await get("write").execute("unicode", { path: "unicode.txt", content: unicode });
  assert.equal(readFileSync(join(cwd, "unicode.txt"), "utf8"), unicode);
  await assert.rejects(get("read").execute("outside", { path: secret }));
  await assert.rejects(get("read").execute("symlink", { path: "escape" }));
  await assert.rejects(get("write").execute("write", { path: secret, content: "changed" }));
  assert.equal(readFileSync(secret, "utf8"), "do not expose this");
  const mock = await server(t, (_, res) => res.end("bad"));
  const code = `import os,socket; assert not os.environ.get('OPENROUTER_API_KEY'); s=socket.socket(); assert s.connect_ex(('127.0.0.1',${new URL(mock.fixtureEndpoint).port})) != 0; s.close(); s=socket.socket(); s.bind(('127.0.0.1',0)); print('isolated',s.getsockname()[1]); s.close()`;
  const result = await get("bash").execute("env", { command: `python -c "${code}"` });
  assert.match(result.content[0].text, /isolated \d+/);
  await assert.rejects(get("bash").execute("timeout", { command: "sleep 30" }), /time limit/);
});

test("request limit keeps a partial transcript and stops before another HTTP request", { timeout: 30_000 }, async (t) => {
  const setup = await fixture(t);
  const mock = await server(t, (index, res) => completion(res, index, ["read", { path: "README.md" }]));
  const result = await runTrajectory({ ...setup, ...mock, manifest: { ...manifest, maxModelCalls: 2 }, metadata, apiKey: "fixture" });
  assert.equal(result.status, "model_request_limit");
  assert.equal(mock.requests.length, 2);
  assert.equal(result.toolResults, 2);
  assert.ok(readJson(join(setup.caseDir, "trajectory/messages.json")).length >= 5);
  await assert.rejects(runTrajectory({ ...setup, ...mock, manifest, metadata, apiKey: "fixture" }), /EEXIST/);
  assert.equal(mock.requests.length, 2);
});

test("oversized input is rejected before HTTP", async (t) => {
  const setup = await fixture(t);
  const mock = await server(t, (index, res) => completion(res, index));
  const result = await runTrajectory({ ...setup, ...mock, manifest: { ...manifest, maxRequestBytes: 10 }, metadata, apiKey: "fixture" });
  assert.equal(result.status, "request_size_limit");
  assert.equal(mock.requests.length, 0);
});

test("provider errors are retained and not retried", async (t) => {
  const setup = await fixture(t);
  const mock = await server(t, (_, res) => { res.writeHead(500); res.end('{"error":{"message":"offline failure"}}'); });
  const result = await runTrajectory({ ...setup, ...mock, manifest, metadata, apiKey: "fixture" });
  assert.equal(result.status, "failed");
  assert.equal(result.modelRequests, 1);
  assert.equal(mock.requests.length, 1);
  assert.match(readFileSync(join(setup.caseDir, "trajectory/http/001.response.txt"), "utf8"), /offline failure/);
});

test("wall timeout aborts a stalled stream and retains its bytes", { timeout: 30_000 }, async (t) => {
  const setup = await fixture(t);
  const mock = await server(t, (_, res) => {
    res.writeHead(200, { "Content-Type": "text/event-stream" });
    res.write(': an explicitly offline stalled stream\n\n');
  });
  const result = await runTrajectory({ ...setup, ...mock, manifest: { ...manifest, timeoutSeconds: 1 }, metadata, apiKey: "fixture" });
  assert.equal(result.status, "time_limit");
  assert.match(readFileSync(join(setup.caseDir, "trajectory/http/001.response.txt"), "utf8"), /stalled stream/);
});

test("length cutoff is not counted as a completed trajectory", async (t) => {
  const setup = await fixture(t);
  const mock = await server(t, (index, res) => completion(res, index, null, "length"));
  const result = await runTrajectory({ ...setup, ...mock, manifest, metadata, apiKey: "fixture" });
  assert.equal(result.status, "incomplete");
});

test("manifest and symlink provenance fail closed", async (t) => {
  const setup = await fixture(t);
  assert.equal(contextFile(setup.cwd, setup.scenario).resolvedPath, "CLAUDE.md");
  assert.throws(() => contextFile(setup.cwd, { ...setup.scenario, ruleFile: "README.md" }), /differs/);
  const filename = join(setup.caseDir, "bad-manifest.json");
  for (const change of [{ model: "other-model" }, { maxModelCalls: 0 },
    { scenarios: [manifest.scenarios[0], manifest.scenarios[0]] },
    { scenarios: [{ ...manifest.scenarios[0], revision: "main" }] }]) {
    writeJson(filename, { ...manifest, ...change });
    assert.throws(() => loadManifest(filename));
  }
});

test("Python launcher aliases preserve the project interpreter inside the sandbox", async (t) => {
  const { caseDir, cwd } = await fixture(t);
  const environment = join(caseDir, "venv");
  mkdirSync(join(environment, "bin"), { recursive: true });
  const launcher = join(caseDir, "python-launcher");
  symlinkSync(realpathSync(join(projectDir, ".venv/bin/python")), launcher);
  symlinkSync(launcher, join(environment, "bin/python"));
  writeFileSync(join(environment, "pyvenv.cfg"), readFileSync(join(projectDir, ".venv/pyvenv.cfg")));
  const result = spawnSync("bwrap", sandboxArgs(cwd, join(caseDir, "tool-state"), environment), {
    input: JSON.stringify({ cwd, name: "bash", id: "python-alias", parameters: {
      command: "python -c 'import sys; print(sys.prefix); print(sys.version_info[:2])'",
    } }), encoding: "utf8", env: { PATH: "/usr/bin:/bin" }, timeout: 10_000,
  });
  assert.equal(result.status, 0, result.stderr);
  const output = JSON.parse(result.stdout);
  assert.equal(output.error, undefined, output.error);
  assert.equal(output.result.content[0].text, `${environment}\n(3, 13)\n`);
});

test("preparation checks out exact commits from cached refs with independent Git objects", { timeout: 60_000 }, async (t) => {
  const setup = await fixture(t);
  const scenario = { ...setup.scenario, repository: `file://${setup.cwd}` };
  const second = { ...scenario, id: "second-fixture" };
  const plan = { ...manifest, scenarios: [scenario, second] };
  const output = join(setup.caseDir, "prepared");
  await prepare(plan, output, join(setup.caseDir, "sources"));
  const firstCwd = join(output, scenario.id, "workspace");
  const secondCwd = join(output, second.id, "workspace");
  assert.equal((await git(firstCwd, "rev-parse", "HEAD")).trim(), scenario.revision);
  assert.equal((await git(secondCwd, "rev-parse", "HEAD")).trim(), scenario.revision);
  assert.equal(readFileSync(join(firstCwd, "README.md"), "utf8"), readFileSync(join(setup.cwd, "README.md"), "utf8"));
  const packs = ".git/objects/pack";
  const pack = readdirSync(join(firstCwd, packs)).find((name) => name.endsWith(".pack"));
  assert.ok(pack);
  assert.notEqual(statSync(join(firstCwd, packs, pack)).ino, statSync(join(secondCwd, packs, pack)).ino);
  assert.equal(readJson(join(output, scenario.id, "prepared.json")).context.resolvedPath, "CLAUDE.md");
  writeFileSync(join(firstCwd, "new.txt"), "first workspace only");
  await git(firstCwd, "add", "new.txt");
  await git(firstCwd, "commit", "-m", "test: independent commit");
  assert.equal((await git(secondCwd, "status", "--porcelain")).trim(), "");
  assert.equal((await git(secondCwd, "rev-parse", "HEAD")).trim(), scenario.revision);
  mkdirSync(join(output, scenario.id, "trajectory"));
  await assert.rejects(prepare(plan, output, join(setup.caseDir, "sources")), /Collection already started/);
});

test("CLI paid opt-in is checked before any provider access", async (t) => {
  const output = mkdtempSync(join(projectDir, "out/pi-cli-test-"));
  t.after(() => rmSync(output, { recursive: true, force: true }));
  writeJson(join(output, "plan.json"), { manifest, collector: collectorIdentity() });
  const original = globalThis.fetch;
  globalThis.fetch = () => { throw new Error("must not access provider"); };
  try {
    await assert.rejects(main(["collect", "--output", output]), /--allow-paid is required/);
    writeJson(join(output, "plan.json"), { manifest, collector: {} });
    await assert.rejects(main(["collect", "--output", output, "--allow-paid"]), /collector changed/);
  } finally { globalThis.fetch = original; }
});

test("adaptive user turns retain one native session, observed history, and cumulative evidence", async (t) => {
  const setup = await fixture(t);
  setup.scenario.maxUserTurns = 3;
  const mock = await server(t, (index, res) => completion(res, index));
  let reviews = 0;
  const outcome = await runTrajectory({ ...setup, ...mock, manifest, metadata, apiKey: "fixture",
    nextTurn: async ({ turn, messages, directory }) => {
      reviews++;
      assert.equal(turn, reviews);
      assert.equal(messages.at(-1).stopReason, "stop");
      assert.deepEqual(readJson(join(directory, `turns/${String(turn).padStart(3, "0")}.messages.json`)), messages);
      if (turn === 2) return null;
      return { prompt: `You reported: ${messages.at(-1).content[0].text} Please check it again.`,
        rationale: "Follow-up grounded in the actual preceding assistant response" };
    } });
  assert.equal(outcome.status, "completed");
  assert.equal(outcome.userTurns, 2);
  assert.equal(outcome.modelRequests, 2);
  assert.equal(reviews, 2);
  assert.equal(mock.requests[1].messages.filter(m => m.role === "user").length, 2);
  assert.ok(mock.requests[1].messages.some(m => m.role === "assistant" && m.content.includes("Observed output: 43")));
  const messages = readJson(join(setup.caseDir, "trajectory/messages.json"));
  const native = readFileSync(outcome.sessionFile, "utf8").trim().split("\n").map(JSON.parse)
    .filter(e => e.type === "message").map(e => e.message);
  assert.deepEqual(native, messages);
  const second = readJson(join(setup.caseDir, "trajectory/turns/002.input.json"));
  assert.match(second.rationale, /actual preceding/);
  assert.match(second.basedOnMessagesSha256, /^[a-f0-9]{64}$/);
  assert.ok(outcome.artifactHashes["turns/001.patch"]);
});

test("request budget is shared by all user turns", async (t) => {
  const setup = await fixture(t);
  setup.scenario.maxUserTurns = 2;
  const mock = await server(t, (index, res) => completion(res, index));
  const outcome = await runTrajectory({ ...setup, ...mock, manifest: { ...manifest, maxModelCalls: 1 }, metadata, apiKey: "fixture",
    nextTurn: async () => ({ prompt: "Please check again.", rationale: "Budget boundary test" }) });
  assert.equal(outcome.status, "model_request_limit");
  assert.equal(outcome.userTurns, 2);
  assert.equal(mock.requests.length, 1);
});

test("operator wait is bounded and cannot hide a timeout as completion", async (t) => {
  const setup = await fixture(t);
  setup.scenario.maxUserTurns = 2;
  const mock = await server(t, (index, res) => completion(res, index));
  const outcome = await runTrajectory({ ...setup, ...mock, manifest: { ...manifest, timeoutSeconds: 1 }, metadata, apiKey: "fixture",
    nextTurn: async () => new Promise(() => {}) });
  assert.equal(outcome.status, "time_limit");
  assert.equal(outcome.modelRequests, 1);
});

test("multi-turn scenarios require an operator before starting an attempt", async (t) => {
  const setup = await fixture(t);
  setup.scenario.maxUserTurns = 2;
  await assert.rejects(runTrajectory({ ...setup, manifest, metadata, apiKey: "fixture" }), /interactive operator/);
});

test("Kimi selection and plaintext reasoning survive tools and follow-up turns", async (t) => {
  const setup = await fixture(t);
  setup.scenario.maxUserTurns = 2;
  const selected = { ...manifest, model: KIMI_MODEL };
  const mock = await server(t, (index, res) => completion(res, index,
    index === 0 ? ["read", { path: "README.md" }] : null, "stop",
    { model: KIMI_MODEL, reasoning: `Readable offline reasoning for request ${index}.` }));
  const outcome = await runTrajectory({ ...setup, ...mock, manifest: selected,
    metadata: { ...metadata, id: KIMI_MODEL }, apiKey: "fixture",
    nextTurn: async () => ({ prompt: "Now implement the task.", rationale: "Observed the investigation response" }) });
  assert.equal(outcome.status, "completed", JSON.stringify(outcome));
  assert.equal(outcome.modelRequests, 3);
  assert.equal(outcome.userTurns, 2);
  const messages = readJson(join(setup.caseDir, "trajectory/messages.json"));
  const assistants = messages.filter(m => m.role === "assistant");
  assert.equal(assistants.length, 3);
  assistants.forEach((m, i) => {
    assert.equal(m.model, KIMI_MODEL);
    assert.equal(m.content.find(b => b.type === "thinking").thinking, `Readable offline reasoning for request ${i}.`);
  });
  mock.requests.forEach(r => assert.equal(r.model, KIMI_MODEL));
  assert.equal(mock.requests[1].messages.find(m => m.role === "assistant").reasoning, "Readable offline reasoning for request 0.");
  assert.equal(mock.requests[2].messages.filter(m => m.role === "assistant")[1].reasoning, "Readable offline reasoning for request 1.");
  assert.match(readFileSync(join(setup.caseDir, "trajectory/http/001.response.txt"), "utf8"), /reasoning.text/);
  const native = readFileSync(outcome.sessionFile, "utf8").trim().split("\n").map(JSON.parse)
    .filter(e => e.type === "message").map(e => e.message);
  assert.deepEqual(native, messages);
});

test("model metadata follows the requested slug and mismatches stop before a session", async (t) => {
  const original = globalThis.fetch;
  globalThis.fetch = async () => ({ ok: true, json: async () => ({ data: [
    { ...metadata, supported_parameters: ["tools", "reasoning"] },
    { ...metadata, id: KIMI_MODEL, supported_parameters: ["tools", "reasoning"] },
  ] }) });
  try {
    assert.equal((await modelMetadata(KIMI_MODEL)).id, KIMI_MODEL);
    assert.equal((await modelMetadata()).id, MODEL);
    await assert.rejects(modelMetadata("missing/model"), /unavailable/);
  } finally { globalThis.fetch = original; }
  const setup = await fixture(t);
  await assert.rejects(runTrajectory({ ...setup, manifest: { ...manifest, model: KIMI_MODEL }, metadata, apiKey: "fixture" }), /Wrong model metadata/);
});

test("transport rejects a substituted model or fallback list before HTTP", async (t) => {
  const setup = await fixture(t);
  const transport = captureTransport(join(setup.caseDir, "capture"), { ...manifest, model: KIMI_MODEL }, "http://127.0.0.1:1/v1");
  try {
    for (const payload of [
      { model: MODEL, max_tokens: manifest.maxOutputTokens },
      { model: KIMI_MODEL, models: [MODEL], max_tokens: manifest.maxOutputTokens },
    ]) {
      await assert.rejects(fetch("http://127.0.0.1:1/v1/chat/completions", { body: JSON.stringify(payload) }), /Unexpected model, fallback/);
    }
    assert.equal(transport.state.requests, 0);
  } finally { await transport.close(); }
});
