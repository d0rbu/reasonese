import assert from "node:assert/strict";
import test from "node:test";
import { frontierProvider } from "../frontier-provider.mjs";
import { branchTurns, validateChecks } from "../frontier.mjs";

test("Claude returns signed thinking and matching tool IDs unchanged", () => {
  const api = frontierProvider({ provider: "anthropic", model: "claude-opus-5-5", effort: "high" });
  const history = [api.user("Original Kimi prompt")];
  const content = [
    { type: "thinking", thinking: "", signature: "opaque" },
    { type: "redacted_thinking", data: "opaque-redaction" },
    { type: "tool_use", id: "toolu_1", name: "read", input: { path: "README.md" } },
  ];
  const raw = { model: "claude-opus-5-5", stop_reason: "tool_use", content };
  const result = api.decode(raw);
  assert.equal(result.signed, true);
  assert.deepEqual(result.calls[0], { id: "toolu_1", name: "read", args: { path: "README.md" } });
  history.push(...result.saved, ...api.results([{ id: result.calls[0].id, text: "file content", error: false }]));
  const next = api.request(history, "system", [], 8192);
  assert.deepEqual(next.messages[1], { role: "assistant", content });
  assert.equal(next.messages[2].content[0].tool_use_id, "toolu_1");
  assert.equal(next.thinking.block_binding.prefix_mismatch_behavior, "error");
  assert.throws(() => api.decode({ ...raw, input_transformations: [{ reason: "prefix_binding_mismatch" }] }), /changed/);
});

test("Responses stateless history keeps all output items in provider order", () => {
  const api = frontierProvider({ provider: "openai", model: "gpt-6.2-astra", effort: "high" });
  const user = api.user("Original Kimi prompt");
  const output = [
    { type: "reasoning", id: "rs_1", encrypted_content: "opaque", summary: [] },
    { type: "function_call", id: "fc_1", call_id: "call_1", name: "read", arguments: "{}" },
  ];
  const raw = { model: "gpt-6.2-astra", status: "completed", output };
  const result = api.decode(raw);
  assert.equal(result.signed, true);
  const toolResult = api.results([{ id: result.calls[0].id, text: "content", error: false }]);
  const payload = api.request([user, ...result.saved, ...toolResult], "system", [], 8192);
  assert.equal(payload.store, false);
  assert.equal(payload.reasoning.context, "all_turns");
  assert.deepEqual(payload.input, [user, ...output, { type: "function_call_output", call_id: "call_1", output: "content" }]);
  assert.throws(() => api.decode({ ...raw, status: "incomplete" }), /did not complete/);
  assert.throws(() => api.decode({ ...raw, output: [{ type: "reasoning", summary: [] }] }), /Incomplete opaque/);
});

test("plans reuse only their original prefix and preserve both delivery schedules", () => {
  const prefix = [{ role: "user" }, { role: "assistant" }];
  const turns = [{ prompt: "B" }];
  const immediate = { prefix: [...prefix, { role: "user" }], turns,
    metadata: { historicalPlacements: [{ message: 2 }], authoringInputs: [{ content: "A variant" }] } };
  assert.deepEqual(branchTurns(immediate, ["investigate", "A"]), { unchanged: 1, turns: [["A variant", "B"]] });
  const delayed = { prefix, turns: [{ prompt: "A variant" }, ...turns], metadata: {} };
  assert.deepEqual(branchTurns(delayed, ["investigate", "A"]), { unchanged: 1, turns: [["A variant"], ["B"]] });
  assert.deepEqual(branchTurns({ prefix: [], turns, metadata: {} }, ["investigate", "A"]), { unchanged: 0, turns: [["B"]] });
});

test("qualification requires checks at every turn and confines paths to workspace files", () => {
  assert.doesNotThrow(() => validateChecks([[{ kind: "answer", contains: "observed" }],
    [{ kind: "file", path: "src/implementation.py", contains: "expected" }]], 2));
  assert.throws(() => validateChecks([[]], 1), /nonempty/);
  assert.throws(() => validateChecks([[{ kind: "file", path: "src/../other", missing: true }]], 1), /relative/);
  assert.throws(() => validateChecks([[{ kind: "file", path: "src/implementation.py", contains: "expected", missing: true }]], 1), /exactly one/);
});
