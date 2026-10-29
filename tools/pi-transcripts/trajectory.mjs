import { isDeepStrictEqual } from "node:util";
import { appendFileSync, existsSync, mkdirSync, readFileSync, readdirSync, realpathSync, writeFileSync } from "node:fs";
import { join, relative } from "node:path";
import {
  AuthStorage, createAgentSession, DefaultResourceLoader, ModelRegistry,
  SessionManager, SettingsManager,
} from "@mariozechner/pi-coding-agent";
import { neutralCwd, sandboxGit, sandboxTools, toolDir } from "./sandbox.mjs";
import { buildSystemPrompt } from "./node_modules/@mariozechner/pi-coding-agent/dist/core/system-prompt.js";
import { isNeutralCase } from "./backing.mjs";
import { snapshotWorkspace } from "./checkpoints.mjs";
import { collectorIdentity, contextFile, MODEL, readJson, sha256, sha256File, validateBaseCollection, verifyWorkspace, writeJson } from "./workspaces.mjs";

function hasFinalAnswer(messages) {
  const last = messages.at(-1);
  return last?.role === "assistant" && last.stopReason === "stop" &&
    last.content.some((block) => block.type === "text" && block.text.trim());
}

function artifactHashes(directory, prefix = "") {
  const hashes = {};
  for (const entry of readdirSync(join(directory, prefix), { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
    const name = join(prefix, entry.name);
    if (name === "outcome.json") continue;
    if (entry.isDirectory()) Object.assign(hashes, artifactHashes(directory, name));
    else if (entry.isFile()) hashes[name] = sha256File(join(directory, name));
    else throw new Error(`Unexpected evidence entry: ${name}`);
  }
  return hashes;
}

export function verifyCompleted(caseDir, manifest, scenario) {
  const directory = join(caseDir, "trajectory");
  const outcome = readJson(join(directory, "outcome.json"));
  const provenance = readJson(join(directory, "provenance.json"));
  if (outcome.status !== "completed" || !hasFinalAnswer(readJson(join(directory, "messages.json"))) ||
    provenance.kind !== "live_pi_session" ||
    JSON.stringify(provenance.manifest) !== JSON.stringify(manifest) ||
    JSON.stringify(provenance.scenario) !== JSON.stringify(scenario) ||
    JSON.stringify(provenance.collector) !== JSON.stringify(collectorIdentity()) ||
    JSON.stringify(outcome.artifactHashes) !== JSON.stringify(artifactHashes(directory))) {
    throw new Error(`Incomplete, altered, or non-live attempt: ${scenario.id}; inspect it and use a fresh output directory for any authorized retry`);
  }
  return outcome;
}

export async function modelMetadata(requestedModel = MODEL) {
  const response = await fetch("https://openrouter.ai/api/v1/models", { signal: AbortSignal.timeout(30_000) });
  if (!response.ok) throw new Error(`OpenRouter model lookup: HTTP ${response.status}`);
  const metadata = (await response.json()).data.find((model) => model.id === requestedModel);
  if (!metadata?.supported_parameters.includes("tools") || !metadata.supported_parameters.includes("reasoning")) {
    throw new Error(`Requested model is unavailable or lacks tools/reasoning: ${requestedModel}`);
  }
  return metadata;
}

function modelDefinition(metadata, baseUrl) {
  const price = (name) => {
    const value = Number(metadata.pricing[name]);
    if (!Number.isFinite(value) || value < 0) throw new Error(`Invalid model price: ${name}`);
    return value * 1_000_000;
  };
  return {
    id: metadata.id, name: metadata.name, api: "openai-completions", provider: "openrouter", baseUrl,
    reasoning: true, input: ["text"], contextWindow: metadata.context_length,
    maxTokens: metadata.top_provider.max_completion_tokens,
    cost: { input: price("prompt"), output: price("completion"), cacheRead: price("input_cache_read"), cacheWrite: 0 },
    compat: { supportsDeveloperRole: false, supportsStore: false, thinkingFormat: "openrouter",
      maxTokensField: "max_tokens", openRouterRouting: { require_parameters: true } },
  };
}

// Capture the actual request bodies and raw SSE bytes, independently of Pi's normalized messages.
// This also counts real HTTP requests, so a transport retry cannot evade the request limit.
export function captureTransport(directory, manifest, baseUrl) {
  mkdirSync(directory);
  const originalFetch = globalThis.fetch;
  const pending = [];
  const state = { requests: 0, limit: null, captureErrors: [] };
  globalThis.fetch = async (url, init) => {
    if (String(url) !== `${baseUrl}/chat/completions`) throw new Error("Unexpected model endpoint");
    const body = init?.body;
    if (typeof body !== "string") throw new Error("Expected a JSON model request");
    const payload = JSON.parse(body);
    if (payload.model !== manifest.model || payload.models || payload.route || payload.max_tokens !== manifest.maxOutputTokens) {
      throw new Error("Unexpected model, fallback, or token limit");
    }
    if (state.requests >= manifest.maxModelCalls || Buffer.byteLength(body) > manifest.maxRequestBytes) {
      state.limit = state.requests >= manifest.maxModelCalls ? "model_request_limit" : "request_size_limit";
      throw new Error(state.limit);
    }
    const index = ++state.requests;
    const prefix = join(directory, String(index).padStart(3, "0"));
    writeFileSync(`${prefix}.request.json`, body + "\n");
    // Authorization headers deliberately never enter any artifact.
    const response = await originalFetch(url, init);
    writeJson(`${prefix}.response.json`, {
      status: response.status,
      headers: Object.fromEntries([...response.headers].filter(([name]) =>
        ["content-type", "x-request-id", "date"].includes(name))),
    });
    const copy = response.clone();
    writeFileSync(`${prefix}.response.txt`, "");
    pending.push((async () => {
      try {
        for await (const chunk of copy.body) appendFileSync(`${prefix}.response.txt`, chunk);
      } catch (error) {
        state.captureErrors.push({ request: index, error: error.message });
      }
    })());
    return response;
  };
  return { state, close: async () => { globalThis.fetch = originalFetch; await Promise.all(pending); } };
}

export async function runTrajectory({ manifest, scenario, caseDir, metadata, apiKey,
  fixtureEndpoint = null, nextTurn = null, continuation = null }) {
  if (!continuation) validateBaseCollection(manifest, [scenario]);
  if ((scenario.maxUserTurns ?? 1) > 1 && !nextTurn && !continuation) throw new Error("Multi-turn scenarios require an interactive operator");
  const baseUrl = fixtureEndpoint ?? "https://openrouter.ai/api/v1";
  if (fixtureEndpoint && !/^http:\/\/127\.0\.0\.1:\d+\/v1$/.test(fixtureEndpoint)) throw new Error("Fixtures must use localhost");
  if (metadata.id !== manifest.model) throw new Error("Wrong model metadata");
  caseDir = realpathSync(caseDir);
  if (!fixtureEndpoint && !isNeutralCase(caseDir)) throw new Error("Live collection requires neutral physical backing");
  const cwd = join(caseDir, "workspace");
  const context = continuation ? contextFile(cwd, scenario) : await verifyWorkspace(cwd, scenario);
  if (continuation?.checkpointGitState) {
    const expected = continuation.checkpointGitState;
    const head = (await sandboxGit(cwd, join(caseDir, "tool-state"), "rev-parse", "HEAD")).trim();
    const status = await sandboxGit(cwd, join(caseDir, "tool-state"), "status", "--porcelain=v1", "--untracked-files=all");
    if (head !== expected.head || status !== expected.status) throw new Error("Restored Git state differs from source checkpoint");
  }
  const directory = join(caseDir, "trajectory");
  // An existing attempt, including a partial or failed one, is never rerun or overwritten.
  mkdirSync(directory);
  const provenance = {
    kind: fixtureEndpoint ? "offline_test_fixture" : continuation ? "live_pi_continuation" : "live_pi_session", scenario,
    continuation: continuation?.provenance, seededMessageCount: continuation?.prefix.length ?? 0,
    collectionPurpose: manifest.collectionPurpose,
    manifest, collector: collectorIdentity(), modelMetadata: metadata,
    context, startedAt: new Date().toISOString(), taskSuccess: "unjudged",
    pythonEnvironment: existsSync(join(caseDir, "prepared.json")) ? readJson(join(caseDir, "prepared.json")).pythonEnvironment : undefined,
    toolEnvironment: existsSync(join(caseDir, "prepared.json")) ? readJson(join(caseDir, "prepared.json")).toolEnvironment : undefined,
    userAuthor: continuation ? "Per-input variant authors; see hashed authoring inputs and source provenance" : "Codex operator; authored task and adaptive follow-ups, not messages from a human participant",
  };
  writeJson(join(directory, "provenance.json"), provenance);
  writeFileSync(join(directory, "prompt.txt"), scenario.prompt + "\n");
  writeJson(join(directory, "outcome.json"), { status: "running", startedAt: provenance.startedAt });
  let session, transport, timer, interruption, failure;
  let userTurns = 0;
  const controller = new AbortController();
  const interrupt = (reason) => { interruption = reason; controller.abort(); void session?.abort(); };
  const onSignal = () => interrupt("interrupted");
  process.once("SIGINT", onSignal);
  process.once("SIGTERM", onSignal);
  try {
    const authStorage = AuthStorage.inMemory();
    authStorage.setRuntimeApiKey("openrouter", apiKey);
    const modelRegistry = ModelRegistry.inMemory(authStorage);
    const model = modelDefinition(metadata, baseUrl);
    modelRegistry.registerProvider("openrouter", {
      baseUrl, api: model.api, apiKey: "OPENROUTER_API_KEY", models: [model],
    });
    const settingsManager = SettingsManager.inMemory({
      compaction: { enabled: false }, retry: { enabled: false },
    });
    const agentDir = join(directory, "agent");
    const tools = sandboxTools(cwd, join(caseDir, "tool-state"), manifest.toolTimeoutSeconds, provenance.pythonEnvironment?.path, provenance.toolEnvironment);
    const baseSystem = buildSystemPrompt({ cwd: neutralCwd(), selectedTools: tools.map(t => t.name),
      toolSnippets: Object.fromEntries(tools.map(t => [t.name, t.promptSnippet])),
      promptGuidelines: tools.flatMap(t => t.promptGuidelines ?? []), contextFiles: [], skills: [] })
      .replace(/\nCurrent date: [^\n]+\nCurrent working directory: [^\n]+$/, "")
      .replaceAll(realpathSync(join(toolDir, "node_modules")), "/opt/pi/node_modules")
      .replaceAll(join(toolDir, "node_modules"), "/opt/pi/node_modules");
    const resourceLoader = new DefaultResourceLoader({
      cwd: neutralCwd(), agentDir, settingsManager,
      noExtensions: true, noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true,
      // Pi's ordinary system prompt + the real project's root context, without host ancestors.
      agentsFilesOverride: () => ({ agentsFiles: [{ path: join(neutralCwd(), relative(cwd, context.path)), content: context.content }] }),
      systemPromptOverride: () => baseSystem, appendSystemPromptOverride: () => [],
    });
    await resourceLoader.reload();
    const sessionManager = SessionManager.create(neutralCwd(), join(directory, "sessions"));
    ({ session } = await createAgentSession({
      cwd: neutralCwd(), agentDir, resourceLoader, authStorage, modelRegistry, model,
      thinkingLevel: manifest.thinking, settingsManager,
      tools: ["read", "bash", "edit", "write"],
      customTools: tools,
      sessionManager,
    }));
    if (continuation) {
      const definitions = session.agent.state.tools.map(({ name, description, parameters }) => ({ name, description, parameters }));
      if (!isDeepStrictEqual(definitions, continuation.tools)) throw new Error("Continuation tool definitions differ from source");
      session.agent.state.messages = structuredClone(continuation.prefix);
      for (const message of continuation.prefix) sessionManager.appendMessage(structuredClone(message));
    }
    writeFileSync(join(directory, "system.txt"), continuation?.system ?? session.systemPrompt);
    writeJson(join(directory, "tools.json"), session.agent.state.tools.map(({ name, description, parameters }) =>
      ({ name, description, parameters })));
    session.subscribe((event) => appendFileSync(join(directory, "events.jsonl"), JSON.stringify(event) + "\n"));
    const stream = session.agent.streamFn;
    session.agent.streamFn = (selectedModel, modelContext, options) => stream(selectedModel, continuation ? { ...modelContext, systemPrompt: continuation.system } : modelContext, {
      ...options, maxTokens: manifest.maxOutputTokens, maxRetries: 0,
      timeoutMs: manifest.timeoutSeconds * 1000,
    });
    transport = captureTransport(join(directory, "http"), manifest, baseUrl);
    mkdirSync(join(directory, "turns"));
    writeJson(join(directory, "turns/000.workspace.json"), await snapshotWorkspace(caseDir, join(directory, "turns/000"), {
      head: (await sandboxGit(cwd, join(caseDir, "tool-state"), "rev-parse", "HEAD")).trim(),
      status: await sandboxGit(cwd, join(caseDir, "tool-state"), "status", "--porcelain=v1", "--untracked-files=all"),
    }));
    timer = setTimeout(() => interrupt("time_limit"), manifest.timeoutSeconds * 1000);
    const queued = continuation ? structuredClone(continuation.turns) : null;
    let turn = queued ? queued.shift() : { prompt: scenario.prompt, rationale: "Initial task from the prepared manifest" };
    while (turn && !interruption) {
      if (typeof turn.prompt !== "string" || !turn.prompt.trim() || typeof turn.rationale !== "string" || !turn.rationale.trim()) {
        throw new Error("Each follow-up needs a nonempty prompt and operator rationale");
      }
      const prefix = join(directory, "turns", String(++userTurns).padStart(3, "0"));
      const input = { ...turn, turn: userTurns, authoredAt: new Date().toISOString(),
        basedOnMessagesSha256: sha256(JSON.stringify(session.messages)) };
      writeJson(`${prefix}.input.json`, input);
      await session.prompt(turn.prompt);
      writeJson(join(directory, "messages.json"), session.messages);
      writeJson(`${prefix}.messages.json`, session.messages);
      const diff = await sandboxGit(cwd, join(caseDir, "tool-state"), "diff", "--binary", "--no-ext-diff", "--no-textconv", scenario.revision);
      writeFileSync(`${prefix}.patch`, diff);
      writeJson(`${prefix}.workspace.json`, await snapshotWorkspace(caseDir, prefix, {
        head: (await sandboxGit(cwd, join(caseDir, "tool-state"), "rev-parse", "HEAD")).trim(),
        status: await sandboxGit(cwd, join(caseDir, "tool-state"), "status", "--porcelain=v1", "--untracked-files=all"),
      }));
      if (interruption || !hasFinalAnswer(session.messages) || userTurns >= (scenario.maxUserTurns ?? 1)) break;
      // The same Pi session and workspace survive the operator's review. No synthetic
      // assistant history, hidden steering, or replacement tool results are inserted.
      turn = queued ? queued.shift() : await Promise.race([
        nextTurn({ turn: userTurns, messages: session.messages, directory, signal: controller.signal }),
        new Promise((resolve) => controller.signal.addEventListener("abort", () => resolve(null), { once: true })),
      ]);
    }
  } catch (error) {
    failure = error.message;
  } finally {
    clearTimeout(timer);
    process.removeListener("SIGINT", onSignal);
    process.removeListener("SIGTERM", onSignal);
    await transport?.close();
  }
  const messages = session?.messages ?? [];
  writeJson(join(directory, "messages.json"), messages);
  const last = messages.at(-1);
  let status = interruption ?? transport?.state.limit ??
    (failure || transport?.state.captureErrors.length || last?.stopReason === "error" ? "failed" :
      hasFinalAnswer(messages) ? "completed" : "incomplete");
  let finalGit;
  try {
    const check = (...args) => sandboxGit(cwd, join(caseDir, "tool-state"), ...args);
    finalGit = {
      head: (await check("rev-parse", "HEAD")).trim(),
      status: await check("status", "--porcelain=v1", "--untracked-files=all"),
      log: await check("log", "-3", "--format=fuller"),
    };
    writeFileSync(join(directory, "changes.patch"), await check("diff", "--binary", "--no-ext-diff", "--no-textconv", scenario.revision));
  } catch (error) { failure = error.message; status = "failed"; }
  const outcome = {
    status, error: failure ?? last?.errorMessage ?? null, taskSuccess: "unjudged",
    startedAt: provenance.startedAt, finishedAt: new Date().toISOString(),
    modelRequests: transport?.state.requests ?? 0, captureErrors: transport?.state.captureErrors ?? [],
    toolResults: messages.slice(continuation?.prefix.length ?? 0).filter((message) => message.role === "toolResult").length,
    userTurns,
    assistantUsage: messages.slice(continuation?.prefix.length ?? 0).filter((message) => message.role === "assistant").map((message) => message.usage),
    sessionFile: session?.sessionFile, finalGit,
    messagesSha256: sha256(readFileSync(join(directory, "messages.json"))),
    // Workspaces and tool-state remain alongside the evidence, including untracked/ignored outputs.
    workspaceRetained: existsSync(cwd),
  };
  session?.dispose();
  outcome.artifactHashes = artifactHashes(directory);
  writeJson(join(directory, "outcome.json"), outcome);
  return outcome;
}
