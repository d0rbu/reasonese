#!/usr/bin/env node
import { createHash } from "node:crypto";
import { createReadStream, lstatSync, mkdirSync, readFileSync, realpathSync, writeFileSync } from "node:fs";
import { dirname, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { isDeepStrictEqual, parseArgs } from "node:util";
import { validateToolArguments } from "@mariozechner/pi-ai";
import { artifact, preflightPlan, validateHistory, validateRestoredState } from "./continue.mjs";
import { restoreCheckpoint, snapshotWorkspace } from "./checkpoints.mjs";
import { frontierProvider } from "./frontier-provider.mjs";
import { neutralCwd, projectDir, sandboxGit, sandboxTools } from "./sandbox.mjs";
import { KIMI_MODEL, readJson, sha256, writeJson } from "./workspaces.mjs";

function loadSource(descriptor) {
  const bytes = readFileSync(descriptor);
  const source = JSON.parse(bytes), root = dirname(descriptor);
  if (source.kind !== "pi_source_template" || source.captureKind !== "live_pi_session" ||
      source.manifest?.model !== KIMI_MODEL || !source.checkpoints?.["0"]) throw new Error("Expected a live Kimi base source with checkpoint 0");
  const messages = JSON.parse(artifact(root, source.messages));
  validateHistory(messages);
  if (messages[0]?.role !== "user" || messages.at(-1)?.role !== "assistant" || messages.at(-1)?.stopReason !== "stop" ||
      messages.some(message => message.role === "assistant" && (message.model !== KIMI_MODEL ||
        message.provider !== "openrouter" || !["stop", "toolUse"].includes(message.stopReason)))) throw new Error("Invalid Kimi base history");
  const prompts = messages.filter(message => message.role === "user").map(message => {
    if (!Array.isArray(message.content) || !message.content.length ||
        message.content.some(block => block.type !== "text" || typeof block.text !== "string")) throw new Error("Expected text user turn");
    return message.content.map(block => block.text).join("");
  });
  const system = artifact(root, source.system).toString();
  const tools = JSON.parse(artifact(root, source.tools));
  const actualTools = sandboxTools(neutralCwd(), "", 1).map(({ name, description, parameters }) => ({ name, description, parameters }));
  if (!isDeepStrictEqual(tools, actualTools)) throw new Error("Source tools differ from local sandbox tools");
  for (const limit of ["maxModelCalls", "maxOutputTokens", "maxRequestBytes", "timeoutSeconds", "toolTimeoutSeconds"]) {
    if (!Number.isSafeInteger(source.manifest[limit]) || source.manifest[limit] < 1) throw new Error(`Invalid source limit: ${limit}`);
  }
  return { source, root, sourceSha256: sha256(bytes), prompts, system, tools };
}

export function validateChecks(checks, turns) {
  if (!Array.isArray(checks) || checks.length !== turns || checks.some(turn => !Array.isArray(turn) || !turn.length)) {
    throw new Error("Provide one nonempty check list for every source user turn");
  }
  for (const check of checks.flat()) {
    if (!check || !["answer", "file"].includes(check.kind)) throw new Error("Expected answer or file check");
    const file = check.kind === "file";
    if (file && (typeof check.path !== "string" || check.path.split("/").some(part => !part || [".", "..", ".git"].includes(part)))) {
      throw new Error("Expected a workspace-relative file path");
    }
    const keys = ["contains", "sha256", "missing"].filter(key => Object.hasOwn(check, key));
    const text = typeof check.contains === "string" && Boolean(check.contains);
    const hash = typeof check.sha256 === "string" && /^[a-f0-9]{64}$/.test(check.sha256);
    if (keys.length !== 1 || !(text || file && (hash || check.missing === true))) throw new Error("Expected exactly one check predicate");
  }
}

async function checkOutcome(checks, text, workspace) {
  const results = [];
  for (const check of checks) {
    let passed;
    if (check.kind === "answer") {
      passed = text.includes(check.contains);
    } else {
      const pieces = check.path.split("/");
      try {
        let path = workspace;
        for (const part of pieces.slice(0, -1)) {
          path = join(path, part);
          if (!lstatSync(path).isDirectory()) throw new Error("Outcome check parent is not a workspace directory");
        }
        path = join(path, pieces.at(-1));
        const stat = lstatSync(path);
        passed = false;
        if (!check.missing && stat.isFile() && realpathSync(path) === path) {
          const hash = createHash("sha256");
          if (check.sha256) {
            for await (const chunk of createReadStream(path)) hash.update(chunk);
            passed = hash.digest("hex") === check.sha256;
          } else if (stat.size <= 8 * 1024 * 1024) {
            passed = readFileSync(path, "utf8").includes(check.contains);
          }
        }
      } catch (error) {
        if (error.code !== "ENOENT") throw error;
        passed = check.missing === true;
      }
    }
    results.push({ check, passed });
  }
  return results;
}

async function execute({ base, config, apiKey, output, checkpoint, archive, sourceCaseDir, prepared,
  prefix = [], priorModel = null, turns, checks = null, branch = null, maxRequestBytes }) {
  const provider = frontierProvider(config);
  const restored = await restoreCheckpoint(archive, output, {
    sha256: checkpoint.workspace.sha256, sourceCaseDir, prepared,
  });
  await validateRestoredState(restored, base.source, checkpoint, false);
  const manifest = base.source.manifest, scratch = join(restored.destination, "tool-state");
  const tools = sandboxTools(restored.workspace, scratch, manifest.toolTimeoutSeconds,
    restored.prepared.pythonEnvironment.path, restored.prepared.toolEnvironment);
  const directory = join(output, "frontier");
  mkdirSync(directory, { mode: 0o700 });
  writeJson(join(directory, "provider.json"), config);
  writeJson(join(directory, "prepared.json"), restored.prepared);
  if (branch) writeJson(join(directory, "branch.json"), branch);
  const history = structuredClone(prefix);
  const boundaries = [];
  let requests = 0, resolvedModel = priorModel;
  const signal = AbortSignal.timeout(manifest.timeoutSeconds * 1000);
  let status = "failed", error = null;
  try {
    for (const [index, prompts] of turns.entries()) {
      history.push(...prompts.map(provider.user));
      let turnHasReasoning = false, finalText;
      while (finalText === undefined) {
        signal.throwIfAborted();
        const body = JSON.stringify(provider.request(history, base.system, base.tools, manifest.maxOutputTokens));
        if (requests >= manifest.maxModelCalls) throw new Error("Frontier model request count limit reached");
        const requestBytes = Buffer.byteLength(body);
        if (requestBytes > maxRequestBytes) throw new Error(`Frontier request size ${requestBytes} exceeds ${maxRequestBytes} bytes`);
        const stem = join(directory, String(++requests).padStart(3, "0"));
        writeFileSync(`${stem}.request.json`, body + "\n");
        const response = await fetch(provider.url, {
          method: "POST", headers: { "content-type": "application/json", ...provider.headers(apiKey) }, body, signal,
        });
        const raw = await response.text();
        writeFileSync(`${stem}.response.json`, raw);
        if (!response.ok) throw new Error(`Frontier API HTTP ${response.status}`);
        const data = JSON.parse(raw), result = provider.decode(data);
        resolvedModel ??= data.model;
        if (data.model !== resolvedModel) throw new Error("Provider changed the resolved model");
        history.push(...result.saved);
        turnHasReasoning ||= result.signed;
        if (!result.calls.length) {
          finalText = result.text;
          continue;
        }
        const results = [];
        for (const call of result.calls) {
          const tool = tools.find(tool => tool.name === call.name);
          try {
            if (!tool) throw new Error("Unknown sandbox tool");
            const args = validateToolArguments(tool, { name: call.name,
              arguments: typeof call.args === "string" ? JSON.parse(call.args) : call.args });
            const result = await tool.execute(call.id, args, signal);
            if (result.content.some(block => block.type !== "text")) throw new Error("Expected text tool results");
            results.push({ id: call.id, text: result.content.map(block => block.text).join("\n"), error: Boolean(result.isError) });
          } catch (failure) {
            signal.throwIfAborted();
            results.push({ id: call.id, text: failure.message, error: true });
          }
        }
        history.push(...provider.results(results));
      }
      const verdicts = checks ? await checkOutcome(checks[index], finalText, restored.workspace) : null;
      const boundary = { prompts, historyLength: history.length, historySha256: sha256(JSON.stringify(history)), verdicts,
        visibleAnswer: finalText, opaqueReasoning: turnHasReasoning };
      boundaries.push(boundary);
      if (!finalText.trim() || !turnHasReasoning || verdicts?.some(result => !result.passed)) {
        status = checks ? "rejected" : "incomplete";
        break;
      }
      const git = (...args) => sandboxGit(restored.workspace, scratch, ...args);
      const gitState = {
        head: (await git("rev-parse", "HEAD")).trim(),
        status: await git("status", "--porcelain=v1", "--untracked-files=all"),
      };
      boundary.workspace = await snapshotWorkspace(restored.destination, join(directory, `turn-${index + 1}`), gitState);
    }
    if (boundaries.length === turns.length && boundaries.at(-1)?.workspace) status = checks ? "accepted" : "completed";
  } catch (failure) {
    error = failure.message;
  } finally {
    writeJson(join(directory, "history.json"), history);
    writeJson(join(directory, "outcome.json"), { status, requests, error, sourceSha256: base.sourceSha256,
      config, maxRequestBytes, resolvedModel, boundaries, historySha256: sha256(JSON.stringify(history)) });
  }
  return { status, requests, error };
}

export function branchTurns(plan, sourcePrompts) {
  const historical = plan.metadata.historicalPlacements ?? [];
  const prefix = historical.length ? plan.prefix.slice(0, -1) : plan.prefix;
  const unchanged = prefix.filter(message => message.role === "user").length;
  if (unchanged > sourcePrompts.length) throw new Error("Plan exceeds source user-turn count");
  const turns = plan.turns.map(turn => [turn.prompt]);
  if (historical.length) turns[0].unshift(plan.metadata.authoringInputs[0].content);
  return { unchanged, turns };
}

function below(root, path) {
  const rel = relative(realpathSync(root), path);
  return rel && !rel.split(sep).includes("..") && !rel.startsWith(sep);
}

function initialCheckpoint(base) {
  const checkpoint = base.source.checkpoints["0"];
  const archive = realpathSync(resolve(base.root, checkpoint.workspace.path));
  if (!below(base.root, archive) || !lstatSync(archive).isFile()) throw new Error("Invalid initial checkpoint archive");
  return { checkpoint, archive, sourceCaseDir: base.source.sourceCaseDir, prepared: base.source.prepared };
}

async function collect({ base, config, apiKey, output, checks, attempts, initial, maxRequestBytes }) {
  const summary = { sourceSha256: base.sourceSha256, config, checks, maxRequestBytes, attempts: [], accepted: null };
  mkdirSync(output, { mode: 0o700 });
  for (let index = 1; index <= attempts; index++) {
    const name = `attempt-${index}`;
    const result = await execute({ base, config, apiKey, output: join(output, name), ...initial,
      turns: base.prompts.map(prompt => [prompt]), checks, maxRequestBytes });
    summary.attempts.push({ name, ...result });
    if (result.status === "accepted") summary.accepted = name;
    writeJson(join(output, "collection.json"), summary);
    if (summary.accepted) return summary;
    if (result.status === "failed") throw new Error(`${name}: ${result.error}`);
  }
  throw new Error(`No matching transcript in ${attempts} attempts`);
}

function prepareBranch({ base, config, collection, checked, initial }) {
  const accepted = readJson(join(collection, "collection.json"));
  if (!/^attempt-[1-9][0-9]*$/.test(accepted.accepted ?? "") ||
      accepted.sourceSha256 !== base.sourceSha256 || !isDeepStrictEqual(accepted.config, config)) throw new Error("Baseline source/provider configuration mismatch");
  const caseDir = join(collection, accepted.accepted), directory = join(caseDir, "frontier");
  const outcome = readJson(join(directory, "outcome.json"));
  const history = readJson(join(directory, "history.json"));
  if (outcome.status !== "accepted" || outcome.sourceSha256 !== base.sourceSha256 ||
      !isDeepStrictEqual(outcome.config, config) || sha256(JSON.stringify(history)) !== outcome.historySha256 ||
      !isDeepStrictEqual(outcome.boundaries.map(item => item.prompts), base.prompts.map(prompt => [prompt]))) {
    throw new Error("Invalid accepted frontier history");
  }
  const { unchanged, turns } = branchTurns(checked.plan, base.prompts);
  let prefix = [], { checkpoint, archive, sourceCaseDir, prepared } = initial;
  if (unchanged) {
    const boundary = outcome.boundaries[unchanged - 1];
    prefix = history.slice(0, boundary.historyLength);
    if (sha256(JSON.stringify(prefix)) !== boundary.historySha256) throw new Error("Frontier checkpoint prefix hash mismatch");
    checkpoint = { workspace: boundary.workspace, gitState: boundary.workspace.gitState };
    archive = join(directory, `turn-${unchanged}.workspace.tar.gz`);
    sourceCaseDir = realpathSync(caseDir);
    prepared = readJson(join(directory, "prepared.json"));
  }
  const branch = { collection: realpathSync(collection), attempt: accepted.accepted,
    unchanged, planSha256: checked.planSha256, plan: checked.plan, qa: checked.qa,
    prefixSha256: sha256(JSON.stringify(prefix)), workspaceSha256: checkpoint.workspace.sha256 };
  return { checkpoint, archive, sourceCaseDir, prepared, prefix, priorModel: outcome.resolvedModel, turns, branch };
}

export async function main(args) {
  process.umask(0o077);
  const { positionals, values } = parseArgs({ args, allowPositionals: true, options: {
    source: { type: "string" }, checks: { type: "string" }, attempts: { type: "string" },
    provider: { type: "string" }, model: { type: "string" }, effort: { type: "string" },
    out: { type: "string" }, collection: { type: "string" }, plan: { type: "string" },
    "max-request-bytes": { type: "string" }, "qa-receipt": { type: "string" }, "allow-paid": { type: "boolean" }, "preflight-only": { type: "boolean" },
  } });
  const [mode] = positionals;
  if (positionals.length !== 1 || !["collect", "branch"].includes(mode) || !values.source ||
      (mode === "collect" ? !values.checks || !/^[1-9][0-9]*$/.test(values.attempts ?? "") ||
        !Number.isSafeInteger(Number(values.attempts)) : !values.collection || !values.plan || !values["qa-receipt"])) {
    throw new Error("Use collect --source --checks --attempts, or branch --source --collection --plan --qa-receipt; both require --provider --model --effort");
  }
  const config = { provider: values.provider, model: values.model, effort: values.effort };
  frontierProvider(config);
  const base = loadSource(realpathSync(values.source));
  const cap = values["max-request-bytes"];
  const maxRequestBytes = cap === undefined ? base.source.manifest.maxRequestBytes : Number(cap);
  if (cap !== undefined && (!/^[1-9][0-9]*$/.test(cap) || !Number.isSafeInteger(maxRequestBytes))) {
    throw new Error("--max-request-bytes requires a positive byte count");
  }
  const initial = initialCheckpoint(base);
  let checks, preparedBranch;
  if (mode === "collect") {
    checks = readJson(values.checks);
    validateChecks(checks, base.prompts.length);
  } else {
    const checked = preflightPlan(values.plan, values["qa-receipt"]);
    if (checked.qa.fixture || checked.plan.sourceDescriptor.sha256 !== base.sourceSha256) throw new Error("Plan and QA must match the live baseline source");
    preparedBranch = prepareBranch({ base, config, collection: values.collection, checked, initial });
  }
  if (values["preflight-only"]) {
    console.log(JSON.stringify({ valid: true, sourceSha256: base.sourceSha256, config, maxRequestBytes, inferenceRequests: 0 }));
    return;
  }
  if (!values["allow-paid"] || !values.out) throw new Error("Live frontier execution requires --out and --allow-paid");
  const root = join(projectDir, "out");
  const output = resolve(values.out);
  if (!output.startsWith(root + sep)) throw new Error("Output must be below this checkout's out/");
  mkdirSync(root, { recursive: true, mode: 0o700 });
  mkdirSync(dirname(output), { recursive: true, mode: 0o700 });
  if (dirname(output) !== root && !below(root, realpathSync(dirname(output)))) throw new Error("Output parent escapes out/");
  const apiKey = process.env[config.provider === "anthropic" ? "ANTHROPIC_API_KEY" : "OPENAI_API_KEY"];
  if (!apiKey) throw new Error("Missing native provider API key");
  const common = { base, config, apiKey, output, maxRequestBytes };
  const result = mode === "collect"
    ? await collect({ ...common, attempts: Number(values.attempts), checks, initial })
    : await execute({ ...common, ...preparedBranch });
  console.log(JSON.stringify(result));
  if (mode === "branch" && result.status !== "completed") process.exitCode = 1;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main(process.argv.slice(2)).catch(error => { console.error(error.message); process.exitCode = 1; });
}
