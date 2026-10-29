// Continue a native Pi source at its matching task-state checkpoint.
import { createHash } from "node:crypto";
import { closeSync, existsSync, openSync, readFileSync, readSync, realpathSync } from "node:fs";
import { basename, isAbsolute, join, relative, resolve } from "node:path";
import { isDeepStrictEqual } from "node:util";
import { fileURLToPath } from "node:url";
import { neutralCwd, sandboxGit, sandboxTools } from "./sandbox.mjs";
import { restoreCheckpoint } from "./checkpoints.mjs";
import { validateNodeIdentity } from "./environments.mjs";
import { runTrajectory, modelMetadata } from "./trajectory.mjs";
import { contextFile, readJson, sha256, writeJson } from "./workspaces.mjs";

const equal = isDeepStrictEqual;
function within(root, filename) {
  const path = realpathSync(resolve(root, filename));
  if (relative(realpathSync(root), path).split(/[\\/]/).includes("..")) throw new Error("Artifact escapes source directory");
  return path;
}
function artifact(root, ref) {
  const path = within(root, ref.path);
  const bytes = readFileSync(path);
  if (sha256(bytes) !== ref.sha256) throw new Error("Source artifact hash mismatch");
  return bytes;
}
export function validateHistory(messages) {
  const pending = new Map();
  for (const message of messages) {
    if (!["user", "assistant", "toolResult"].includes(message.role)) throw new Error("Unsupported source role");
    if (message.role === "assistant") {
      for (const block of message.content ?? []) if (block.type === "toolCall") {
        if (pending.has(block.id)) throw new Error("Duplicate pending tool call");
        pending.set(block.id, block.name);
      }
    } else if (message.role === "toolResult") {
      if (pending.get(message.toolCallId) !== message.toolName) throw new Error("Orphan or mismatched tool result");
      pending.delete(message.toolCallId);
    } else if (pending.size) throw new Error("User turn interrupts pending tools");
  }
  if (pending.size) throw new Error("Source prefix has unresolved tools");
}

function hashFile(path) {
  const fd = openSync(path, "r"), hash = createHash("sha256"), buffer = Buffer.alloc(1024 * 1024);
  try { let length; while ((length = readSync(fd, buffer, 0, buffer.length, null))) hash.update(buffer.subarray(0, length)); }
  finally { closeSync(fd); }
  return hash.digest("hex");
}

export function preflightPlan(planPath, receiptPath = null, { fixture = false } = {}) {
  const bytes = readFileSync(planPath), plan = JSON.parse(bytes), planSha256 = sha256(bytes);
  if (plan.kind !== "pi_experiment_plan" || plan.model !== "moonshotai/kimi-k3.1-pro" || plan.thinking !== "high") throw new Error("Unsupported native experiment plan");
  if (plan.containsPlaceholders || !plan.turns?.length || plan.turns.some(t => typeof t.prompt !== "string" || !t.prompt.trim() || t.prompt.trim().startsWith("TODO:"))) throw new Error("Placeholder or empty user turn");
  const inputs = plan.metadata?.authoringInputs;
  if (!inputs?.length || inputs.some(i => !i.content?.trim() || i.content.trim().startsWith("TODO:"))) throw new Error("No complete authoring inputs for required message QA");
  const qa = receiptPath ? readJson(receiptPath) : null;
  if (qa && (qa.kind !== "pi_message_qa_receipt" || qa.passed !== true || qa.judge !== "openai/gpt-5.6-luna:batch" || qa.planSha256 !== planSha256 || qa.inputs?.length !== inputs.length || qa.inputs.some((v, i) => v.complies !== true || !equal(v.spec, inputs[i].spec) || v.contentSha256 !== sha256(inputs[i].content)))) throw new Error("Required compliant message QA does not match exact plan");
  if (!isAbsolute(plan.corpusRoot)) throw new Error("Expected resolved corpus root");
  const root = resolve(plan.corpusRoot), source = plan.source;
  if (!fixture && source.captureKind !== "live_pi_session") throw new Error("Production preflight requires a live source capture");
  if (source.kind !== "pi_source_template") throw new Error("Expected native source descriptor");
  if (!plan.sourceDescriptor || !equal(JSON.parse(artifact(root, plan.sourceDescriptor)), source)) throw new Error("Plan source differs from pinned complete descriptor");
  const messages = JSON.parse(artifact(root, source.messages));
  if (artifact(root, source.system).toString() !== plan.system || !equal(JSON.parse(artifact(root, source.tools)), plan.tools)) throw new Error("Plan system/tools differ from source");
  const currentTools = sandboxTools(neutralCwd(), "", 1)
    .map(({ name, description, parameters }) => ({ name, description, parameters }));
  if (!equal(currentTools, plan.tools)) throw new Error("Source tool definitions are incompatible with current Pi SDK");
  if (messages.at(-1)?.role !== "assistant" || messages.at(-1)?.stopReason !== "stop") throw new Error("Source is not a completed native trajectory");
  if (source.manifest?.model !== plan.model || source.manifest?.thinking !== plan.thinking ||
      messages.some(m => m.role === "assistant" && (m.model !== plan.model || m.provider !== "openrouter" || m.api !== "openai-completions" || !["stop", "toolUse"].includes(m.stopReason)))) throw new Error("Cross-model or aborted source replay is unsupported");
  validateHistory(messages); validateHistory(plan.prefix);
  const historical = plan.metadata.historicalPlacements ?? [];
  let actualPrefix = plan.prefix;
  if (historical.length) {
    if (historical.length !== 1 || historical[0].message !== plan.prefix.length - 1) throw new Error("Invalid historical user placement");
    actualPrefix = plan.prefix.slice(0, -1);
  }
  const users = messages.flatMap((m, i) => m.role === "user" ? [i] : []);
  const boundaries = [0, ...users.slice(1), messages.length];
  const checkpointIndex = boundaries.findIndex(end => equal(messages.slice(0, end), actualPrefix));
  if (checkpointIndex < 0) throw new Error("Plan prefix differs from source boundary");
  const checkpoint = source.checkpoints[String(checkpointIndex)];
  const path = within(root, checkpoint.workspace.path);
  if (path !== resolve(plan.checkpoint.workspace.path) || checkpoint.workspace.sha256 !== plan.checkpoint.workspace.sha256) throw new Error("Plan checkpoint differs from source");
  if (checkpoint.prefixSha256 && checkpoint.prefixSha256 !== sha256(JSON.stringify(messages.slice(0, boundaries[checkpointIndex])))) throw new Error("Checkpoint history binding mismatch");
  const delivered = [...historical.map(p => plan.prefix[p.message].content.map(b => b.text).join("")), ...plan.turns.map(t => t.prompt)];
  if (!equal(delivered, inputs.map(i => i.content))) throw new Error("QA text differs from actual user turns");
  const variants = [...historical.map(p => ({ variant: plan.metadata.authoringVariants?.[p.position - 1], spec: inputs[p.position - 1].spec, content: inputs[p.position - 1].content })),
    ...plan.turns.map((t, i) => ({ variant: t.variant, spec: inputs[historical.length + i].spec, content: t.prompt }))];
  if (variants.some(item => !item.variant)) throw new Error("Every authored input needs variant provenance");
  for (const { variant, spec, content } of variants) {
    if (readFileSync(variant.path, "utf8").trim() !== content) throw new Error("Delivered text differs from authored variant file");
    if (sha256(readFileSync(variant.path)) !== variant.sha256) throw new Error("Variant file changed since plan compilation");
    const data = {};
    for (const name of ["authoringSource", "sourceProvenance", "instructionSource", "sourceBinding"]) {
      const ref = variant[name];
      if (!ref || sha256(data[name] = readFileSync(ref.path)) !== ref.sha256) throw new Error("Variant source provenance changed");
    }
    const binding = JSON.parse(data.sourceBinding), provenance = JSON.parse(data.sourceProvenance);
    if (binding.kind !== "reviewed_native_source_binding" || !binding.reviewNote?.trim() ||
        binding.target.descriptorSha256 !== plan.sourceDescriptor.sha256 ||
        binding.target.messagesSha256 !== source.messages.sha256 ||
        !equal(binding.target.checkpoints, Object.fromEntries(Object.entries(source.checkpoints).map(([key, value]) => [key, value.workspace.sha256])))) throw new Error("Variant lacks a reviewed binding to source history/state");
    const key = variant.slot + (variant.side ? "/" + variant.side : "");
    if (!equal(spec, variant.spec) || data.instructionSource.toString().trim() !== spec.instruction ||
        provenance.authoringSourceSha256 !== sha256(data.authoringSource.toString().trim()) ||
        ["authoringSource", "sourceProvenance", "instructionSource"].some(name => binding.slots[key]?.[name] !== variant[name].sha256)) throw new Error("Variant policy or authoring source differs from reviewed binding");
  }
  if (hashFile(path) !== checkpoint.workspace.sha256) throw new Error("Checkpoint archive hash mismatch");
  if (typeof source.sourceCaseDir !== "string" || !source.prepared || !source.scenarioConfig || !source.manifest) throw new Error("Source lacks environment restoration metadata");
  if (!fixture && (!source.prepared.pythonEnvironment?.path ||
      !source.prepared.pythonEnvironment.runtime?.executableSha256)) throw new Error("Native source needs a pinned task-owned Python runtime");
  return { plan, planSha256, checkpoint, archive: path, qa };
}

async function validateRestoredState(restored, source, checkpoint, fixture) {
  const cwd = join(restored.destination, "workspace"), prepared = restored.prepared;
  contextFile(cwd, source.scenarioConfig);
  const python = prepared.pythonEnvironment;
  if (python) {
    if (realpathSync(python.path) !== join(restored.destination, "python-env")) throw new Error("Restored Python environment escapes task state");
    const executable = realpathSync(join(python.path, "bin/python"));
    const runtime = python.runtime;
    if (!fixture && !runtime?.executableSha256) throw new Error("Native source needs a pinned Python runtime");
    if (runtime) {
      const root = realpathSync(runtime.path);
      if (!executable.startsWith(root + "/") || hashFile(executable) !== runtime.executableSha256) throw new Error("Prepared Python runtime changed");
      for (const library of runtime.sharedLibraries ?? []) {
        if (basename(library.file) !== library.file || hashFile(join(root, "lib", library.file)) !== library.sha256) throw new Error("Prepared Python shared library changed");
      }
    }
  } else if (!fixture) throw new Error("Native source needs a task-owned Python environment");
  const environment = prepared.toolEnvironment ?? {}, dependencies = environment.dependencies ?? {};
  const taskNode = validateNodeIdentity(environment, restored.destination);
  if (taskNode) {
    const pnpm = taskNode.packageManager?.name === "pnpm";
    const result = await sandboxTools(cwd, join(restored.destination, "tool-state"), 30, python?.path, environment)
      .find(tool => tool.name === "bash").execute("validate-node", { command: pnpm ? "node --version && pnpm --version" : "node --version" });
    const version = result.content.map(block => block.text ?? "").join("").trim();
    const expected = pnpm ? `${taskNode.version}\n${taskNode.packageManager.version}` : taskNode.version;
    if (result.isError || version !== expected) throw new Error("Prepared task Node or package manager version changed");
  }
  const immutableTools = [
    ["go", "/opt/go", source => join(source, "bin/go")],
    ["bun", "/opt/bin/bun", source => source],
    ["uv", "/opt/bin/uv", source => source],
  ];
  for (const [name, target, executable] of immutableTools) {
    const identity = dependencies[name];
    if (!identity) continue;
    const mount = environment.mounts?.find(m => m.target === target);
    if (!mount || !identity.executableSha256 || hashFile(executable(mount.source)) !== identity.executableSha256) throw new Error(`Prepared ${name} runtime changed`);
  }
  const head = (await sandboxGit(cwd, join(restored.destination, "tool-state"), "rev-parse", "HEAD")).trim();
  const status = await sandboxGit(cwd, join(restored.destination, "tool-state"), "status", "--porcelain=v1", "--untracked-files=all");
  if (checkpoint.gitState && (head !== checkpoint.gitState.head || status !== checkpoint.gitState.status)) throw new Error("Restored Git state differs from source checkpoint");
}

export async function runContinuation({ planPath, receiptPath, output, apiKey, metadata, fixtureEndpoint = null }) {
  // All local validation occurs before model lookup or provider access.
  if (!receiptPath) throw new Error("Required compliant message QA receipt missing");
  const checked = preflightPlan(planPath, receiptPath, { fixture: Boolean(fixtureEndpoint) });
  if (existsSync(output)) throw new Error("Continuation output must be fresh; failed attempts are retained");
  const { plan, checkpoint, archive } = checked;
  if (!fixtureEndpoint && (plan.source.captureKind !== "live_pi_session" || checked.qa.fixture)) throw new Error("Production continuations require live source captures and real QA receipts");
  const restored = await restoreCheckpoint(archive, output, { sha256: checkpoint.workspace.sha256,
    sourceCaseDir: plan.source.sourceCaseDir, prepared: plan.source.prepared });
  writeJson(join(output, "prepared.json"), restored.prepared);
  writeJson(join(output, "plan.json"), plan);
  writeJson(join(output, "message-qa.json"), checked.qa);
  await validateRestoredState(restored, plan.source, checkpoint, Boolean(fixtureEndpoint));
  const manifest = { ...plan.source.manifest, collectionPurpose: "native_user_turn_experiment", model: plan.model, thinking: plan.thinking };
  const scenario = { ...plan.source.scenarioConfig, prompt: plan.turns[0].prompt, maxUserTurns: plan.turns.length };
  return runTrajectory({ manifest, scenario, caseDir: output, apiKey,
    metadata: metadata ?? await modelMetadata(plan.model), fixtureEndpoint,
    continuation: { prefix: plan.prefix, system: plan.system, tools: plan.tools,
      turns: plan.turns.map(t => ({ prompt: t.prompt, rationale: "Exact authored variant selected by native plan", variant: t.variant })),
      checkpointGitState: checkpoint.gitState ?? null,
      provenance: { planSha256: checked.planSha256, source: plan.source, trial: plan.trial,
        metadata: plan.metadata, qa: checked.qa } } });
}

export async function main(argv = process.argv.slice(2)) {
  const values = {};
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === "--allow-paid") values.allowPaid = true;
    else if (argv[i] === "--preflight-only") values.preflightOnly = true;
    else if (["--plan", "--qa-receipt", "--out"].includes(argv[i]) && argv[i + 1]) values[argv[i].slice(2)] = argv[++i];
    else throw new Error(`Unknown or incomplete argument: ${argv[i]}`);
  }
  if (values.preflightOnly) {
    if (!values.plan) throw new Error("--plan is required");
    const checked = preflightPlan(values.plan);
    console.log(JSON.stringify({ valid: true, planSha256: checked.planSha256, inferenceRequests: 0 }));
    return;
  }
  if (!values.allowPaid) throw new Error("--allow-paid is required for native continuation");
  if (!values.plan || !values["qa-receipt"] || !values.out) throw new Error("Use --plan FILE --qa-receipt FILE --out FRESH_DIR --allow-paid");
  preflightPlan(values.plan, values["qa-receipt"]);
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) throw new Error("OPENROUTER_API_KEY is required");
  const result = await runContinuation({ planPath: values.plan, receiptPath: values["qa-receipt"], output: resolve(values.out), apiKey });
  console.log(JSON.stringify(result));
  if (result.status !== "completed") process.exitCode = 1;
}
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main().catch(error => { console.error(error.message); process.exitCode = 1; });
