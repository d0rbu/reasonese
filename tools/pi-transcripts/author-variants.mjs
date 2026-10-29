// Full-user-turn authoring, separate from assistant execution and message QA.
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const MODEL = "moonshotai/kimi-k3.1-pro";
export const FRAMINGS = ["normal", "casual", "persuasive"];
export const hash = (value) => createHash("sha256").update(value).digest("hex");
const read = (path) => JSON.parse(readFileSync(path, "utf8"));
const write = (path, value) => {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, JSON.stringify(value, null, 2) + "\n");
};

export function tasksFor(config) {
  if (config.model !== MODEL || config.thinking !== "high" ||
      JSON.stringify(Object.keys(config.framings)) !== JSON.stringify(FRAMINGS)) {
    throw new Error("Expected Kimi high reasoning and the three selected framings");
  }
  if (!Number.isInteger(config.maxOutputTokens) || config.maxOutputTokens < 1 || config.maxOutputTokens > 8192 ||
      !Number.isInteger(config.maxRequests) || config.maxRequests < 1 ||
      !Number.isInteger(config.concurrency) || config.concurrency < 1 || config.concurrency > 3 ||
      !Number.isFinite(config.requestTimeoutSeconds) || config.requestTimeoutSeconds <= 0 || config.requestTimeoutSeconds > 180) {
    throw new Error("Invalid authoring workload bounds");
  }
  const tasks = [];
  const ids = new Set();
  for (const scenario of config.scenarios) {
    if (!/^[a-z0-9-]+$/.test(scenario.id) || ids.has(scenario.id)) throw new Error("Invalid/duplicate scenario ID");
    ids.add(scenario.id);
    for (const [relative, source] of [["investigation", scenario.investigation],
      ["implementation/a", scenario.implementation.a], ["implementation/b", scenario.implementation.b]]) {
      if (typeof source.source !== "string" || !source.source.trim() ||
          typeof source.prompt !== "string" || !source.prompt.trim()) throw new Error("Source turn text is required");
      if (source.canonicalInstruction && !source.prompt.includes(source.canonicalInstruction)) {
        throw new Error("Adapted source must retain its selected canonical requirement");
      }
      for (const framing of FRAMINGS) tasks.push({
        id: `${scenario.id}/${relative}/${framing}`,
        directory: `${scenario.id}/${relative}`, framing, source,
        sourceReference: scenario.source, guidance: config.framings[framing],
      });
    }
  }
  if (!tasks.length || tasks.length > config.maxRequests || config.maxRequests > 108) throw new Error("Author request bound exceeded");
  return tasks;
}

export function authorRequest(task, config) {
  return {
    model: MODEL, reasoning: { effort: "high", exclude: false },
    max_tokens: config.maxOutputTokens, provider: { require_parameters: true },
    messages: [{ role: "system", content:
      "Rewrite the supplied complete user turn for a coding assistant. " + task.guidance + "\n\n" +
      "Preserve the task, scope, all concrete constraints and success criteria, exact code identifiers, " +
      "string literals and values. Preserve the investigation-before-editing boundary where present. " +
      "This is a complete user message, not only a short base requirement. " +
      "Do not answer the request, solve the task, add new requirements, refer to variants or " +
      "evaluation, claim unsupported facts, or add style labels. Reply only with the rewritten user turn." },
    { role: "user", content: task.source.prompt }],
  };
}

export function scaffold(config, output) {
  const tasks = tasksFor(config);
  for (const task of tasks) {
    const directory = join(output, "variants", task.directory);
    mkdirSync(join(directory, "human"), { recursive: true });
    mkdirSync(join(directory, "kimi"), { recursive: true });
    for (const [name, text] of [["source.txt", task.source.source], ["authoring-source.txt", task.source.prompt],
      ["canonical-instruction.txt", task.source.canonicalInstruction ?? ""]]) {
      const path = join(directory, name);
      if (existsSync(path) && readFileSync(path, "utf8") !== text + "\n") throw new Error(`Changed existing source: ${path}`);
      writeFileSync(path, text + "\n");
    }
    write(join(directory, "source-provenance.json"), {
      ...task.sourceReference, sourceTurnSha256: hash(task.source.source),
      authoringSourceSha256: hash(task.source.prompt),
      sourceCheckpointSha256: task.source.sourceCheckpointSha256,
      adaptation: task.source.adaptation, authoringSourceKind: "operator_adaptation",
    });
    const manual = join(directory, "human", `${task.framing}.txt`);
    if (!existsSync(manual)) writeFileSync(manual,
      `TODO: Write the complete ${task.framing} user turn from ../authoring-source.txt.\n`);
  }
  return tasks;
}

export function checkVariants(config, output) {
  const tasks = tasksFor(config);
  return tasks.map((task) => {
    const paths = Object.fromEntries(["human", "kimi"].map((author) => {
      const path = join(output, "variants", task.directory, author, `${task.framing}.txt`);
      const text = existsSync(path) ? readFileSync(path, "utf8").trim() : "";
      return [author, { path, status: !text ? "missing" : /^TODO:/i.test(text) ? "placeholder" : "written", sha256: text ? hash(text) : null }];
    }));
    return { id: task.id, ...paths };
  });
}

export async function generate(config, output, apiKey) {
  if (!apiKey) throw new Error("OPENROUTER_API_KEY is required");
  const tasks = scaffold(config, output);
  const metadataResponse = await fetch("https://openrouter.ai/api/v1/models", { signal: AbortSignal.timeout(30_000) });
  if (!metadataResponse.ok) throw new Error(`Model catalog HTTP ${metadataResponse.status}`);
  const metadata = (await metadataResponse.json()).data.find((entry) => entry.id === MODEL);
  if (!metadata?.supported_parameters.includes("reasoning")) throw new Error("Kimi reasoning is unavailable");
  write(join(output, "model-metadata.json"), metadata);
  const planHash = hash(JSON.stringify(config));
  const manifest = join(output, "authoring-plan.json");
  if (existsSync(manifest) && read(manifest).planSha256 !== planHash) throw new Error("Changed authoring plan; choose a fresh output directory");
  write(manifest, { planSha256: planHash, model: MODEL, tasks: tasks.map(t => t.id), config });
  const records = [];
  let cursor = 0;
  async function worker() {
    while (cursor < tasks.length) {
      const task = tasks[cursor++];
      const directory = join(output, "authoring", task.id);
      const receiptPath = join(directory, "receipt.json");
      if (existsSync(directory)) {
        if (!existsSync(receiptPath)) throw new Error(`Incomplete attempt retained: ${task.id}; inspect it rather than silently retrying`);
        const previous = read(receiptPath);
        const target = join(output, "variants", task.directory, "kimi", `${task.framing}.txt`);
        if (previous.status !== "completed" || previous.planSha256 !== planHash ||
            hash(readFileSync(target)) !== previous.outputFileSha256) throw new Error(`Changed/failed attempt: ${task.id}`);
        records.push(previous); continue;
      }
      mkdirSync(directory, { recursive: true });
      const request = authorRequest(task, config), startedAt = new Date().toISOString();
      write(join(directory, "request.json"), request);
      let receipt = { id: task.id, startedAt, planSha256: planHash, status: "failed", requestedModel: MODEL };
      try {
        const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
          method: "POST", headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
          body: JSON.stringify(request), signal: AbortSignal.timeout(config.requestTimeoutSeconds * 1000),
        });
        const raw = await response.text();
        writeFileSync(join(directory, "response.body"), raw);
        write(join(directory, "http.json"), { status: response.status,
          headers: Object.fromEntries([...response.headers].filter(([name]) => ["content-type", "date", "x-request-id"].includes(name))) });
        if (!response.ok) throw new Error(`OpenRouter HTTP ${response.status}`);
        const payload = JSON.parse(raw), choice = payload.choices?.[0];
        if (payload.error || choice?.error || choice?.finish_reason !== "stop" || typeof choice.message?.content !== "string" || !choice.message.content.trim()) {
          throw new Error("Author response did not complete with usable text");
        }
        const text = choice.message.content.trim();
        const target = join(output, "variants", task.directory, "kimi", `${task.framing}.txt`);
        writeFileSync(target, text + "\n");
        receipt = { ...receipt, status: "completed", reportedModel: payload.model, provider: payload.provider,
          usage: payload.usage, responseId: payload.id,
          outputFileSha256: hash(readFileSync(target)), outputTextSha256: hash(text),
          sourceTurnSha256: hash(task.source.source), authoringSourceSha256: hash(task.source.prompt),
          semanticQA: "not independently judged", textTransform: "trim surrounding whitespace only" };
      } catch (error) { receipt.error = String(error.message ?? error); }
      receipt.finishedAt = new Date().toISOString();
      write(receiptPath, receipt); records.push(receipt);
      console.log(JSON.stringify({ id: task.id, status: receipt.status, error: receipt.error }));
    }
  }
  await Promise.all(Array.from({ length: Math.min(3, config.concurrency) }, worker));
  const summary = { model: MODEL, planSha256: planHash, requests: records.length,
    completed: records.filter(r => r.status === "completed").length,
    failures: records.filter(r => r.status !== "completed"),
    providerUsageCost: records.reduce((sum, r) => sum + (r.usage?.cost ?? 0), 0),
    records: records.sort((a, b) => a.id.localeCompare(b.id)),
    semanticQA: "No independent model judge; outputs require task/style review before experiments" };
  write(join(output, "authoring-summary.json"), summary);
  write(join(output, "variant-status.json"), checkVariants(config, output));
  return summary;
}

async function main() {
  const [command, source, destination, paid] = process.argv.slice(2);
  if (!["scaffold", "generate", "check"].includes(command) || !source || !destination) {
    throw new Error("Usage: node author-variants.mjs scaffold|generate|check sources.json output [--allow-paid]");
  }
  const config = read(source), output = resolve(destination);
  if (command === "scaffold") console.log(JSON.stringify({ tasks: scaffold(config, output).length }));
  else if (command === "check") console.log(JSON.stringify(checkVariants(config, output), null, 2));
  else {
    if (paid !== "--allow-paid") throw new Error("Explicit --allow-paid is required before author inference");
    const summary = await generate(config, output, process.env.OPENROUTER_API_KEY);
    console.log(JSON.stringify({ requests: summary.requests, completed: summary.completed, failures: summary.failures.length }));
    if (summary.failures.length) process.exitCode = 1;
  }
}
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(error => { console.error(error.message); process.exitCode = 1; });
}
