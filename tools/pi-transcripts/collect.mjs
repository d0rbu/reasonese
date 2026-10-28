#!/usr/bin/env node
import { existsSync } from "node:fs";
import { join, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { createInterface } from "node:readline";
import { projectDir } from "./sandbox.mjs";
import { modelMetadata, runTrajectory, verifyCompleted } from "./trajectory.mjs";
import { collectorIdentity, loadManifest, prepare, readJson, writeJson } from "./workspaces.mjs";

export async function main(args) {
  const { values, positionals } = parseArgs({ args, allowPositionals: true, options: {
    manifest: { type: "string", default: join(projectDir, "configs/pi_transcripts.json") },
    output: { type: "string" }, "allow-paid": { type: "boolean", default: false },
    interactive: { type: "boolean", default: false },
  } });
  if (positionals.length !== 1 || !["prepare", "collect"].includes(positionals[0]) || !values.output) {
    throw new Error("Usage: node tools/pi-transcripts/collect.mjs prepare|collect --output out/<run> [--allow-paid]");
  }
  const output = resolve(values.output);
  if (!output.startsWith(join(projectDir, "out") + sep)) throw new Error("Output must be below this checkout's out/");
  const manifest = loadManifest(values.manifest);
  if (positionals[0] === "prepare") {
    await prepare(manifest, output);
    writeJson(join(output, "model.json"), await modelMetadata());
    return;
  }
  const plan = readJson(join(output, "plan.json"));
  if (JSON.stringify(plan.manifest) !== JSON.stringify(manifest) ||
    JSON.stringify(plan.collector) !== JSON.stringify(collectorIdentity())) {
    throw new Error("Prepared plan or collector changed; prepare a fresh run");
  }
  // Gate before metadata lookup or session creation; a provider default never grants spending.
  if (!values["allow-paid"]) throw new Error("--allow-paid is required to collect Pi transcripts");
  if (values.interactive && manifest.scenarios.length !== 1) throw new Error("Interactive collection requires a single-scenario manifest");
  if (manifest.scenarios.some((scenario) => (scenario.maxUserTurns ?? 1) > 1) && !values.interactive) throw new Error("Use --interactive for multi-turn scenarios");
  if (!process.env.OPENROUTER_API_KEY) throw new Error("OPENROUTER_API_KEY is required");
  const metadata = await modelMetadata();
  for (const scenario of manifest.scenarios) {
    const caseDir = join(output, scenario.id);
    if (!existsSync(join(caseDir, "prepared.json"))) throw new Error(`Unprepared scenario: ${scenario.id}`);
    const outcomeFile = join(caseDir, "trajectory/outcome.json");
    if (existsSync(outcomeFile)) {
      verifyCompleted(caseDir, manifest, scenario);
      console.log(`Already completed: ${scenario.id}`);
      continue;
    }
    console.log(`Collecting ${scenario.id}`);
    const input = values.interactive ? createInterface({ input: process.stdin, crlfDelay: Infinity, terminal: false }) : null;
    const lines = input?.[Symbol.asyncIterator]();
    let outcome;
    try {
      outcome = await runTrajectory({ manifest, scenario, caseDir, metadata, apiKey: process.env.OPENROUTER_API_KEY,
        nextTurn: input ? async ({ turn, messages, directory }) => {
          console.log(JSON.stringify({ awaitingOperator: true, completedUserTurn: turn, directory,
            lastAssistant: messages.at(-1) }));
          console.log('Send one JSON line {"prompt":"...","rationale":"..."}, or null to finish.');
          const line = await lines.next();
          if (line.done) throw new Error("Operator input closed before explicit finish");
          return JSON.parse(line.value);
        } : null });
    } finally { input?.close(); }
    console.log(`${scenario.id}: ${outcome.status} (${outcome.modelRequests} requests, ${outcome.toolResults} tool results)`);
    if (outcome.status !== "completed") throw new Error(`Collection stopped at ${scenario.id}: ${outcome.status}`);
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main(process.argv.slice(2)).catch((error) => { console.error(error.message); process.exitCode = 1; });
}
