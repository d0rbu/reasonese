// Run with Bun from the pinned repository; import its actual TypeScript module.
import assert from "node:assert/strict";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
const options = await import(pathToFileURL(resolve("packages/bun-plugin-svelte/src/options.ts")).href);
const plugin = { forceSide: "client" }, config = { target: "node" };
const component = options.getBaseCompileOptions(plugin, config).generate;
const module = options.getBaseModuleCompileOptions(plugin, config).generate;
assert.equal(component, "client");
assert.equal(module, "client");
assert.doesNotThrow(() => options.validateOptions({ development: "yes" }));
console.log(JSON.stringify({ kind: "pinned_source_baseline", component, module, invalid_development_accepted: true }));
