// TypeScript syntax transpilation only; the actual env implementation and filesystem run unchanged.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const ts = require(path.join(process.env.PI_NODE_MODULES, "typescript"));
require.extensions[".ts"] = (module, filename) => {
  const { outputText } = ts.transpileModule(fs.readFileSync(filename, "utf8"), {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, esModuleInterop: true },
  });
  module._compile(outputText, filename);
};
const { parseEnvFiles, loadEnvFiles } = require(path.resolve("packages/@expo/env/src/index.ts"));
const root = fs.mkdtempSync("/tmp/expo-baseline-");
try {
  const file = path.join(root, ".env");
  fs.writeFileSync(file, "API_URL=https://local.invalid\n");
  const systemEnv = { API_URL: "https://ci.invalid" };
  const parsed = parseEnvFiles([file], { systemEnv });
  assert.equal(parsed.env.API_URL, "https://local.invalid");
  assert.equal(systemEnv.API_URL, "https://ci.invalid");
  loadEnvFiles([file], { systemEnv });
  assert.equal(systemEnv.API_URL, "https://ci.invalid");
  console.log(JSON.stringify({ kind: "pinned_source_baseline", parsed: parsed.env.API_URL, loaded: systemEnv.API_URL }));
} finally {
  fs.rmSync(root, { recursive: true });
}
