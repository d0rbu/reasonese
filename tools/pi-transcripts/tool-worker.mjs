import { createCodingTools } from "@mariozechner/pi-coding-agent";
import { execFileSync } from "node:child_process";

let input = "";
process.stdin.setEncoding("utf8");
for await (const chunk of process.stdin) input += chunk;
try {
  const { cwd, name, id, parameters, git } = JSON.parse(input);
  let result;
  if (git) {
    result = execFileSync("git", ["-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false", ...git], {
      cwd, encoding: "utf8", maxBuffer: 32 * 1024 * 1024,
    });
  } else {
    const tool = createCodingTools(cwd).find((candidate) => candidate.name === name);
    if (!tool) throw new Error(`Unknown Pi tool: ${name}`);
    result = await tool.execute(id, parameters);
  }
  process.stdout.write(JSON.stringify({ result }));
} catch (error) {
  process.stdout.write(JSON.stringify({ error: error.message }));
}
