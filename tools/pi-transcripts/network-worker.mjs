// Controlled namespace setup runs before the inner, unprivileged tool process.
import { execFileSync, spawn } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync, writeSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { encodeBwrapArgs } from "./bwrap-args.mjs";

writeSync(3, "ready\n");
let input = "";
for await (const chunk of process.stdin) input += chunk;
const { args, request, blockedAddresses } = JSON.parse(input);
const privateRanges = ["0.0.0.0/8", "10.0.0.0/8", "100.64.0.0/10", "169.254.0.0/16",
  "172.16.0.0/12", "192.0.0.0/24", "192.0.2.0/24", "192.168.0.0/16", "198.18.0.0/15",
  "168.63.129.16/32", "198.51.100.0/24", "203.0.113.0/24", "224.0.0.0/4", "240.0.0.0/4", ...blockedAddresses];
const rules = `table inet development {
 chain output { type filter hook output priority 0; policy accept;
  oifname "lo" accept
  ip daddr 10.0.2.3 udp dport 53 accept
  ip daddr 10.0.2.3 tcp dport 53 accept
  ip daddr { ${privateRanges.join(", ")} } reject
  meta nfproto ipv6 reject
 }
}\n`;
// Ubuntu nftables requires a regular rules file; Node stdin uses a socketpair.
const rulesDir = mkdtempSync(join(tmpdir(), "development-network-"));
try {
  const file = join(rulesDir, "rules.nft");
  writeFileSync(file, rules, { mode: 0o600 });
  execFileSync("/usr/sbin/nft", ["-f", file]);
} finally { rmSync(rulesDir, { recursive: true, force: true }); }
// The network namespace belongs to the outer user namespace. The inner user
// namespace cannot remove these rules or enter the host network namespace.
const child = spawn("bwrap", ["--args", "3", ...args.slice(-2)], {
  stdio: ["pipe", "inherit", "inherit", "pipe"], env: { PATH: "/usr/bin:/bin" },
});
// Bubblewrap consumes and closes this private setup descriptor before tools run.
// Its normal PID1 reaper remains enabled, with no mount vector in /proc/1/cmdline.
child.stdio[3].on("error", error => process.stderr.write(error.message));
child.stdio[3].end(encodeBwrapArgs([...args.slice(0, 1), "--share-net", ...args.slice(1, -2)]));
child.stdin.end(JSON.stringify(request));
child.on("exit", code => process.exit(code ?? 1));
child.on("error", error => { process.stderr.write(error.message); process.exit(1); });
