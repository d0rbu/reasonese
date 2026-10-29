import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { existsSync, mkdtempSync, mkdirSync, readFileSync, readdirSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";
import { downloadVerifiedFile, prepareUv, UV_VERSION } from "../environments.mjs";

const hash = bytes => createHash("sha256").update(bytes).digest("hex");
function fixture(t) {
  const root = mkdtempSync(join(tmpdir(), "pi-uv-cache-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  return root;
}

test("verified dependency cache publishes complete bytes and reuses them without network", async t => {
  const root = fixture(t), path = join(root, "runtime/wheel.whl"), bytes = Buffer.from("Offline wheel fixture");
  let downloads = 0;
  const fetcher = async () => { downloads++; return new Response(bytes); };
  await Promise.all([1, 2, 3].map(() => downloadVerifiedFile(path, "https://example.invalid/wheel.whl", hash(bytes), fetcher)));
  assert.deepEqual(readFileSync(path), bytes);
  assert.equal(downloads, 3);
  await downloadVerifiedFile(path, "https://example.invalid/wheel.whl", hash(bytes), () => assert.fail("Valid cache used network"));
  assert.deepEqual(readdirSync(dirname(path)), ["wheel.whl"]);
});

test("corrupt downloads, caches and symlink substitutions fail without replacement", async t => {
  const root = fixture(t), bytes = Buffer.from("Expected pinned wheel"), digest = hash(bytes);
  const path = join(root, "runtime/wheel.whl");
  await assert.rejects(downloadVerifiedFile(path, "https://example.invalid/", digest,
    async () => new Response("Wrong download")), /checksum mismatch/);
  assert.equal(existsSync(path), false);
  mkdirSync(dirname(path), { recursive: true }); writeFileSync(path, "Retain corrupt cache evidence");
  await assert.rejects(downloadVerifiedFile(path, "https://example.invalid/", digest,
    () => assert.fail("Corrupt cache must fail before download")), /checksum mismatch/);
  assert.equal(readFileSync(path, "utf8"), "Retain corrupt cache evidence");
  const target = join(root, "valid-target"); writeFileSync(target, bytes);
  const link = join(root, "symlink.whl"); symlinkSync(target, link);
  await assert.rejects(downloadVerifiedFile(link, "https://example.invalid/", digest,
    () => assert.fail("Symlink cache must fail before download")), /checksum mismatch/);
  assert.deepEqual(readFileSync(target), bytes);
});

test("pinned uv rejects a changed cached executable before fetching or executing it", async t => {
  const root = fixture(t), binary = join(root, `uv-${UV_VERSION}/bin/uv`);
  mkdirSync(dirname(binary), { recursive: true });
  writeFileSync(binary, "#!/bin/sh\nexit 0\n", { mode: 0o755 });
  await assert.rejects(prepareUv({ cacheRoot: root, fetcher: () => assert.fail("Changed binary must fail before network") }), /Cached uv binary checksum mismatch/);
  assert.equal(readFileSync(binary, "utf8"), "#!/bin/sh\nexit 0\n");
});
