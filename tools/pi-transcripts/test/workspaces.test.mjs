import assert from "node:assert/strict";
import { existsSync, mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import test from "node:test";
import { completeCheckoutHistory, fetchSource, git } from "../workspaces.mjs";

async function historyFixture(t) {
  const root = mkdtempSync(join(tmpdir(), "pinned-source-history-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const upstream = join(root, "upstream", "project.git");
  mkdirSync(upstream, { recursive: true });
  await git(upstream, "init", "-b", "main");
  await git(upstream, "config", "user.name", "Developer");
  await git(upstream, "config", "user.email", "developer@localhost");
  const commit = async (filename, content) => {
    writeFileSync(join(upstream, filename), content);
    await git(upstream, "add", filename);
    await git(upstream, "commit", "-m", content);
    return (await git(upstream, "rev-parse", "HEAD")).trim();
  };
  const first = await commit("base.txt", "base");
  await commit("main.txt", "main ancestor");
  await git(upstream, "checkout", "-b", "side", first);
  await commit("side.txt", "side ancestor");
  await git(upstream, "checkout", "main");
  await git(upstream, "merge", "--no-ff", "side", "-m", "pinned merge");
  const revision = (await git(upstream, "rev-parse", "HEAD")).trim();
  const tree = (await git(upstream, "rev-parse", "HEAD^{tree}")).trim();
  const ancestors = (await git(upstream, "rev-list", revision)).trim().split("\n").sort();
  assert.equal(ancestors.length, 4);
  const descendant = await commit("later.txt", "later descendant");
  await git(upstream, "tag", "later-release", descendant);
  const scenario = { repository: pathToFileURL(upstream).href, revision };
  const sources = join(root, "sources");
  const cache = join(sources, scenario.repository.split("/").slice(-2).join("-") + "-" + revision);
  return { scenario, sources, cache, tree, ancestors, descendant };
}

async function assertPinnedHistory(fixture, cache) {
  assert.equal(cache, fixture.cache);
  assert.equal((await git(cache, "rev-parse", "--is-shallow-repository")).trim(), "false");
  assert.equal((await git(cache, "rev-parse", "HEAD")).trim(), fixture.scenario.revision);
  assert.equal((await git(cache, "rev-parse", "FETCH_HEAD")).trim(), fixture.scenario.revision);
  assert.equal((await git(cache, "rev-parse", "HEAD^{tree}")).trim(), fixture.tree);
  assert.deepEqual((await git(cache, "rev-list", "--all")).trim().split("\n").sort(), fixture.ancestors);
  assert.equal((await git(cache, "for-each-ref", "--format=%(refname) %(objectname)")).trim(),
    `refs/heads/source ${fixture.scenario.revision}`);
  assert.equal((await git(cache, "symbolic-ref", "HEAD")).trim(), "refs/heads/source");
  await assert.rejects(git(cache, "cat-file", "-e", `${fixture.descendant}^{commit}`));
}

test("new pinned source cache includes every merge ancestor without later refs or commits", async (t) => {
  const fixture = await historyFixture(t);
  await assertPinnedHistory(fixture, await fetchSource(fixture.scenario, fixture.sources));
  // Reusing the exact pin must not follow the upstream's newer branch or tag.
  await assertPinnedHistory(fixture, await fetchSource(fixture.scenario, fixture.sources));
});

test("cached shallow source at matching FETCH_HEAD is unshallowed at the same pinned tree", async (t) => {
  const fixture = await historyFixture(t);
  mkdirSync(fixture.cache, { recursive: true });
  await git(fixture.cache, "init", "--bare");
  await git(fixture.cache, "remote", "add", "origin", fixture.scenario.repository);
  await git(fixture.cache, "fetch", "--depth=1", "--no-tags", "origin", fixture.scenario.revision);
  await git(fixture.cache, "update-ref", "refs/heads/source", fixture.scenario.revision);
  await git(fixture.cache, "symbolic-ref", "HEAD", "refs/heads/source");
  assert.equal((await git(fixture.cache, "rev-parse", "FETCH_HEAD")).trim(), fixture.scenario.revision);
  assert.equal((await git(fixture.cache, "rev-list", "--count", "HEAD")).trim(), "1");
  assert.equal((await git(fixture.cache, "rev-parse", "--is-shallow-repository")).trim(), "true");
  await assertPinnedHistory(fixture, await fetchSource(fixture.scenario, fixture.sources));
});

test("unstarted shallow checkout gains pinned ancestors without changing refs or exposing cache in FETCH_HEAD", async (t) => {
  const fixture = await historyFixture(t);
  mkdirSync(fixture.cache, { recursive: true });
  await git(fixture.cache, "init", "--bare");
  await git(fixture.cache, "remote", "add", "origin", fixture.scenario.repository);
  await git(fixture.cache, "fetch", "--depth=1", "--no-tags", "origin", fixture.scenario.revision);
  await git(fixture.cache, "update-ref", "refs/heads/source", fixture.scenario.revision);
  await git(fixture.cache, "symbolic-ref", "HEAD", "refs/heads/source");
  const cwd = join(fixture.sources, "unstarted-checkout");
  await git(fixture.sources, "clone", "--no-hardlinks", "--no-checkout", fixture.cache, cwd);
  await git(cwd, "checkout", "--detach", fixture.scenario.revision);
  await git(cwd, "remote", "set-url", "origin", fixture.scenario.repository);
  const refs = await git(cwd, "show-ref");
  assert.equal((await git(cwd, "rev-parse", "--is-shallow-repository")).trim(), "true");
  assert.equal(existsSync(join(cwd, ".git/FETCH_HEAD")), false);
  await fetchSource(fixture.scenario, fixture.sources);
  await completeCheckoutHistory(cwd, fixture.cache, fixture.scenario.revision);
  assert.equal((await git(cwd, "rev-parse", "--is-shallow-repository")).trim(), "false");
  assert.equal((await git(cwd, "rev-parse", "HEAD")).trim(), fixture.scenario.revision);
  assert.equal((await git(cwd, "rev-parse", "HEAD^{tree}")).trim(), fixture.tree);
  assert.deepEqual((await git(cwd, "rev-list", "HEAD")).trim().split("\n").sort(), fixture.ancestors);
  assert.equal(await git(cwd, "show-ref"), refs);
  assert.equal((await git(cwd, "remote", "get-url", "origin")).trim(), fixture.scenario.repository);
  assert.equal((await git(cwd, "status", "--porcelain=v1")).trim(), "");
  assert.equal(existsSync(join(cwd, ".git/FETCH_HEAD")), false);
  await assert.rejects(git(cwd, "cat-file", "-e", `${fixture.descendant}^{commit}`));
});
