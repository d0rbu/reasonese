import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdtempSync, mkdirSync, readFileSync, readdirSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { corpusDownload, prose } from '../viewer/app.js';

const toolRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const digest = value => createHash('sha256').update(value).digest('hex');
const compactHash = value => digest(JSON.stringify(value));
const model = 'moonshotai/kimi-k3.1-pro';
const malicious = '<img src=x onerror="globalThis.compromised=true"><script>alert(1)</script>';
const multipart = { kind: 'multipart_archive', url: 'https://example.org/case.parts.json',
  sha256: 'a'.repeat(64), bytes: 300, archiveSha256: 'b'.repeat(64), archiveBytes: 5, partCount: 2,
  parts: [{ url: 'https://example.org/case.part-0001', sha256: 'c'.repeat(64), bytes: 3 },
    { url: 'https://example.org/case.part-0002', sha256: 'd'.repeat(64), bytes: 2 }] };
const python = `import importlib.util,json,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('export_review',sys.argv[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
args=json.loads(sys.argv[2])
print(json.dumps(module.export_review(*args,fixture=True)))`;

function fixture(t) {
  const root = mkdtempSync(join(tmpdir(), 'pi-export-fixture-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const caseDir = join(root, 'case'), source = join(caseDir, 'trajectory');
  mkdirSync(join(caseDir, 'workspace'), { recursive: true });
  const put = (name, data) => {
    const path = join(source, name); mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, typeof data === 'string' ? data : JSON.stringify(data, null, 3) + '\n');
  };
  const initial = 'Hm, can you investigate this before changing anything?';
  const requirement = 'Please repair the first behavior and add regression coverage.';
  // Adaptive operator turns may paraphrase the canonical requirement. The exact
  // delivered text must match the native capture, not a canonical substring.
  const implementation = 'Okay, nice. Can you fix that first behavior and cover it with a regression test?';
  const assistant = (index, stopReason, blocks) => ({ role: 'assistant', api: 'openai-completions',
    model, provider: 'openrouter', responseId: `gen-fixture-${index}`, content: blocks,
    usage: { totalTokens: 12, cost: { total: 0.000000003 } }, stopReason, timestamp: 1000 + index });
  const messages = [
    { role: 'user', content: [{ type: 'text', text: initial }], timestamp: 1000 },
    assistant(1, 'toolUse', [{ type: 'thinking', thinking: 'Inspect the current implementation.' },
      { type: 'toolCall', id: 'read-1', name: 'read', arguments: { path: 'example.txt' } }]),
    { role: 'toolResult', toolName: 'read', toolCallId: 'read-1', content: [{ type: 'text', text: 'old' }], timestamp: 1002 },
    assistant(2, 'stop', [{ type: 'thinking', thinking: 'The existing behavior explains the issue.' },
      { type: 'text', text: 'Here is the investigation.' }]),
    { role: 'user', content: [{ type: 'text', text: implementation }], timestamp: 1004 },
    assistant(3, 'stop', [{ type: 'thinking', thinking: 'Repair and check the behavior.' },
      { type: 'text', text: malicious }]),
  ];
  const tools = [{ name: 'read', description: 'Read a file.', parameters: { type: 'object', properties: {} } }];
  put('messages.json', messages);
  put('system.txt', 'You are a coding assistant.'); put('tools.json', tools);
  put('changes.patch', 'diff --git a/example.txt b/example.txt\n-old\n+new\n');
  put('provenance.json', { kind: 'live_pi_session', collectionPurpose: 'base_transcript', fixture: true,
    collector: { pi: '0.73.1' }, context: { content: 'Repository instructions.' },
    manifest: { model, thinking: 'high', maxOutputTokens: 8192 },
    scenario: { id: 'fixture-case', title: 'Fixture scenario', repository: 'https://github.com/example/project.git',
      revision: 'a'.repeat(40), prompt: initial, baseInstruction: requirement, maxUserTurns: 4 } });
  for (const [index, end] of [[1, 4], [2, 6]]) {
    put(`turns/00${index}.input.json`, { turn: index, prompt: index === 1 ? initial : implementation,
      basedOnMessagesSha256: compactHash(index === 1 ? [] : messages.slice(0, 4)) });
    put(`turns/00${index}.messages.json`, messages.slice(0, end));
  }
  const snapshot = 'Retained fixture workspace archive bytes.';
  put('turns/001.workspace.tar.gz', snapshot);
  put('turns/001.workspace.json', { archive: '001.workspace.tar.gz', sha256: digest(snapshot), bytes: Buffer.byteLength(snapshot) });
  put('sessions/native.jsonl', [{ type: 'session', id: 'fixture-session' },
    ...messages.map(message => ({ type: 'message', message }))].map(JSON.stringify).join('\n') + '\n');
  const nativeText = message => message.content.filter(b => b.type === 'text').map(b => b.text).join('');
  const nativeReasoning = message => message.content.filter(b => b.type === 'thinking').map(b => b.thinking).join('');
  for (const [index, end] of [[1, 1], [2, 3], [3, 5]]) {
    const previous = messages.slice(0, end).map(message => message.role === 'assistant'
      ? { role: 'assistant', content: nativeText(message), reasoning: nativeReasoning(message) }
      : { role: message.role === 'toolResult' ? 'tool' : message.role, content: nativeText(message) });
    put(`http/00${index}.request.json`, { model, stream: true, max_tokens: 8192,
      reasoning: { effort: 'high' }, messages: [{ role: 'system', content: 'You are a coding assistant.' }, ...previous],
      tools: tools.map(tool => ({ type: 'function', function: { ...tool, strict: false } })) });
    const response = messages.filter(m => m.role === 'assistant')[index - 1];
    put(`http/00${index}.response.json`, { status: 200, headers: { 'content-type': 'text/event-stream' } });
    put(`http/00${index}.response.txt`, 'data: ' + JSON.stringify({ model, provider: 'fixture-provider',
      id: response.responseId, choices: [{ delta: { content: nativeText(response), reasoning: nativeReasoning(response) } }] }) + '\n\ndata: [DONE]\n');
  }
  function paths(directory, prefix = '') {
    return readdirSync(directory, { withFileTypes: true }).flatMap(entry => entry.isDirectory()
      ? paths(join(directory, entry.name), prefix + entry.name + '/') : [prefix + entry.name]);
  }
  const hashes = Object.fromEntries(paths(source).map(name => [name, digest(readFileSync(join(source, name)))]));
  const outcome = { status: 'completed', error: null, startedAt: '2026-10-29T00:00:00.000Z',
    finishedAt: '2026-10-29T00:01:00.000Z', userTurns: 2, modelRequests: 3, toolResults: 1,
    assistantUsage: [], captureErrors: [], messagesSha256: hashes['messages.json'],
    sessionFile: '/neutral/trajectory/sessions/native.jsonl', artifactHashes: hashes };
  put('outcome.json', outcome);
  writeFileSync(join(caseDir, 'workspace/example.txt'), malicious);
  const binding = { sourceMessagesSha256: hashes['messages.json'], sourcePatchSha256: hashes['changes.patch'] };
  const support = [
    { ...binding, sourceTranscriptUnchanged: true, sourcePatchUnchanged: true,
      allChecksPassed: true, checks: [{ name: 'Fixture only', passed: true, observed: 'Offline fixture', interpretation: 'No live inference' }] },
    { ...binding, turns: [{ turn: 1, title: 'Investigate', summary: 'Read the implementation.' },
      { turn: 2, title: 'Repair', summary: 'Make the requested change.' }],
      findingTitle: 'Fixture result', finding: 'Offline export worked.', limit: 'No live model call.' },
    { ...binding, files: [{ path: 'example.txt', sha256: digest(malicious) }, { path: 'removed.txt', deleted: true, sha256: null }] },
    { sourceMessagesSha256: hashes['messages.json'], selectionConditioned: true,
      selectedAttempt: 'attempt-2', criteria: ['No verbalized evaluation awareness in selected output.'],
      attempts: [{ id: 'attempt-1', status: 'failed', retained: true },
        { id: 'attempt-2', status: 'completed', retained: true, sourceMessagesSha256: hashes['messages.json'] }] },
  ];
  const supportPaths = support.map((value, index) => {
    const path = join(root, `support-${index}.json`); writeFileSync(path, JSON.stringify(value, null, 3) + '\n'); return path;
  });
  const output = join(root, 'page');
  const run = (env = {}) => spawnSync('python3', ['-c', python, join(toolRoot, 'export-review.py'),
    JSON.stringify([caseDir, ...supportPaths, output])], { encoding: 'utf8', env: { ...process.env, OPENROUTER_API_KEY: '', ...env } });
  return { root, caseDir, source, output, support, supportPaths, hashes, run };
}

test('fresh export preserves native bytes, carries all reasoning, and excludes full workspace archives', t => {
  const f = fixture(t), result = f.run();
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(readFileSync(join(f.output, 'evidence/messages.json')), readFileSync(join(f.source, 'messages.json')));
  assert.deepEqual(readFileSync(join(f.output, 'evidence/http/003.response.txt')), readFileSync(join(f.source, 'http/003.response.txt')));
  assert.deepEqual(readFileSync(join(f.output, 'selection.json')), readFileSync(f.supportPaths[3]));
  const data = JSON.parse(readFileSync(join(f.output, 'data.json')));
  assert.equal(data.fixture, true);
  assert.equal(data.reasoning.allAvailableReasoningPreserved, true);
  assert.equal(data.explainer.turns[1].messageIndex, 4);
  assert.equal(data.messages[4].content[0].text.includes(data.provenance.scenario.baseInstruction), false);
  assert.equal(data.files[0].content, malicious);
  assert.equal(data.artifacts.some(a => a.name.endsWith('.workspace.tar.gz')), false);
  assert.equal(data.corpus.snapshots[0].sha256, f.hashes['turns/001.workspace.tar.gz']);
  const members = spawnSync('tar', ['-tzf', join(f.output, 'transcript-evidence.tar.gz')], { encoding: 'utf8' });
  assert.equal(members.status, 0, members.stderr);
  assert.ok(members.stdout.includes('evidence/turns/001.workspace.json'));
  assert.equal(members.stdout.includes('.workspace.tar.gz'), false);
  assert.equal(JSON.parse(readFileSync(join(f.output, 'export-receipt.json'))).attempts, 2);
});

test('stale support or changed raw source hashes fail before creating a page', t => {
  const f = fixture(t);
  const validation = { ...f.support[0], sourceMessagesSha256: '0'.repeat(64) };
  writeFileSync(f.supportPaths[0], JSON.stringify(validation));
  let result = f.run();
  assert.notEqual(result.status, 0); assert.match(result.stderr, /stale message hash/);
  writeFileSync(f.supportPaths[0], JSON.stringify(f.support[0]));
  writeFileSync(join(f.source, 'http/003.response.txt'), 'changed');
  result = f.run();
  assert.notEqual(result.status, 0); assert.match(result.stderr, /Source artifact hash differs/);
  assert.equal(readdirSync(f.root).includes('page'), false);
});

test('lost older reasoning and mismatched native model identities fail fidelity checks', t => {
  const f = fixture(t);
  const requestPath = join(f.source, 'http/003.request.json');
  const request = JSON.parse(readFileSync(requestPath));
  request.messages.find(message => message.role === 'assistant').reasoning = 'lost older reasoning';
  writeFileSync(requestPath, JSON.stringify(request));
  const outcomePath = join(f.source, 'outcome.json');
  const outcome = JSON.parse(readFileSync(outcomePath));
  outcome.artifactHashes['http/003.request.json'] = digest(readFileSync(requestPath));
  writeFileSync(outcomePath, JSON.stringify(outcome));
  let result = f.run();
  assert.notEqual(result.status, 0); assert.match(result.stderr, /Prior readable reasoning was not carried/);
  const messagesPath = join(f.source, 'messages.json');
  const messages = JSON.parse(readFileSync(messagesPath));
  messages.find(message => message.role === 'assistant').api = 'other-api';
  writeFileSync(messagesPath, JSON.stringify(messages));
  outcome.artifactHashes['messages.json'] = outcome.messagesSha256 = digest(readFileSync(messagesPath));
  writeFileSync(outcomePath, JSON.stringify(outcome));
  result = f.run();
  assert.notEqual(result.status, 0); assert.match(result.stderr, /Native model\/API identity differs/);
});

test('workspace symlinks and changed files are rejected, and explicit credentials never export', t => {
  const f = fixture(t);
  rmSync(join(f.caseDir, 'workspace/example.txt'));
  symlinkSync(join(f.source, 'system.txt'), join(f.caseDir, 'workspace/example.txt'));
  let result = f.run(); assert.notEqual(result.status, 0); assert.match(result.stderr, /symbolic links/);
  rmSync(f.output, { recursive: true, force: true });
  rmSync(join(f.caseDir, 'workspace/example.txt'));
  writeFileSync(join(f.caseDir, 'workspace/example.txt'), malicious + 'stale');
  result = f.run(); assert.notEqual(result.status, 0); assert.match(result.stderr, /Changed file hash/);
  rmSync(f.output, { recursive: true, force: true });
  writeFileSync(join(f.caseDir, 'workspace/example.txt'), malicious);
  result = f.run({ OPENROUTER_API_KEY: 'globalThis.compromised=true' });
  assert.notEqual(result.status, 0); assert.match(result.stderr, /Credential scan found/);
  assert.equal(result.stderr.includes('globalThis.compromised=true'), false);
});

test('multipart exports retain separate manifest and assembled archive pins', t => {
  const f = fixture(t);
  f.support[3].downloadableCorpus = multipart;
  writeFileSync(f.supportPaths[3], JSON.stringify(f.support[3]));
  const result = f.run(); assert.equal(result.status, 0, result.stderr);
  const corpus = JSON.parse(readFileSync(join(f.output, 'corpus.json')));
  assert.deepEqual(corpus.download, multipart);
  assert.notEqual(corpus.download.sha256, corpus.download.archiveSha256);
  assert.equal(corpus.download.parts.reduce((sum, part) => sum + part.bytes, 0), corpus.download.archiveBytes);
});

test('multipart validation rejects malformed parts, byte totals, hashes and unsafe download references', () => {
  const bad = mutate => { const value = structuredClone(multipart); mutate(value); return value; };
  const invalid = [
    bad(c => { c.parts = []; }), bad(c => { c.parts.pop(); }),
    bad(c => { c.parts = {}; }), bad(c => { c.partCount = 3; }),
    bad(c => { c.archiveBytes = 6; }), bad(c => { c.archiveBytes = true; }),
    bad(c => { c.parts[0].bytes = -1; }), bad(c => { c.parts[0].bytes = 1.5; }),
    bad(c => { c.parts[0].sha256 = 'z'.repeat(64); }),
    bad(c => { c.archiveSha256 = 'bad'; }), bad(c => { c.sha256 = 'z'.repeat(64); }),
    bad(c => { c.parts[0].url = 'javascript:alert(1)'; }),
    bad(c => { c.url = 'https:///missing-host'; }),
    bad(c => { c.parts[1].url = c.parts[0].url; }),
    bad(c => { delete c.kind; }), bad(c => { c.kind = 'unknown'; }),
  ];
  const script = `import importlib.util,json,sys
spec=importlib.util.spec_from_file_location('export_review',sys.argv[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
valid,invalid=json.loads(sys.argv[2]);module.validate_corpus_download(valid)
module.validate_corpus_download({'url':'https://example.org/case.tar.gz','sha256':'a'*64,'bytes':123})
for index,value in enumerate(invalid):
 try:module.validate_corpus_download(value)
 except ValueError:continue
 raise AssertionError('accepted invalid multipart fixture '+str(index))
print(len(invalid))`;
  const result = spawnSync('python3', ['-c', script, join(toolRoot, 'export-review.py'),
    JSON.stringify([multipart, invalid])], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  assert.equal(Number(result.stdout.trim()), invalid.length);
});

test('viewer renders model HTML as literal text through the actual prose formatter', () => {
  class Node {
    constructor(tag = '#text', value = '') { this.tag = tag; this.children = []; this.value = value; }
    append(...children) { this.children.push(...children); }
    set textContent(value) { this.value = value; this.children = []; }
    get textContent() { return this.value + this.children.map(child => child.textContent).join(''); }
  }
  globalThis.document = { createElement: tag => new Node(tag), createTextNode: text => new Node('#text', text) };
  try {
    const output = prose(malicious + '\n\n`<svg onload=evil()>`\n\n```html\n<script>evil()</script>\n```');
    assert.ok(output.textContent.includes(malicious));
    const tags = node => [node.tag, ...node.children.flatMap(tags)];
    assert.ok(tags(output).every(tag => ['div', 'p', 'code', 'pre', '#text'].includes(tag)));
    assert.equal(globalThis.compromised, undefined);
  } finally { delete globalThis.document; }
});

test('viewer labels manifest versus archive pins and safely renders ordered part downloads', () => {
  class Node {
    constructor(tag = '#text', value = '') { this.tag = tag; this.children = []; this.value = value; }
    append(...children) { this.children.push(...children); }
    set textContent(value) { this.value = value; this.children = []; }
    get textContent() { return this.value + this.children.map(child => child.textContent).join(''); }
  }
  globalThis.document = { createElement: tag => new Node(tag), createTextNode: text => new Node('#text', text) };
  try {
    const output = corpusDownload(multipart);
    assert.ok(output.textContent.includes('Download full-corpus parts manifest'));
    assert.ok(output.textContent.includes(`Manifest SHA-256 ${multipart.sha256}`));
    assert.ok(output.textContent.includes(`Assembled archive SHA-256 ${multipart.archiveSha256}`));
    const nodes = node => [node, ...node.children.flatMap(nodes)];
    const links = nodes(output).filter(node => node.tag === 'a');
    assert.deepEqual(links.map(link => link.href), [multipart.url, ...multipart.parts.map(part => part.url)]);
    assert.deepEqual(links.slice(1).map(link => link.textContent), ['Download part 1 →', 'Download part 2 →']);
    assert.ok(nodes(output).every(node => ['div', 'a', 'p', 'ol', 'li'].includes(node.tag)));
    assert.throws(() => corpusDownload({ ...multipart, url: 'javascript:alert(1)' }), /require HTTPS/);
    const single = corpusDownload({ url: 'https://example.org/case.tar.gz', sha256: 'e'.repeat(64), bytes: 10 });
    assert.ok(single.textContent.includes('Download full corpus'));
    assert.equal(single.textContent.includes('manifest'), false);
    assert.equal(globalThis.compromised, undefined);
  } finally { delete globalThis.document; }
});
