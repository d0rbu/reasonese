const $ = (query, root = document) => root.querySelector(query);
const $$ = (query, root = document) => [...root.querySelectorAll(query)];
const el = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};
let data, selectedRole = 'all', expanded = false, toastTimer;
const json = value => JSON.stringify(value, null, 2);
const number = value => new Intl.NumberFormat('en-US').format(value);
export function originalCohortUrl(value) {
  // Archive links stay in a directory on this site, including under previews.
  return typeof value === 'string' && /^\/(?:[A-Za-z0-9_-]+\/)+$/.test(value) ? value : null;
}
function toast(text) {
  $('#toast').textContent = text; $('#toast').classList.add('visible');
  clearTimeout(toastTimer); toastTimer = setTimeout(() => $('#toast').classList.remove('visible'), 2200);
}
async function copy(text) {
  try { await navigator.clipboard.writeText(text); toast('Copied to clipboard'); }
  catch { toast('Clipboard unavailable. Use the download link instead.'); }
}
function inline(text, parent) {
  text.split(/(`[^`\n]+`)/g).forEach(part => {
    parent.append(part.startsWith('`') && part.endsWith('`') ? el('code', '', part.slice(1, -1)) : document.createTextNode(part));
  });
}
// Only text and code are rendered. Transcript content is never interpreted as HTML.
export function prose(text) {
  const node = el('div', 'content-text');
  text.split(/(````[^\n]*\n[\s\S]*?````|```[^\n]*\n[\s\S]*?```)/g).forEach(part => {
    const fence = part.startsWith('````') ? 4 : 3;
    if (part.startsWith('```')) node.append(el('pre', 'code-block', part.slice(part.indexOf('\n') + 1, -fence).replace(/\n$/, '')));
    else part.split(/\n\s*\n/).filter(Boolean).forEach(paragraph => { const p = el('p'); inline(paragraph, p); node.append(p); });
  });
  return node;
}
function details(label, text, className) {
  const node = el('details', className); node.dataset.expand = 'true'; node.open = expanded;
  node.append(el('summary', '', label), el('pre', '', text));
  return node;
}
function renderMessage(message, index) {
  const final = index === data.messages.length - 1 && message.role === 'assistant';
  const article = el('article', `event ${final ? 'final' : ''}`);
  article.id = `message-${index + 1}`; article.dataset.role = message.role;
  const labels = { user: `User turn ${data.messages.slice(0, index + 1).filter(m => m.role === 'user').length}`, assistant: final ? 'Final response' : 'Assistant', toolResult: `${message.toolName} result` };
  article.append(el('div', `avatar ${message.role}`, { user: 'U', assistant: 'K', toolResult: '>_' }[message.role] ?? '?'));
  const body = el('div', 'event-body'), header = el('div', 'event-header');
  header.append(el('strong', '', labels[message.role] ?? message.role));
  if (message.role === 'toolResult') header.append(el('span', 'event-label', message.isError ? 'Error' : 'Success'));
  if (message.role === 'assistant') header.append(el('span', 'event-label', final ? 'Complete' : `Response ${data.messages.slice(0, index + 1).filter(m => m.role === 'assistant').length}`));
  const offset = Math.max(0, (message.timestamp - Date.parse(data.outcome.startedAt)) / 1000);
  header.append(el('span', 'event-time', `${String(index + 1).padStart(2, '0')} / +${offset.toFixed(2)}s`));
  body.append(header);
  const card = el('div', 'event-card');
  const blocks = Array.isArray(message.content) ? message.content : [{ type: 'text', text: String(message.content ?? '') }];
  for (const block of blocks) {
    if (block.type === 'thinking') card.append(details(data.normalization ? 'Kimi reasoning · paths normalized' : 'Kimi reasoning · plaintext returned by provider', block.thinking ?? json(block), 'reasoning'));
    else if (block.type === 'text') card.append(message.role === 'toolResult' ? el('pre', 'tool-output', block.text) : prose(block.text));
    else if (block.type === 'toolCall') {
      const tool = el('div', 'tool-call'), title = el('div', 'tool-title');
      title.append(el('strong', '', block.name), el('span', '', block.arguments?.path ?? 'Shell command'));
      const button = el('button', 'copy-button', 'Copy'); button.setAttribute('aria-label', `Copy ${block.name} tool arguments`);
      button.addEventListener('click', () => copy(json(block.arguments)));
      title.append(button); tool.append(title);
      const args = block.arguments ?? {};
      if (block.name === 'write' && typeof args.content === 'string') tool.append(el('pre', '', args.content));
      else if (block.name === 'bash' && typeof args.command === 'string') tool.append(el('pre', '', `$ ${args.command}`));
      else tool.append(el('pre', '', json(args)));
      card.append(tool);
    } else card.append(details(`${block.type ?? 'Additional'} content`, json(block), 'raw-message'));
  }
  card.append(details(data.normalization ? 'Message JSON · paths normalized' : 'Raw message JSON', json(message), 'raw-message'));
  body.append(card);
  if (message.usage) body.append(el('div', 'event-meta', `${number(message.usage.totalTokens)} total tokens · ${message.stopReason === 'toolUse' ? 'tool use' : message.stopReason}`));
  article.append(body); return article;
}
function renderTimeline() {
  const query = $('#search').value.toLocaleLowerCase().trim();
  const timeline = $('#timeline'); timeline.replaceChildren(); let count = 0;
  data.messages.forEach((message, index) => {
    if (selectedRole !== 'all' && message.role !== selectedRole) return;
    if (query && !json(message).toLocaleLowerCase().includes(query)) return;
    timeline.append(renderMessage(message, index)); count++;
  });
  $('#visible-count').textContent = `${count} of ${data.messages.length} messages`;
  $('#empty').hidden = count !== 0;
}
function changeTab() {
  const name = ['overview', 'files', 'context', 'transcript'].includes(location.hash.slice(1)) ? location.hash.slice(1) : 'overview';
  $$('.panel').forEach(panel => { panel.hidden = panel.id !== name; });
  $$('.tabs a').forEach(link => {
    const active = link.hash === `#${name}`; link.classList.toggle('active', active);
    if (active) link.setAttribute('aria-current', 'page'); else link.removeAttribute('aria-current');
  });
}
function provenance(label, value, href) {
  const cell = el('div', 'provenance-item'); cell.append(el('span', '', label));
  const item = el('div');
  if (href) { const a = el('a', '', value); a.href = href; a.target = '_blank'; a.rel = 'noopener noreferrer'; item.append(a); }
  else item.textContent = value;
  cell.append(item); $('#provenance-grid').append(cell);
}
async function init() {
  const response = await fetch('data.json', {cache: 'no-store'}); if (!response.ok) throw new Error(`Transcript data: HTTP ${response.status}`);
  data = await response.json();
  if (data.fixture) $('#scenario-subtitle').textContent = 'Offline exporter fixture; no live model call was made.';
  if (data.selection) {
    const note = $('#selection-note'); note.hidden = false;
    const selection = data.selection;
    note.append(el('strong', '', 'Selected genuine resample'), el('p', '',
      `This transcript was selected from ${selection.attempts.length} retained attempts. Selected attempt: ${selection.selectedAttempt}. Selection was conditioned on the criteria below; this is not an unselected sample.`));
    const criteria = Array.isArray(selection.criteria) ? selection.criteria.join(' · ') : selection.criteria;
    note.append(el('p', '', `Selection criteria: ${criteria}`), el('p', '',
      'Screening for verbalized evaluation awareness does not establish the absence of internal awareness. Model messages and provider streams are preserved as captured.'));
    const attempts = el('ul');
    selection.attempts.forEach(attempt => attempts.append(el('li', '', `${attempt.id}: ${attempt.status}; retained`)));
    note.append(attempts);
    const link = el('a', '', 'Download selection receipt →'); link.href = 'selection.json'; note.append(link);
    const archive = originalCohortUrl(selection.originalCohortUrl);
    if (archive) {
      const original = el('a', 'artifact-link', 'Review the original 12-scenario cohort →');
      original.href = archive; note.append(original);
    }
  }
  if (data.corpus) {
    const note = $('#corpus-note'); note.hidden = false;
    note.append(el('strong', '', 'Full workspace corpus'), el('p', '', data.corpus.notice));
    if (data.corpus.download) {
      const corpus = data.corpus.download, link = el('a', '', 'Download full corpus →');
      link.href = corpus.url; note.append(link, el('p', '', `SHA-256 ${corpus.sha256} · ${number(corpus.bytes)} B`));
    } else note.append(el('p', '', 'A full-corpus download URL has not been attached to this page.'));
    for (const snapshot of data.corpus.snapshots) note.append(el('p', 'file-footer',
      `${snapshot.name} · ${number(snapshot.bytes)} B · SHA-256 ${snapshot.sha256}`));
  }
  if (data.normalization) {
    const note = $('#normalization-note'); note.hidden = false;
    note.append(el('strong', '', 'Paths normalized after collection'), el('p', '', data.normalization.notice));
    if (data.normalization.streamNotice) note.append(el('p', '', data.normalization.streamNotice));
    for (const [label, href] of [['Download context for reuse', data.normalization.context], ['Path changes & cue audit', data.normalization.audit], ['Original captured transcript', data.normalization.originalReview], ['Original evidence bundle', data.normalization.originalBundle]]) {
      const link = el('a', 'artifact-link', label); link.href = href; note.append(link);
    }
    note.append(el('p', '', 'The audit flags remaining evaluator speculation and environment restrictions. Read the collection-wide findings at the link below.'));
    const report = el('a', '', 'Read the environment audit →'); report.href = '/kimi-transcripts/environment-audit.html'; note.append(report);
  }
  const catalogResponse = await fetch(data.catalogUrl ?? '/kimi-transcripts/catalog.json', {cache:'no-store'});
  if (!catalogResponse.ok) throw new Error(`Scenario list: HTTP ${catalogResponse.status}`);
  const catalog = await catalogResponse.json();
  const selected = catalog.scenarios.find(entry => entry.id === data.provenance.scenario.id);
  if (!selected) throw new Error('Scenario is absent from the catalog');
  document.title = `${selected.title} · Kimi base transcripts · Reasonese`;
  $('#scenario-title').textContent = selected.title;
  $('#scenario-eyebrow').textContent = `BASE SESSION · ${selected.project.toUpperCase()}`;
  $('#task-caption').textContent = selected.caption;
  $('#task-description').textContent = data.provenance.scenario.prompt;
  $('#catalog-heading').textContent = `SCENARIOS · ${catalog.scenarios.length}`;
  for (const [index, entry] of catalog.scenarios.entries()) {
    const link = el('a', 'scenario-card' + (entry.id === selected.id ? ' active' : ''));
    link.href = entry.path; link.dataset.scenario = entry.id;
    if (entry.id === selected.id) link.setAttribute('aria-current', 'page');
    const number = el('span', 'scenario-number', String(index + 1).padStart(2, '0'));
    const text = el('span', 'scenario-card-text');
    text.append(el('strong', '', entry.title), el('small', '', `${entry.project} · ${entry.userTurns} turns`));
    text.append(el('small', 'scenario-review' + (entry.checksPassed ? '' : ' needs-review'), entry.checksPassed ? 'Scoped checks passed' : 'Review needed'));
    link.append(number, text); $('#scenario-list').append(link);
  }
  $('#scenario-list .active')?.scrollIntoView({block:'nearest', inline:'nearest'});
  $('#status').textContent = data.outcome.status;
  $('#run-date').textContent = new Date(data.outcome.startedAt).toISOString().slice(0, 10);
  $('#harness').textContent = `Pi ${data.provenance.collector.pi}`;
  $('#user-turns').textContent = data.outcome.userTurns;
  $('#duration').textContent = `${((Date.parse(data.outcome.finishedAt) - Date.parse(data.outcome.startedAt)) / 60000).toFixed(1)} min including review`;
  $('#requests').textContent = data.outcome.modelRequests;
  $('#tools').textContent = data.outcome.toolResults;
  $('#tool-names').textContent = [...new Set(data.messages.filter(m => m.role === 'toolResult').map(m => m.toolName))].join(' · ');
  $('#cost').textContent = `$${data.outcome.assistantUsage.reduce((sum, use) => sum + (use?.cost?.total ?? 0), 0).toFixed(4)}`;
  $('#message-count').textContent = data.messages.length;
  $('#reasoning-finding').textContent = data.reasoning.summary;
  const card = el('article', 'check-card');
  card.append(el('span', 'eyebrow', 'FIRST INSTRUCTION'), el('p', 'pair-text', data.provenance.scenario.baseInstruction));
  $('#base-requirement').append(card);
  $('#sidebar-date').textContent = new Date(data.outcome.startedAt).toLocaleDateString('en-US', {timeZone:'America/Chicago', day:'2-digit', month:'short', year:'numeric'}).toUpperCase();
  data.explainer.turns.forEach(step => {
    const card = el('article', 'story-step');
    card.append(el('h3', '', `Turn ${step.turn} · ${step.title}`), el('p', '', step.summary));
    const link = el('a', '', 'Read this exchange →'); link.href = '#transcript';
    link.addEventListener('click', () => {
      selectedRole = 'all'; $('#search').value = '';
      $$('.filters button').forEach(b => { b.classList.toggle('selected', b.dataset.role === 'all'); b.setAttribute('aria-pressed', String(b.dataset.role === 'all')); });
      renderTimeline();
      setTimeout(() => document.getElementById(`message-${step.messageIndex + 1}`).scrollIntoView({ block: 'start' }), 50);
    });
    card.append(link); $('#story').append(card);
  });
  $('#finding').append(el('strong', '', data.explainer.findingTitle), el('p', '', data.explainer.finding), el('p', '', data.explainer.limit));
  const table = el('table', 'behavior-table');
  const head = el('tr'); ['Check', 'Observed result', 'Interpretation'].forEach(v => head.append(el('th', '', v))); table.append(head);
  data.validation.checks.forEach(check => { const row = el('tr'); [check.name, check.observed, check.interpretation].forEach(v => row.append(el('td', '', v))); table.append(row); });
  $('#behavior').append(table);
  $('#validation-json').textContent = json(data.validation);
  $('#patch').textContent = data.patch;
  data.files.forEach(file => {
    const card = el('details', 'evidence-details'); card.append(el('summary', '', file.path), el('pre', '', file.content));
    if (!file.deleted) {
      const link = el('a', 'action', 'Download file ↓'); link.href = `files/${file.path.split('/').map(encodeURIComponent).join('/')}`; link.download = file.path.split('/').at(-1);
      card.append(link, el('p', 'file-footer', `SHA-256 ${file.sha256}`));
    }
    $('#files-list').append(card);
  });
  $('#repo-context').textContent = data.provenance.context.content;
  $('#system-prompt').textContent = data.system;
  $('#tool-definitions').textContent = json(data.toolDefinitions);
  $('#outcome-json').textContent = json(data.outcome);
  $('#operator-audit').textContent = json(data.turns);
  const source = data.provenance.scenario, repo = source.repository.replace(/\.git$/, '');
  provenance('Repository', repo.split('github.com/')[1], `${repo}/tree/${source.revision}`);
  provenance('Pinned revision', source.revision.slice(0, 12), `${repo}/commit/${source.revision}`);
  provenance('Requested model', data.provenance.manifest.model);
  provenance('Reported model', data.reportedModels.join(', '));
  provenance('Inference provider', data.reportedProviders.join(', '));
  provenance('User-message author', 'Codex operator · adaptive, not a human session log');
  provenance('Original transcript SHA-256', data.outcome.messagesSha256);
  if (data.normalization) provenance('Edited messages SHA-256', data.normalization.messagesSha256);
  provenance('Environment', 'Prepared task dependencies; native tool execution');
  data.artifacts.forEach(artifact => { const a = el('a', 'artifact-link'); a.href = `evidence/${artifact.name.split('/').map(encodeURIComponent).join('/')}`; a.download = artifact.name.split('/').at(-1); a.append(el('span', '', artifact.name), el('small', '', `${number(artifact.bytes)} B ↓`)); $('#artifact-list').append(a); });
  $('#search').addEventListener('input', renderTimeline);
  $$('.filters button').forEach(button => button.addEventListener('click', () => { selectedRole = button.dataset.role; $$('.filters button').forEach(b => { b.classList.toggle('selected', b === button); b.setAttribute('aria-pressed', String(b === button)); }); renderTimeline(); }));
  $('#expand').addEventListener('click', () => { expanded = !expanded; $('#expand').textContent = expanded ? 'Collapse all' : 'Expand all'; $$('[data-expand]').forEach(d => d.open = expanded); });
  window.addEventListener('hashchange', changeTab);
  document.addEventListener('keydown', e => { if (e.key === '/' && !['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) { e.preventDefault(); location.hash = 'transcript'; $('#search').focus(); } });
  renderTimeline(); changeTab();
}
if (typeof document !== 'undefined') init().catch(error => { $('#error').hidden = false; $('#error').textContent = error.message; });
