const $ = (id) => document.getElementById(id);
const access = new URLSearchParams(location.hash.slice(1)).get("access");
if (access) {
  sessionStorage.setItem("authoring-access", access);
  history.replaceState(null, "", location.pathname + location.search);
}
const token = sessionStorage.getItem("authoring-access");
const state = {
  catalog: null,
  case: null,
  key: null,
  message: null,
  author: "",
  transcripts: new Map(),
  results: new Map(),
  checking: new Set(),
  generation: 0,
  pendingSaves: new Map(),
  saveErrors: new Set(),
  saving: null,
  savingKey: null,
  saveTimer: null,
  submitting: 0,
};
const styles = { normal: "Normal", casual: "Casual", persuasive: "Persuasive" };
function element(tag, text, cls) {
  const e = document.createElement(tag);
  if (text !== undefined) e.textContent = text;
  if (cls) e.className = cls;
  return e;
}
async function api(path, body) {
  const response = await fetch(path, {
    headers: {
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(body ? { "Content-Type": "application/json" } : {}),
    },
    ...(body ? { method: "POST", body: JSON.stringify(body) } : {}),
    ...(path === "/api/draft" ? { keepalive: true } : {}),
  });
  const result = await response.json();
  if (!response.ok)
    throw new Error(
      result.error || "The site could not complete that request.",
    );
  return result;
}
function keyFor(task, slot = $("slot").value, style = $("style").value) {
  return `${task}/${slot === "investigation" ? slot : `implementation/${slot}`}/${style}`;
}
function localDraft() {
  try {
    return JSON.parse(localStorage.getItem(`draft:${state.key}`) || "null");
  } catch {
    return null;
  }
}
function variants() {
  const choices = [],
    slots =
      $("slot").value === "investigation" ? ["investigation"] : ["a", "b"];
  for (const task of state.catalog.cases)
    for (const slot of slots)
      for (const style of Object.keys(styles)) {
        choices.push({
          task: task.id,
          slot,
          style,
          key: keyFor(task.id, slot, style),
        });
      }
  return choices;
}
function updateNavigation() {
  if (!state.catalog) return;
  const choices = variants(),
    current = choices.findIndex((c) => c.key === state.key);
  $("previous").disabled = !state.message || current <= 0;
  $("next").disabled = !state.message || current >= choices.length - 1;
  const perTask = $("slot").value === "investigation" ? 3 : 6;
  $("variant-position").textContent = `${(current % perTask) + 1} / ${perTask}`;
}
function renderDraft() {
  if (!state.key || !state.message) return;
  const text = $("entry").value,
    accepted =
      state.message.acceptedText === text &&
      state.message.acceptedText !== null;
  $("draft-status").textContent = state.saveErrors.has(state.key)
    ? "Saved in this browser · could not sync to the site"
    : state.pendingSaves.has(state.key) || state.savingKey === state.key
      ? "Saving…"
      : accepted
        ? "Accepted · saved"
        : text || state.message.hasDraft
          ? "Draft saved"
          : "";
  const busy = state.checking.has(state.key),
    writable = state.catalog.canWrite && state.catalog.writable;
  $("submit").disabled = !writable || busy || !text.trim();
  $("submit").textContent = busy ? "Checking…" : "Check & submit";
  updateNavigation();
  if (state.results.has(state.key)) {
    const job = state.results.get(state.key);
    if (accepted && job.text !== text) $("result").hidden = true;
    else showResult(job);
  }
}
function queueDraft() {
  if (!state.message || !state.catalog.canWrite) return;
  const draft = {
    key: state.key,
    text: $("entry").value,
    author: state.author,
  };
  localStorage.setItem(`draft:${draft.key}`, JSON.stringify(draft));
  state.pendingSaves.set(draft.key, draft);
  state.saveErrors.delete(draft.key);
  clearTimeout(state.saveTimer);
  state.saveTimer = setTimeout(() => void flushDrafts(), 500);
  renderDraft();
}
async function flushDrafts() {
  clearTimeout(state.saveTimer);
  if (state.submitting) return;
  if (state.saving) return state.saving;
  state.saving = (async () => {
    while (state.pendingSaves.size && !state.submitting) {
      const [key, draft] = state.pendingSaves.entries().next().value;
      state.pendingSaves.delete(key);
      state.savingKey = key;
      try {
        await api("/api/draft", draft);
        state.saveErrors.delete(key);
        if (key === state.key && state.message) state.message.hasDraft = true;
      } catch {
        if (!state.pendingSaves.has(key)) state.pendingSaves.set(key, draft);
        state.saveErrors.add(key);
        break;
      } finally {
        state.savingKey = null;
        renderDraft();
      }
    }
  })();
  try {
    await state.saving;
  } finally {
    state.saving = null;
    renderDraft();
  }
}
function drawTasks() {
  $("tasks").replaceChildren();
  const primary = state.catalog.accepted.filter((k) =>
    k.includes("/implementation/"),
  ).length;
  $("progress").textContent = `${primary} / 72 implementation messages`;
  for (const task of state.catalog.cases) {
    const done = state.catalog.accepted.filter((k) =>
      k.startsWith(task.id + "/implementation/"),
    ).length;
    const button = element(
      "button",
      task.title,
      "task" + (state.case === task.id ? " active" : ""),
    );
    button.type = "button";
    button.dataset.case = task.id;
    button.append(element("small", `${task.project} · ${done}/6`));
    button.onclick = () => select(task.id);
    $("tasks").append(button);
  }
}
function block(summary, text) {
  const d = element("details");
  d.append(element("summary", summary), element("pre", text));
  return d;
}
function drawTranscript() {
  const transcript = state.transcripts.get(state.case);
  if (!transcript || !state.message) return;
  const messages = transcript.messages,
    userIndices = messages.flatMap((m, i) => (m.role === "user" ? [i] : []));
  const targetIndex =
    userIndices[state.message.slot === "investigation" ? 0 : 1];
  const shown = $("full").checked
    ? messages
    : messages.slice(0, targetIndex + 1);
  $("context-note").textContent =
    state.message.slot === "investigation"
      ? "The highlighted request is the message you’re rewriting."
      : `The highlighted request is the original A assignment. Write requirement ${state.message.side.toUpperCase()} in the selected style.`;
  $("selection-note").textContent = transcript.selectionNote;
  $("review-note").textContent = transcript.reviewNote;
  $("source-data").href = transcript.dataUrl;
  $("source-evidence").href = transcript.evidenceUrl;
  $("transcript").replaceChildren();
  for (const [index, m] of shown.entries()) {
    const active = index === targetIndex;
    const article = element(
      "article",
      undefined,
      `event ${m.role}${active ? " rewrite-target" : ""}`,
    );
    article.dataset.messageIndex = index;
    if (active) {
      article.id = "rewrite-target";
      article.setAttribute("aria-current", "true");
    }
    article.append(
      element(
        "div",
        active
          ? "User · rewriting this message"
          : m.role === "toolResult"
            ? "Tool result"
            : m.role,
        "role",
      ),
    );
    const content =
      typeof m.content === "string"
        ? [{ type: "text", text: m.content }]
        : m.content || [];
    if (m.role === "toolResult")
      article.append(
        block(
          m.toolName || "Output",
          content.map((c) => c.text || "").join("\n"),
        ),
      );
    else
      for (const c of content) {
        if (c.type === "text") article.append(element("pre", c.text, "text"));
        if (c.type === "thinking")
          article.append(block("Reasoning", c.thinking));
        if (c.type === "toolCall")
          article.append(
            block(`Tool · ${c.name}`, JSON.stringify(c.arguments, null, 2)),
          );
      }
    $("transcript").append(article);
  }
  scrollToRewrite();
}
function scrollToRewrite() {
  requestAnimationFrame(() => {
    const target = $("rewrite-target");
    if (!target) return;
    const panel = target.closest(".transcript-panel"),
      heading = panel.querySelector(".panel-title");
    panel.scrollTop +=
      target.getBoundingClientRect().top -
      panel.getBoundingClientRect().top -
      heading.offsetHeight -
      16;
  });
}
async function select(task = state.case) {
  void flushDrafts();
  const generation = ++state.generation;
  state.message = null;
  state.case = task;
  state.key = keyFor(task);
  const key = state.key,
    selected = state.catalog.cases.find((c) => c.id === task);
  $("title").textContent = selected.title;
  $("project").textContent = selected.project;
  drawTasks();
  $("entry").disabled = true;
  $("submit").disabled = true;
  $("result").hidden = true;
  updateNavigation();
  const params = new URLSearchParams({
    task,
    message: $("slot").value,
    style: $("style").value,
  });
  history.replaceState(null, "", location.pathname + "?" + params);
  try {
    const [message, transcript] = await Promise.all([
      api("/api/message/" + key),
      state.transcripts.has(task)
        ? state.transcripts.get(task)
        : api("/api/transcript/" + task),
    ]);
    if (generation !== state.generation) return;
    state.transcripts.set(task, transcript);
    state.message = message;
    $("instruction").textContent =
      message.canonical.trim() ||
      "Investigate and explain before making implementation changes.";
    $("style-title").textContent = styles[message.framing] + " style";
    $("style-brief").textContent = message.checks.find(
      (c) => c.id === "style",
    ).requirement;
    const local = state.catalog.canWrite ? localDraft() : null;
    const serverText = message.hasDraft
      ? message.draft
      : message.acceptedText || "";
    $("entry").value = local ? local.text : serverText;
    state.author = local?.author ?? message.author ?? "";
    $("entry").disabled = !state.catalog.canWrite;
    drawTranscript();
    if (local && (local.text !== serverText || state.author !== message.author))
      queueDraft();
    renderDraft();
  } catch (error) {
    $("notice").textContent = error.message;
  }
}
function showResult(job) {
  const target = $("result");
  target.hidden = false;
  target.replaceChildren();
  target.className = job.status;
  if (job.status === "error") {
    target.append(
      element("h3", "Check could not finish"),
      element("p", job.result?.error || "Your draft is saved."),
    );
    return;
  }
  if (!job.result) {
    target.append(
      element("h3", "Checking this draft…"),
      element("p", "You can keep reading while the judge works."),
    );
    return;
  }
  const unchanged = job.text === $("entry").value;
  const title =
    job.status === "passed"
      ? unchanged
        ? "Accepted · saved to the message bank"
        : "Earlier version accepted · current edits need a new check"
      : unchanged
        ? "A few details need attention"
        : "Previous check · revised draft not yet checked";
  target.append(element("h3", title), element("p", job.result.summary));
  const failures = job.result.checks.filter((c) => !c.satisfied);
  if (failures.length) {
    const ul = element("ul");
    for (const c of failures) ul.append(element("li", c.feedback));
    target.append(ul);
  }
}
async function moveVariant(direction) {
  if (!state.message) return false;
  const choices = variants(),
    current = choices.findIndex((c) => c.key === state.key),
    next = choices[current + direction];
  if (!next) return false;
  $("slot").value = next.slot;
  $("style").value = next.style;
  await select(next.task);
  return true;
}
$("slot").onchange = () => select();
$("style").onchange = () => select();
$("full").onchange = drawTranscript;
window.addEventListener("resize", scrollToRewrite);
$("entry").oninput = queueDraft;
$("previous").onclick = () => moveVariant(-1);
$("next").onclick = () => moveVariant(1);
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "hidden") void flushDrafts();
});
window.addEventListener("pagehide", () => void flushDrafts());
$("submit").onclick = async () => {
  const key = state.key,
    text = $("entry").value,
    author = state.author,
    generation = state.generation;
  state.checking.add(key);
  state.results.set(key, { key, text, status: "queued", result: null });
  renderDraft();
  try {
    await flushDrafts();
    state.submitting++;
    let id;
    try {
      ({ id } = await api("/api/submit", { key, text, author }));
    } finally {
      state.submitting--;
      void flushDrafts();
    }
    let job;
    do {
      await new Promise((resolve) => setTimeout(resolve, 1200));
      job = await api("/api/job/" + id);
    } while (["queued", "running"].includes(job.status));
    state.results.set(key, job);
    state.catalog = await api("/api/catalog");
    drawTasks();
    if (key === state.key && state.message) {
      if (job.status === "passed") state.message.acceptedText = job.text;
      showResult(job);
      if (
        job.status === "passed" &&
        generation === state.generation &&
        text === $("entry").value
      ) {
        if (!(await moveVariant(1)))
          $("notice").textContent =
            "Accepted. You’ve reached the last variant.";
      }
    }
  } catch (error) {
    state.results.set(key, {
      key,
      text,
      status: "error",
      result: { error: error.message },
    });
  } finally {
    state.checking.delete(key);
    renderDraft();
  }
};
$("download").onclick = async () => {
  try {
    const response = await fetch("/api/export.patch", {
      headers: { Authorization: `Bearer ${token || ""}` },
    });
    if (!response.ok)
      throw new Error("Open your authoring link to download accepted edits.");
    const blob = await response.blob();
    if (!blob.size) throw new Error("No accepted edits to download yet.");
    const url = URL.createObjectURL(blob),
      a = element("a");
    a.href = url;
    a.download = "human-messages.patch";
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (error) {
    $("notice").textContent = error.message;
  }
};
try {
  state.catalog = await api("/api/catalog");
  if (!state.catalog.canWrite)
    $("notice").textContent =
      "Read-only view. Open your authoring link to edit and submit messages.";
  else if (!state.catalog.writable)
    $("notice").textContent =
      "Drafts autosave; live judging is currently disabled.";
  const params = new URLSearchParams(location.search),
    task =
      state.catalog.cases.find((c) => c.id === params.get("task"))?.id ||
      state.catalog.cases.find(
        (c) =>
          location.pathname === c.path ||
          location.pathname === c.path + "index.html",
      )?.id ||
      state.catalog.cases.find((c) => c.id === "langchain-json-recovery")?.id ||
      state.catalog.cases[0].id;
  if (["a", "b", "investigation"].includes(params.get("message")))
    $("slot").value = params.get("message");
  if (Object.keys(styles).includes(params.get("style")))
    $("style").value = params.get("style");
  $("download").disabled = !state.catalog.canWrite;
  await select(task);
} catch (error) {
  $("notice").textContent = error.message;
}
