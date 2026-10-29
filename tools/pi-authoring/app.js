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
  transcripts: new Map(),
  results: new Map(),
  checking: new Set(),
  generation: 0,
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
function updateDraft() {
  if (!state.key || !state.message) return;
  localStorage.setItem(
    `draft:${state.key}`,
    JSON.stringify({ text: $("entry").value, author: $("author").value }),
  );
  $("words").textContent =
    `${$("entry").value.trim().split(/\s+/).filter(Boolean).length} words`;
  const accepted =
    state.message.acceptedText === $("entry").value &&
    state.message.acceptedText !== null;
  $("draft-status").textContent = accepted
    ? "Accepted version"
    : "Draft kept in this browser";
  const busy = state.checking.has(state.key),
    writable = state.catalog.canWrite && state.catalog.writable;
  $("save").disabled = !writable || !$("entry").value.trim();
  $("submit").disabled = !writable || busy || !$("entry").value.trim();
  $("submit").textContent = busy ? "Checking…" : "Check & submit";
  $("next").hidden = !accepted;
  if (state.results.has(state.key)) {
    const job = state.results.get(state.key);
    if (accepted && job.text !== $("entry").value) $("result").hidden = true;
    else showResult(job);
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
  if (!transcript) return;
  const messages = transcript.messages;
  const second = messages.findIndex((m, i) => m.role === "user" && i > 0);
  const full = $("full").checked;
  const shown = full || second < 0 ? messages : messages.slice(0, second);
  $("context-note").textContent = full
    ? "Complete recorded source · requirement A"
    : $("slot").value === "investigation"
      ? "Recorded investigation · your message replaces its opening request"
      : "Investigation context before the implementation assignment";
  $("selection-note").textContent = transcript.selectionNote;
  $("review-note").textContent = transcript.reviewNote;
  $("source-data").href = transcript.dataUrl;
  $("source-evidence").href = transcript.evidenceUrl;
  $("transcript").replaceChildren();
  for (const m of shown) {
    const article = element("article", undefined, `event ${m.role}`);
    const role = m.role === "toolResult" ? "Tool result" : m.role;
    article.append(element("div", role, "role"));
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
}
async function select(task = state.case) {
  const generation = ++state.generation;
  state.message = null;
  state.case = task;
  state.key = keyFor(task);
  const key = state.key;
  const selected = state.catalog.cases.find((c) => c.id === task);
  $("title").textContent = selected.title;
  $("project").textContent = selected.project;
  drawTasks();
  $("entry").disabled = true;
  $("save").disabled = true;
  $("submit").disabled = true;
  $("result").hidden = true;
  $("next").hidden = true;
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
    $("requirements").replaceChildren(
      ...message.checks
        .filter((c) => c.id.startsWith("task-"))
        .map((c) => element("li", c.requirement)),
    );
    $("style-title").textContent = styles[message.framing] + " style";
    $("style-brief").textContent = message.checks.find(
      (c) => c.id === "style",
    ).requirement;
    $("reference").textContent = message.reference;
    const local = localDraft();
    $("entry").value = local
      ? local.text
      : message.draft || message.acceptedText || "";
    $("author").value =
      local?.author ||
      message.author ||
      localStorage.getItem("authoring-name") ||
      "";
    $("entry").disabled = !(state.catalog.canWrite && state.catalog.writable);
    drawTranscript();
    updateDraft();
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
    for (const c of failures) {
      const li = element("li");
      const requirement =
        state.message.checks.find((r) => r.id === c.id)?.requirement || c.id;
      li.append(element("strong", requirement), element("p", c.feedback));
      ul.append(li);
    }
    target.append(ul);
  }
}
$("slot").onchange = () => select();
$("style").onchange = () => select();
$("full").onchange = drawTranscript;
$("entry").oninput = updateDraft;
$("author").oninput = () => {
  localStorage.setItem("authoring-name", $("author").value);
  updateDraft();
};
$("save").onclick = async () => {
  const key = state.key,
    text = $("entry").value;
  try {
    await api("/api/draft", { key, text, author: $("author").value });
    if (key === state.key && text === $("entry").value)
      $("draft-status").textContent = "Draft saved to the site";
  } catch (error) {
    $("notice").textContent = error.message;
  }
};
$("submit").onclick = async () => {
  const key = state.key,
    text = $("entry").value,
    author = $("author").value;
  state.checking.add(key);
  updateDraft();
  state.results.set(key, { key, text, status: "queued", result: null });
  if (key === state.key) showResult(state.results.get(key));
  try {
    const { id } = await api("/api/submit", { key, text, author });
    let job;
    do {
      await new Promise((resolve) => setTimeout(resolve, 1200));
      job = await api("/api/job/" + id);
    } while (["queued", "running"].includes(job.status));
    state.results.set(key, job);
    state.catalog = await api("/api/catalog");
    drawTasks();
    if (key === state.key) {
      if (job.status === "passed") state.message.acceptedText = job.text;
      showResult(job);
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
    if (key === state.key) updateDraft();
  }
};
$("next").onclick = () => {
  const currentOptional = $("slot").value === "investigation",
    choices = [];
  for (const task of state.catalog.cases)
    for (const slot of currentOptional ? ["investigation"] : ["a", "b"])
      for (const style of Object.keys(styles))
        choices.push({
          task: task.id,
          slot,
          style,
          key: keyFor(task.id, slot, style),
        });
  const current = choices.findIndex((c) => c.key === state.key);
  const next = [
    ...choices.slice(current + 1),
    ...choices.slice(0, current + 1),
  ].find((c) => !state.catalog.accepted.includes(c.key));
  if (next) {
    $("slot").value = next.slot;
    $("style").value = next.style;
    select(next.task);
  } else $("notice").textContent = "All messages in this group are accepted.";
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
      "Read-only view. Open your authoring link to save drafts and submit messages.";
  else if (!state.catalog.writable)
    $("notice").textContent =
      "Draft review is available; live judging is currently disabled.";
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
