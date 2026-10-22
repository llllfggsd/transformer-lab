import {createLineComparison} from "/static/comparison.mjs";
import {createAuthScreen, draftStorageKey} from "/static/auth.mjs";

const $ = (selector) => document.querySelector(selector);
const editor = $("#editor");
const comparison = createLineComparison(editor, $("#reference-code"));
let state, lesson, busy = false, loading = false, saveTimer, toastTimer;
let saveQueue = Promise.resolve();
const draftKey = (id) => draftStorageKey(state.user.id, id);
const escape = (value) => String(value).replace(/[&<>"']/g, (char) => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}[char]));
const auth = createAuthScreen({onAuthenticated: loadCourse, onSignedOut: clearCourse});

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    credentials: "same-origin",
    headers: {"Content-Type": "application/json", ...(state ? {"X-Lab-Token": state.token, "X-Lab-User": state.user.id} : {}), ...options.headers},
  });
  const data = await response.json();
  if (!response.ok) {
    if (data.code === "ACCOUNT_CHANGED" || (response.status === 401 && state?.user)) window.location.reload();
    throw new Error(typeof data.detail === "string" ? data.detail : "请求未能完成，请检查输入后重试。");
  }
  return data;
}

function toast(message) {
  clearTimeout(toastTimer);
  $("#toast").textContent = message;
  $("#toast").hidden = false;
  toastTimer = setTimeout(() => { $("#toast").hidden = true; }, 4200);
}

function renderState() {
  const stageNumber = String(state.stage).padStart(2, "0");
  $("#stage-breadcrumb").textContent = `阶段 ${stageNumber}`;
  $("#stage-eyebrow").textContent = `${state.stage_tag} / STAGE ${stageNumber}`;
  $("#stage-title").textContent = state.stage === 1 ? "从第一行 Python 开始。" : state.stage_title;
  $("#stage-description").textContent = state.stage_description;
  $("#export-notebook").href = `/api/export?stage=${state.stage}`;
  $("#dependency-note").hidden = !state.missing_packages.length;
  $("#dependency-note").textContent = `本阶段还需要安装依赖。在运行 main.py 的 Python 环境执行：python -m pip install ${state.missing_packages.join(" ")}`;
  $("#progress-count").innerHTML = `${state.completed} <span>/ ${state.total}</span>`;
  $("#progress-fill").style.width = `${state.completed / state.total * 100}%`;
  $("#progress-note").textContent = state.course_complete ? "全部十个阶段已完成，随时可以回看复习" : state.stage_complete ? state.next_stage ? `验收通过 · 第 ${state.next_stage} 阶段已解锁` : "本阶段验收已通过" : state.quiz_available ? "代码练习完成 · 请完成阶段验收" : "完成代码与验收，继续下一阶段";
  $("#stages").replaceChildren(...state.stages.map((stage) => {
    const button = document.createElement("button");
    button.className = `stage-button${stage.number === state.stage ? " active" : ""}`;
    button.disabled = !stage.available;
    button.title = stage.available ? `${stage.title} · ${stage.completed}/${stage.total} 段已完成` : "完成上一阶段的示例、练习和验收后解锁";
    button.innerHTML = `<span class="stage-num">${stage.status === "completed" ? "✓" : String(stage.number).padStart(2, "0")}</span><span class="stage-name">${escape(stage.title)}</span><span class="stage-lock">${stage.available ? "" : "⌑"}</span>`;
    if (stage.number === state.stage) button.setAttribute("aria-current", "step");
    button.addEventListener("click", () => selectStage(stage.number));
    return button;
  }));
  renderOutline();
  renderResources();
  updateNavigation();
}

function renderResources() {
  const section = $("#course-resources");
  section.replaceChildren();
  section.hidden = !state.sources.length && state.stage !== 9;
  if (section.hidden) return;
  const title = document.createElement("strong");
  title.textContent = "本章参考与拓展";
  section.append(title);
  state.sources.forEach((url, index) => {
    const link = document.createElement("a");
    link.href = url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = `官方资料 ${index + 1} ↗`;
    section.append(link);
  });
  if (state.stage === 9) {
    const link = document.createElement("a");
    link.href = "/api/materials/pretrained";
    link.textContent = "下载真实预训练模型实践代码（需联网下载权重）";
    section.append(link);
  }
}

async function selectStage(number) {
  if (busy || loading || !state.stages.find((item) => item.number === number)?.available) return;
  loading = true;
  updateNavigation();
  try {
    await flushDraft();
    const nextState = await api(`/api/state?stage=${number}`);
    state = nextState;
    lesson = null;
    $("#quiz").hidden = true;
    $("#quiz").replaceChildren();
    $("#outline").hidden = true;
    $("#outline-toggle").setAttribute("aria-expanded", "false");
    renderState();
    loading = false;
    await selectLesson(Math.min(state.frontier, state.total - 1));
    if (state.quiz_available) await renderQuiz();
  } catch (error) { toast(error.message); }
  finally { loading = false; updateNavigation(); }
}

function renderOutline() {
  $("#outline").replaceChildren(...["example", "practice"].map((kind) => {
    const items = state.lessons.map((item, index) => ({...item, index})).filter((item) => item.kind === kind);
    const completed = items.filter((item) => item.completed).length;
    const section = document.createElement("section");
    section.id = `outline-${kind}`;
    section.className = "outline-section";
    const title = kind === "example" ? "示例跟练" : "独立练习";
    section.innerHTML = `<h3>${title}<span>${completed} / ${items.length} 已完成</span></h3><p>${kind === "example" ? "对照完整示例手敲，通过运行与代码结构核对。" : "完成全部示例后依次解锁；自己写解法，通过测试即可，无需与示例一致。"}</p>`;
    const grid = document.createElement("div");
    grid.className = "outline-items";
    for (const item of items) {
      const button = document.createElement("button");
      button.className = lesson?.id === item.id ? "selected" : "";
      button.disabled = !item.unlocked || busy || loading;
      button.textContent = `${item.completed ? "✓" : item.unlocked ? "○" : "⌑"} ${String(item.index + 1).padStart(2, "0")} · ${item.title}`;
      button.title = item.unlocked ? item.title : "请先按顺序完成前面的代码段";
      button.addEventListener("click", () => selectLesson(item.index));
      grid.append(button);
    }
    section.append(grid);
    return section;
  }));
}

function renderLearningPath() {
  const groups = [
    {kind: "example", title: "示例跟练", description: "对照完整代码手敲，运行并核对结构"},
    {kind: "practice", title: "独立练习", description: "根据题目自己写解法，通过测试即可"},
    {kind: "quiz", title: "阶段验收", description: "回答理解题，确认掌握本阶段内容"},
  ];
  const examples = state.lessons.filter((item) => item.kind === "example");
  const remainingExamples = examples.filter((item) => !item.completed).length;
  const active = state.quiz_available && !state.stage_complete ? "quiz" : lesson?.kind || state.lessons[Math.min(state.frontier, state.total - 1)].kind;
  $("#learning-path").replaceChildren(...groups.map((group, index) => {
    const items = state.lessons.map((item, index) => ({...item, index})).filter((item) => item.kind === group.kind);
    const count = items.filter((item) => item.completed).length;
    const unlocked = group.kind === "quiz" ? state.quiz_available : items.some((item) => item.unlocked);
    const completed = group.kind === "quiz" ? state.stage_complete : count === items.length;
    const button = document.createElement("button");
    button.className = `learning-step${active === group.kind ? " active" : ""}${!unlocked ? " locked" : ""}`;
    button.disabled = busy || loading;
    if (active === group.kind) button.setAttribute("aria-current", "step");
    const status = completed ? "已完成" : !unlocked ? "待解锁" : active === group.kind ? "进行中" : "可继续";
    const note = group.kind === "quiz" ? state.stage_complete ? "验收已通过" : unlocked ? `${state.quiz_count} 道理解题 · 点击开始` : "完成示例和练习后解锁" : group.kind === "practice" && !unlocked ? `完成全部示例后解锁，还差 ${remainingExamples} 段 · 查看练习目录` : `${count} / ${items.length} ${group.kind === "practice" ? "题" : "段"}已完成 · 点击继续`;
    button.innerHTML = `<span class="learning-step-title"><span class="learning-step-number">${String(index + 1).padStart(2, "0")}</span><strong>${group.title}</strong><span class="learning-step-status">${status}</span></span><span class="learning-step-description">${group.description}</span><span class="learning-step-note">${note}</span>`;
    button.addEventListener("click", () => {
      if (!unlocked) {
        $("#outline").hidden = false;
        $("#outline-toggle").setAttribute("aria-expanded", "true");
        if (group.kind === "practice") $("#outline-practice").scrollIntoView({behavior: "smooth", block: "nearest"});
        toast(group.kind === "practice" ? `独立练习共有 ${items.length} 道，完成剩余 ${remainingExamples} 段示例后解锁。` : "完成全部示例和独立练习后，才能进入阶段验收。");
      } else if (group.kind === "quiz") {
        $("#quiz").scrollIntoView({behavior: "smooth", block: "start"});
      } else {
        const next = items.find((item) => !item.completed && item.unlocked) || items[0];
        selectLesson(next.index);
      }
    });
    return button;
  }));
}

function highlightCode(code) {
  const tokenPattern = /#[^\n]*|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\b(?:def|class|if|elif|else|for|while|in|return|raise|try|except|with|as|import|from|True|False|None|pass|and|or|not)\b|\b\d+(?:\.\d+)?\b/g;
  return code.split("\n").map((line, index) => {
    let result = "", previous = 0;
    for (const match of line.matchAll(tokenPattern)) {
      result += escape(line.slice(previous, match.index));
      const value = match[0];
      const kind = value.startsWith("#") ? "comment" : /^["']/.test(value) ? "string" : /^\d/.test(value) ? "number" : "keyword";
      result += `<span class="tok-${kind}">${escape(value)}</span>`;
      previous = match.index + value.length;
    }
    return `<span class="code-line"><span class="line-number">${index + 1}</span>${result + escape(line.slice(previous))}</span>`;
  }).join("");
}

function renderOutputs(outputs = []) {
  const container = $("#output");
  container.replaceChildren();
  if (!outputs.length) {
    container.innerHTML = '<div class="output-empty"><span>›_</span><p>每一次运行，都是一次理解。</p><small>输出和错误信息会出现在这里</small></div>';
    return;
  }
  for (const output of outputs) {
    const pre = document.createElement("pre");
    if (output.type === "error") pre.className = "error";
    // Jupyter 错误包含 ANSI 颜色控制码；浏览器使用自身的错误样式。
    pre.textContent = output.text.replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, "");
    container.append(pre);
  }
}

async function selectLesson(index) {
  if (busy || loading || !state.lessons[index]?.unlocked) return;
  loading = true;
  updateNavigation();
  try {
    await flushDraft();
    lesson = await api(`/api/lessons/${state.lessons[index].id}`);
    $("#lesson-title").textContent = lesson.title;
    $("#lesson-description").innerHTML = escape(lesson.description).replace(/`([^`]+)`/g, "<code>$1</code>");
    $("#reference-code").innerHTML = highlightCode(lesson.code);
    $("#reference-filename").textContent = lesson.filename || (lesson.kind === "practice" ? "practice.py" : index < 7 ? "python_basics.py" : "numpy_basics.py");
    $("#lesson-type").textContent = lesson.kind === "practice" ? "独立练习" : state.stage === 1 ? index < 7 ? "PYTHON 基础" : "NUMPY 数组" : state.stage_tag;
    $("#lesson-group").textContent = lesson.kind === "practice" ? "独立练习" : "示例跟练";
    $("#comparison-title").textContent = lesson.kind === "practice" ? "独立练习 · 自己写出解法" : "示例跟练 · 逐行对照";
    $("#reference-title").textContent = lesson.kind === "practice" ? "题目接口" : "示例代码";
    $("#reference-caption").textContent = lesson.kind === "practice" ? "待完成的函数定义（不是答案）" : "只读示例";
    $("#evaluation-rule").textContent = lesson.kind === "practice" ? "通过条件：代码运行成功，并通过本题测试。允许不同解法，不要求与示例一致。" : "通过条件：代码运行成功，且变量、数值、文字内容及结构正确。忽略空白、注释、引号风格和字符串中的中英文符号。";
    $("#lesson-position").textContent = `${String(index + 1).padStart(2, "0")} / ${state.total}`;
    $("#hint-title").textContent = lesson.kind === "practice" ? "把理解写进你的解法" : "动手之前，先预测一下";
    $("#lesson-hint").textContent = lesson.hint;
    $("#reference-guide").textContent = lesson.kind === "practice" ? "本题按运行结果验收，可以自行组织代码与行数。左侧接口供参考，不需要逐行照抄未完成的函数体。" : "从左侧第 1 行开始，在右侧相同行号输入。长代码可在任意一侧滚动，另一侧会同步跟随。";
    $("#cell-caption").textContent = lesson.kind === "practice" ? "手敲定义，写出你的解法" : "在对应行输入左侧示例";
    editor.value = localStorage.getItem(draftKey(lesson.id)) ?? lesson.draft;
    editor.setSelectionRange(0, 0);
    comparison.setReference(lesson.code);
    $("#save-status").textContent = "自动保存";
    $("#execution-count").textContent = "[ ]";
    $("#output-state").textContent = lesson.completed ? "已完成本段" : lesson.outputs.length ? "上次运行结果" : "尚未运行";
    renderOutputs(lesson.outputs);
    $("#feedback").hidden = true;
    if (lesson.completed) showFeedback("这段已通过。可以复习、再次运行，或继续下一段。", true);
    updateCursor();
    renderOutline();
  } catch (error) {
    toast(error.message);
  } finally {
    loading = false;
    updateNavigation();
    renderOutline();
  }
}

function updateNavigation() {
  const disabled = busy || loading || !lesson;
  $("#previous").disabled = disabled || lesson.index === 0;
  const nextIndex = lesson ? lesson.index + 1 : 0;
  $("#next").disabled = disabled || !(nextIndex < state.total ? state.lessons[nextIndex].unlocked : state.quiz_available);
  $("#next").innerHTML = nextIndex === state.total ? '阶段验收 <span>→</span>' : lesson?.kind === "example" && state.lessons[nextIndex]?.kind === "practice" ? '进入独立练习 <span>→</span>' : '下一段 <span>→</span>';
  $("#run").disabled = disabled;
  $("#restart").disabled = busy || loading;
  $("#logout").disabled = busy || loading;
  editor.readOnly = disabled;
  const completed = lesson && state.lessons[lesson.index]?.completed;
  $("#completion-note").textContent = completed ? "本段已完成，继续向前一步" : lesson?.kind === "practice" ? "通过运行与验收检查，解锁下一段" : "手敲完整代码并运行，解锁下一段";
  $("#kernel-dot").className = `kernel-dot${busy ? " busy" : state.kernel_ready ? " ready" : ""}`;
  $("#kernel-label").textContent = busy ? "Python 3 · 运行中…" : state.kernel_ready ? "Python 3 · 就绪" : "Python 3 · 等待运行";
  renderLearningPath();
}

function showFeedback(message, success, differences = []) {
  const feedback = $("#feedback");
  feedback.hidden = false;
  feedback.className = `feedback${success ? "" : " error"}`;
  feedback.textContent = `${success ? "✓ " : "↳ "}${message}`;
  comparison.setDifferences(differences);
  if (!differences.length) return;
  const legend = document.createElement("p");
  legend.className = "difference-legend";
  legend.textContent = "核对已忽略空白和字符串符号；下方保留原文，· 表示空格，⇥ 表示 Tab，↵ 表示换行。";
  feedback.append(legend);
  const visible = (value) => String(value).replace(/ /g, "·").replace(/\t/g, "⇥").replace(/\n/g, "↵\n");
  for (const item of differences) {
    const detail = document.createElement("div");
    detail.className = "difference-detail";
    const locate = document.createElement("button");
    locate.className = "difference-locate";
    locate.textContent = `第 ${item.line} 行${item.line !== item.reference_line ? `（示例第 ${item.reference_line} 行）` : ""} · 点击定位`;
    locate.addEventListener("click", () => {
      const lines = editor.value.split("\n");
      const offset = lines.slice(0, item.line - 1).reduce((sum, line) => sum + line.length + 1, 0) + item.column - 1;
      editor.focus({preventScroll: true});
      editor.setSelectionRange(offset, offset);
      updateCursor(true);
    });
    const explanation = document.createElement("p");
    explanation.textContent = item.message;
    const expected = document.createElement("pre");
    expected.textContent = `示例内容：${visible(item.expected)}`;
    const actual = document.createElement("pre");
    actual.textContent = `你的内容：${visible(item.actual)}`;
    detail.append(locate, explanation, expected, actual);
    feedback.append(detail);
  }
}

function queueSave(id, code, ownerId = state.user.id, token = state.token) {
  const storageKey = draftStorageKey(ownerId, id);
  saveQueue = saveQueue.catch(() => {}).then(async () => {
    try {
      await api("/api/draft", {method: "POST", headers: {"X-Lab-Token": token, "X-Lab-User": ownerId}, body: JSON.stringify({lesson_id: id, code})});
      if (localStorage.getItem(storageKey) === code) localStorage.removeItem(storageKey);
      if (lesson?.id === id && editor.value === code) $("#save-status").textContent = "已保存";
    } catch (error) {
      if (lesson?.id === id) $("#save-status").textContent = "暂存于浏览器";
      throw error;
    }
  });
  // 离线草稿保留在浏览器；错误不产生未处理的 Promise rejection。
  saveQueue.catch(() => {});
  return saveQueue;
}

async function flushDraft() {
  if (!lesson || !state?.user) return;
  clearTimeout(saveTimer);
  const local = localStorage.getItem(draftKey(lesson.id));
  if (local !== null) queueSave(lesson.id, local);
  try { await saveQueue; } catch { /* 草稿已在浏览器持久化。 */ }
}

function onEdit() {
  if (!lesson || !state?.user) return;
  comparison.setDifferences([]);
  $("#feedback").hidden = true;
  localStorage.setItem(draftKey(lesson.id), editor.value);
  $("#save-status").textContent = "保存中…";
  clearTimeout(saveTimer);
  const id = lesson.id, code = editor.value, ownerId = state.user.id, token = state.token;
  saveTimer = setTimeout(() => queueSave(id, code, ownerId, token), 400);
  updateCursor(true);
}

function updateCursor(reveal = false) {
  const before = editor.value.slice(0, editor.selectionStart);
  const lines = before.split("\n");
  $("#line-info").textContent = `第 ${lines.length} 行 · 第 ${lines.at(-1).length + 1} 列`;
  comparison.refresh({reveal: reveal === true});
}

function blockPaste(event) {
  event.preventDefault();
  toast("这一遍，交给你的双手。请逐行输入代码，粘贴与拖入已禁用。");
}
editor.addEventListener("paste", blockPaste);
editor.addEventListener("drop", blockPaste);
editor.addEventListener("dragover", (event) => event.preventDefault());
editor.addEventListener("beforeinput", (event) => {
  if (["insertFromPaste", "insertFromDrop", "insertFromYank", "insertReplacementText"].includes(event.inputType)) blockPaste(event);
});
editor.addEventListener("keydown", (event) => {
  if (((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "v") || (event.shiftKey && event.key === "Insert")) return blockPaste(event);
  if (event.isComposing || busy || loading) return;
  if (event.shiftKey && event.key === "Enter") {
    event.preventDefault();
    runCode();
  } else if (event.key === "Tab") {
    event.preventDefault();
    editor.setRangeText("    ", editor.selectionStart, editor.selectionEnd, "end");
    onEdit();
  } else if (event.key === "Enter") {
    event.preventDefault();
    const current = editor.value.slice(0, editor.selectionStart).split("\n").at(-1);
    const indent = current.match(/^ */)[0] + (current.trimEnd().endsWith(":") ? "    " : "");
    editor.setRangeText(`\n${indent}`, editor.selectionStart, editor.selectionEnd, "end");
    onEdit();
  }
});
editor.addEventListener("input", onEdit);
editor.addEventListener("keyup", () => updateCursor(true));
editor.addEventListener("click", () => updateCursor());
document.addEventListener("selectionchange", () => {
  if (document.activeElement === editor) updateCursor();
});

async function runCode() {
  if (busy || loading || !lesson) return;
  if (!editor.value.trim()) return toast("先手敲第一行代码，再点击运行。");
  busy = true;
  updateNavigation();
  $("#output-state").textContent = "正在执行…";
  $("#feedback").hidden = true;
  try {
    await flushDraft();
    const result = await api("/api/execute", {method: "POST", body: JSON.stringify({lesson_id: lesson.id, code: editor.value})});
    state = result.state;
    renderOutputs(result.outputs);
    if (!result.outputs.length) $("#output").innerHTML = `<pre>${result.ok ? "代码已执行完成，本段没有打印输出。" : "运行未完成。"}</pre>`;
    $("#execution-count").textContent = result.execution_count ? `[${result.execution_count}]` : "[ ]";
    $("#output-state").textContent = result.passed ? "检查通过" : result.ok ? "执行完成" : "需要修改";
    showFeedback(result.message, result.passed, result.differences);
    lesson.completed = state.lessons[lesson.index].completed;
    renderState();
    if (state.quiz_available) await renderQuiz();
  } catch (error) {
    showFeedback(error.message, false);
    $("#output-state").textContent = "连接或运行失败";
  } finally {
    busy = false;
    updateNavigation();
    renderOutline();
  }
}

async function renderQuiz() {
  if (!state.quiz_available) return;
  const data = await api(`/api/quiz?stage=${state.stage}`);
  const section = $("#quiz");
  section.hidden = false;
  section.innerHTML = '<span class="pill">STAGE CHECK</span><h2>把代码，变成自己的理解。</h2><p>完成三个小问题，确认第一阶段已掌握。</p>';
  const form = document.createElement("form");
  data.questions.forEach((question, index) => {
    const fieldset = document.createElement("fieldset");
    fieldset.className = "quiz-question";
    fieldset.innerHTML = `<legend>${index + 1}. ${escape(question.question)}</legend>`;
    question.options.forEach((option, optionIndex) => {
      const label = document.createElement("label");
      label.className = "quiz-option";
      label.innerHTML = `<input type="radio" name="q${index}" value="${optionIndex}" required>${escape(option)}`;
      fieldset.append(label);
    });
    form.append(fieldset);
  });
  const actions = document.createElement("div");
  actions.className = "quiz-actions";
  actions.innerHTML = '<button class="run-button" type="submit">完成阶段验收 <span>→</span></button><p role="status"></p>';
  form.append(actions);
  section.append(form);
  function showCompletion() {
    if (!state.stage_complete) return;
    actions.querySelector("p").textContent = state.next_stage ? `✓ 第 ${state.stage} 阶段已完成，第 ${state.next_stage} 阶段已解锁。` : "✓ 全部十个阶段已完成，可以回看任意章节。";
    if (state.next_stage && !actions.querySelector(".next-stage-button")) {
      const next = document.createElement("button");
      next.type = "button";
      next.className = "next-button next-stage-button";
      next.textContent = `进入第 ${state.next_stage} 阶段 →`;
      const nextNumber = state.next_stage;
      next.addEventListener("click", () => selectStage(nextNumber));
      actions.append(next);
    }
  }
  showCompletion();
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = actions.querySelector("button");
    button.disabled = true;
    loading = true;
    updateNavigation();
    try {
      const values = new FormData(form);
      const result = await api("/api/quiz", {method: "POST", body: JSON.stringify({stage: state.stage, answers: data.questions.map((_, index) => Number(values.get(`q${index}`)))})});
      actions.querySelector("p").textContent = result.message;
      if (result.state) { state = result.state; renderState(); showCompletion(); }
    } catch (error) { actions.querySelector("p").textContent = error.message; }
    finally { button.disabled = false; loading = false; updateNavigation(); }
  });
}

$("#run").addEventListener("click", runCode);
$("#previous").addEventListener("click", () => selectLesson(lesson.index - 1));
$("#next").addEventListener("click", () => {
  if (lesson.index + 1 === state.total) $("#quiz").scrollIntoView({behavior: "smooth", block: "start"});
  else selectLesson(lesson.index + 1);
});
$("#outline-toggle").addEventListener("click", () => {
  $("#outline").hidden = !$("#outline").hidden;
  $("#outline-toggle").setAttribute("aria-expanded", String(!$("#outline").hidden));
});
$("#restart").addEventListener("click", async () => {
  if (busy || loading) return;
  try {
    const result = await api("/api/kernel/restart", {method: "POST"});
    state.kernel_ready = false;
    $("#execution-count").textContent = "[ ]";
    updateNavigation();
    toast(result.message);
  } catch (error) { toast(error.message); }
});

async function loadCourse() {
  try {
    state = await api("/api/state");
    renderState();
    await selectLesson(Math.min(state.frontier, state.total - 1));
    if (state.quiz_available) await renderQuiz();
  } catch (error) {
    $("#lesson-title").textContent = "学习服务尚未连接";
    $("#lesson-description").textContent = "请在 IDE 中运行项目根目录的 main.py，再刷新此页面。";
    toast(error.message);
  }
}

function clearCourse() {
  clearTimeout(saveTimer);
  state = null;
  lesson = null;
  editor.value = "";
  editor.readOnly = true;
  $("#reference-code").replaceChildren();
  comparison.setReference("");
  comparison.refresh();
  $("#quiz").hidden = true;
  $("#quiz").replaceChildren();
  $("#outline").hidden = true;
  $("#outline-toggle").setAttribute("aria-expanded", "false");
  $("#current-username").textContent = "";
  $("#learning-app").hidden = true;
  $("#feedback").hidden = true;
  renderOutputs();
}

$("#logout").addEventListener("click", async () => {
  if (busy || loading) return;
  $("#logout").disabled = true;
  try {
    await flushDraft();
    await auth.signOut();
    toast("已退出登录，你的学习进度已保留。");
  } catch (error) { toast(error.message); }
  finally { $("#logout").disabled = false; }
});

// 同一浏览器的其他标签页切换账号时，刷新当前空间，避免继续编辑旧账号的数据。
window.addEventListener("focus", async () => {
  if (!state?.user || busy || loading) return;
  try {
    const response = await fetch("/api/auth/session", {credentials: "same-origin"});
    const session = await response.json();
    if (!session.authenticated || session.user.id !== state.user.id || session.token !== state.token) window.location.reload();
  } catch { /* 暂时断线时保留当前账号的浏览器草稿。 */ }
});

auth.refresh().catch((error) => {
  $("#auth-error").textContent = `学习服务连接失败：${error.message}。请运行 main.py 后刷新页面。`;
  $("#auth-error").hidden = false;
});
