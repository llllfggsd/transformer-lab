const $ = (selector) => document.querySelector(selector);

export const draftStorageKey = (userId, lessonId) => `transformer-lab:user:${userId}:draft:${lessonId}`;

export function migrateLegacyDrafts(storage, userId) {
  const prefix = "transformer-lab:draft:";
  const keys = Array.from({length: storage.length}, (_, index) => storage.key(index)).filter((key) => key?.startsWith(prefix));
  for (const key of keys) {
    const target = draftStorageKey(userId, key.slice(prefix.length));
    if (storage.getItem(target) === null) storage.setItem(target, storage.getItem(key));
    storage.removeItem(key);
  }
}

export function createAuthScreen({onAuthenticated, onSignedOut}) {
  let session = null, mode = "login", pending = false;
  const form = $("#auth-form");
  const error = $("#auth-error");

  function setMode(next) {
    if (pending) return;
    mode = next;
    const register = mode === "register";
    $("#auth-login-tab").setAttribute("aria-pressed", String(!register));
    $("#auth-register-tab").setAttribute("aria-pressed", String(register));
    $("#auth-title").textContent = register ? "创建你的学习账号" : "登录你的学习空间";
    $("#auth-subtitle").textContent = register ? "为自己留一份进度，从第一行代码开始。" : "登录后继续上次的学习，进度随账号保存。";
    $("#auth-submit").innerHTML = register ? '注册并开始学习 <span>→</span>' : '登录并继续学习 <span>→</span>';
    $("#auth-confirm-group").hidden = !register;
    $("#auth-confirm").required = register;
    $("#auth-register-help").hidden = !register;
    $("#auth-password").autocomplete = register ? "new-password" : "current-password";
    const oldDrafts = Object.keys(localStorage).some((key) => key.startsWith("transformer-lab:draft:"));
    $("#auth-legacy-group").hidden = !(register && session?.first_account && (session.legacy_available || oldDrafts));
    error.hidden = true;
  }

  async function request(path, body) {
    const response = await fetch(path, {
      method: body === undefined ? "GET" : "POST",
      credentials: "same-origin",
      headers: {"Content-Type": "application/json", ...(session ? {"X-Lab-Token": session.token} : {}), ...(session?.user ? {"X-Lab-User": session.user.id} : {})},
      ...(body === undefined ? {} : {body: JSON.stringify(body)}),
    });
    const data = await response.json();
    if (!response.ok) {
      const failure = new Error(typeof data.detail === "string" ? data.detail : "请求未完成，请检查输入后重试。");
      failure.status = response.status;
      throw failure;
    }
    return data;
  }

  async function refresh() {
    session = await request("/api/auth/session");
    document.body.classList.toggle("signed-out", !session.authenticated);
    $("#auth-screen").hidden = session.authenticated;
    $("#learning-app").hidden = !session.authenticated;
    if (session.authenticated) {
      $("#current-username").textContent = session.user.username;
      $("#current-username").title = `当前账号：${session.user.username}`;
      await onAuthenticated(session);
    } else {
      $("#auth-import-legacy").checked = true;
      setMode("login");
    }
    return session;
  }

  $("#auth-login-tab").addEventListener("click", () => setMode("login"));
  $("#auth-register-tab").addEventListener("click", () => setMode("register"));
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (pending) return;
    error.hidden = true;
    if (mode === "register" && $("#auth-password").value !== $("#auth-confirm").value) {
      error.textContent = "两次输入的密码不一致。";
      error.hidden = false;
      return;
    }
    pending = true;
    $("#auth-submit").disabled = true;
    $("#auth-login-tab").disabled = $("#auth-register-tab").disabled = true;
    try {
      const result = await request(`/api/auth/${mode}`, {
        username: $("#auth-username").value,
        password: $("#auth-password").value,
        import_legacy: mode === "register" && !$("#auth-legacy-group").hidden && $("#auth-import-legacy").checked,
      });
      session = result;
      if (result.legacy_claimed) migrateLegacyDrafts(localStorage, result.user.id);
      $("#auth-password").value = $("#auth-confirm").value = "";
      await refresh();
    } catch (failure) {
      error.textContent = failure.message;
      error.hidden = false;
    } finally {
      pending = false;
      $("#auth-submit").disabled = false;
      $("#auth-login-tab").disabled = $("#auth-register-tab").disabled = false;
    }
  });

  async function signOut() {
    try { await request("/api/auth/logout", {}); }
    catch (error) { if (error.status !== 401) throw error; }
    session = null;
    onSignedOut();
    form.reset();
    await refresh();
  }

  return {refresh, signOut, getSession: () => session};
}
