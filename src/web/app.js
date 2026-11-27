// Shared helpers for the support workspace screens. Every value from a ticket is escaped before it is shown.
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

async function api(path, opts = {}) {
  const r = await fetch(path, {headers: {"Content-Type": "application/json"}, credentials: "same-origin", ...opts});
  if (r.status === 401) { location.href = "/"; throw new Error("please sign in"); }
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || r.statusText);
  return body;
}
const post = (path, body) => api(path, {method: "POST", body: JSON.stringify(body || {})});

function statusChip(handled) {
  const cls = /ready/i.test(handled) ? "st-ready" : /agent/i.test(handled) ? "st-agent" : /resolved|answered/i.test(handled) ? "st-done" : "";
  return `<span class="chip ${cls}">${esc(handled)}</span>`;
}

async function header(roleLabel) {
  const me = await api("/staff/me");
  $("who").textContent = `${me.name} · ${me.team || roleLabel}`;
  $("signout").onclick = async () => { await post("/staff/logout"); location.href = "/"; };
  return me;
}
