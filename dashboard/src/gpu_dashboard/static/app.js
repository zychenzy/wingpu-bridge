const state = {
  data: null,
  evictArmed: false,
  busy: false,
  logSource: "llama",
};

const $ = (id) => document.getElementById(id);

function el(tag, attrs, children) {
  const node = document.createElement(tag);
  Object.entries(attrs || {}).forEach(([key, value]) => {
    if (value == null || value === false) return;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "html") node.innerHTML = value;
    else node.setAttribute(key, String(value));
  });
  (children || []).forEach((child) => {
    if (child) node.append(child);
  });
  return node;
}

function clear(node) {
  node.replaceChildren();
}

function fmtMb(value) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return Number(value).toLocaleString("en-US");
}

function fmtDur(seconds) {
  if (seconds == null) return "—";
  const s = Math.max(0, Math.round(Number(seconds)));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${String(r).padStart(2, "0")}s`;
  return `${r}s`;
}

function clockText(iso) {
  if (!iso) {
    return new Date().toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }
  const match = String(iso).match(/T(\d{2}:\d{2}:\d{2})/);
  return match ? match[1] : String(iso);
}

function shortTime(iso) {
  const match = String(iso || "").match(/T(\d{2}:\d{2}:\d{2})/);
  return match ? match[1] : "—";
}

function pill(text, kind) {
  return el("span", { class: `pill${kind ? ` pill-${kind}` : ""}`, text });
}

function button(label, className, onClick, disabled) {
  const node = el("button", { class: `btn ${className}`, type: "button", text: label });
  node.disabled = Boolean(disabled || state.busy);
  node.addEventListener("click", onClick);
  return node;
}

function toast(message) {
  const node = $("toast");
  node.hidden = false;
  node.textContent = message;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => {
    node.hidden = true;
  }, 3200);
}

async function api(path, options) {
  const response = await fetch(path, options);
  const payload = await response.json().catch(() => ({ ok: false, error: "bad response" }));
  if (!response.ok || payload.ok === false) {
    throw new Error(payload.error || `request failed (${response.status})`);
  }
  return payload;
}

async function act(name, body) {
  state.busy = true;
  paint();
  try {
    const payload = await api(`/api/actions/${name}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    const job = payload.job || {};
    if (job.status === "running") {
      toast(`${name} started`);
      await waitJob(job.id);
    } else if (job.status === "error") {
      toast(job.error || `${name} failed`);
    } else {
      toast(job.output || `${name} ok`);
    }
  } catch (error) {
    toast(error.message);
  } finally {
    state.busy = false;
    state.evictArmed = false;
    await refresh();
  }
}

async function waitJob(id) {
  for (let i = 0; i < 60; i += 1) {
    await new Promise((resolve) => setTimeout(resolve, 1000));
    const payload = await api(`/api/jobs/${id}`);
    if (payload.job.status === "ok") {
      toast(payload.job.output || "done");
      return;
    }
    if (payload.job.status === "error") {
      toast(payload.job.error || "failed");
      return;
    }
  }
  toast("still running");
}

function paintHeader() {
  const data = state.data;
  $("sample").hidden = !data.mock;
  $("host-chip").textContent = data.host || "win-gpu";
  const clock = $("clock");
  clock.textContent = clockText(data.sampled_at);
  clock.dateTime = data.sampled_at || "";
  const theme = document.documentElement.dataset.theme || "dark";
  $("theme").textContent = theme === "dark" ? "Light" : "Dark";
}

function paintGpu() {
  const gpu = state.data.gpu || {};
  const used = Number(gpu.mem_used_mb);
  const total = Number(gpu.mem_total_mb);
  const pct = total ? Math.max(0, Math.min(100, (used / total) * 100)) : 0;
  const util = Math.max(0, Math.min(100, Number(gpu.util_pct) || 0));
  const card = $("gpu-card");
  clear(card);
  card.append(
    el("div", { class: "kicker", text: "GPU" }),
    el("h2", { id: "gpu-title", text: gpu.name || "NVIDIA GPU" }),
    el("p", { class: "device", text: "nvidia-smi via the broker" }),
    el("div", { class: "figure" }, [
      el("b", { text: fmtMb(gpu.mem_used_mb) }),
      el("span", { text: `/ ${fmtMb(gpu.mem_total_mb)} MB` }),
    ]),
  );
  card.append(meter("VRAM", `${Math.round(pct)}%`, pct, pct >= 92));
  card.append(meter("Util", `${util}%`, util, false));
  const procs = gpu.processes || [];
  if (procs.length) {
    const table = el("table", { class: "procs" });
    table.append(el("thead", {}, [
      el("tr", {}, [el("th", { text: "Process" }), el("th", { text: "PID" }), el("th", { text: "VRAM" })]),
    ]));
    const body = el("tbody");
    procs.forEach((proc) => {
      body.append(el("tr", {}, [
        el("td", { text: proc.name || "process" }),
        el("td", { class: "mono", text: String(proc.pid ?? "—") }),
        el("td", { text: `${fmtMb(proc.vram_mb)} MB` }),
      ]));
    });
    table.append(body);
    card.append(table);
  }
}

function meter(label, value, pct, warn) {
  const fill = el("div", { class: `meter-fill${warn ? " warn" : ""}` });
  fill.style.width = `${pct}%`;
  return el("div", { class: "meter" }, [
    el("div", { class: "meter-label" }, [el("span", { text: label }), el("span", { text: value })]),
    el("div", { class: "meter-track" }, [fill]),
  ]);
}

function paintLease() {
  const holder = state.data.holder;
  const broker = state.data.broker || {};
  const card = $("lease-card");
  clear(card);
  const pills = el("div", { class: "pills" });
  if (!broker.reachable) pills.append(pill("Unreachable", "bad"));
  else if (!holder) pills.append(pill("Free", "idle"));
  else {
    pills.append(pill(holder.alive ? "Alive" : "Not alive", holder.alive ? "ok" : "bad"));
    pills.append(pill(holder.busy ? "Busy" : "Not busy", holder.busy ? "warn" : "ok"));
  }
  card.append(
    el("div", { class: "card-head" }, [
      el("div", {}, [el("div", { class: "kicker", text: "Lease" }), el("h2", { id: "lease-title", text: holder ? holder.runtime_id : "No holder" })]),
      pills,
    ]),
  );
  const rows = el("dl", { class: "kv" });
  const add = (k, v, mono) => {
    rows.append(el("dt", { text: k }), el("dd", { class: mono ? "mono" : "", text: v || "—" }));
  };
  add("Label", holder && holder.label, true);
  add("Since", holder && shortTime(holder.acquired_at));
  add("Touch", holder && holder.last_active_at ? shortTime(holder.last_active_at) : "never");
  add("State", holder && (holder.state || "active"));
  add("Broker", broker.version ? `${broker.version} · protocol ${broker.protocol}` : "not installed", true);
  card.append(rows);
  if (state.evictArmed) {
    const box = el("div", { class: "confirm" });
    box.append(el("p", { text: `Evict ${holder ? holder.runtime_id : "the holder"}? The broker stops that process and waits for VRAM to drain.` }));
    const row = el("div", { class: "btn-row" });
    row.append(
      button("Evict now", "btn-danger", () => act("evict", { confirm: true })),
      button("Cancel", "btn-ghost", () => {
        state.evictArmed = false;
        paintLease();
      }),
    );
    box.append(row);
    card.append(box);
  } else {
    const row = el("div", { class: "btn-row" });
    row.append(
      button("Evict", "btn-danger", () => {
        state.evictArmed = true;
        paintLease();
      }, !holder),
      button("Install broker", "btn-ghost", () => act("broker-install")),
    );
    card.append(row);
  }
}

function paintIdle() {
  const wd = state.data.watchdog || {};
  const gw = state.data.gateway || {};
  const card = $("idle-card");
  clear(card);
  card.append(el("div", { class: "kicker", text: "Idle offload" }), el("h2", { id: "idle-title", text: "Watchdog and gateway" }));
  const split = el("div", { class: "split" });
  split.append(timerCard(
    "Watchdog",
    wd.running ? `pid ${wd.pid}` : "stopped",
    wd.blocked_reason === "busy" ? "busy, holding" : (wd.next_evict_in_s != null ? fmtDur(wd.next_evict_in_s) : "—"),
    wd.idle_timeout_s != null ? `limit ${fmtDur(wd.idle_timeout_s)}` : "",
  ));
  split.append(timerCard(
    "Gateway",
    gw.up ? (gw.idle_status || "up") : "down",
    gw.seconds_until_offload != null ? fmtDur(gw.seconds_until_offload) : "—",
    gw.idle_timeout_seconds != null ? `limit ${fmtDur(gw.idle_timeout_seconds)}` : "",
  ));
  card.append(split);
  const row = el("div", { class: "btn-row" });
  row.append(
    button(wd.running ? "Stop watchdog" : "Start watchdog", "btn-ghost", () => act(wd.running ? "watchdog-stop" : "watchdog-start")),
  );
  card.append(row);
}

function timerCard(title, detail, value, note) {
  return el("article", {}, [
    el("h3", { text: title }),
    el("p", { text: detail }),
    el("strong", { text: value }),
    note ? el("p", { text: note }) : null,
  ]);
}

function paintEvents() {
  const card = $("events-card");
  clear(card);
  card.append(el("div", { class: "kicker", text: "Broker events" }), el("h2", { id: "events-title", text: "Recent" }));
  const rows = (state.data.events || []).slice().reverse();
  if (!rows.length) {
    card.append(el("p", { class: "empty", text: "No events yet." }));
    return;
  }
  const list = el("div", { class: "events" });
  rows.slice(0, 8).forEach((row) => {
    list.append(el("div", { class: "event" }, [
      el("time", { text: shortTime(row.ts) }),
      el("b", { text: row.event || "event" }),
      el("span", { text: row.runtime_id || "" }),
    ]));
  });
  card.append(list);
}

function field(label, control) {
  return el("div", { class: "field" }, [el("label", { text: label }), control]);
}

function select(id, options, current) {
  const node = el("select", { id });
  options.forEach((option) => {
    const item = el("option", { value: option, text: option });
    if (option === current) item.selected = true;
    node.append(item);
  });
  return node;
}

function editing(root) {
  const active = document.activeElement;
  if (!active || !root.contains(active)) return false;
  return active.tagName === "INPUT" || active.tagName === "SELECT" || active.tagName === "TEXTAREA";
}

function paintLlama() {
  const gw = state.data.gateway || {};
  const svc = (state.data.services || {}).llama || {};
  const catalog = state.data.catalog || { models: [], lanes: [], cache_types: [] };
  const card = $("llama-card");
  if (editing(card)) return;
  clear(card);
  card.classList.toggle("card-hold", Boolean(svc.holds_gpu));
  const pills = el("div", { class: "pills" });
  pills.append(pill(svc.state || "unknown", svc.state === "loaded" ? "ok" : "idle"));
  if (svc.holds_gpu) pills.append(pill("Holds GPU", "ok"));
  card.append(el("div", { class: "card-head" }, [
    el("div", {}, [
      el("div", { class: "kicker", text: "Workload" }),
      el("h2", { id: "llama-title", text: "llama.cpp" }),
      el("p", { class: "command", text: svc.command || "wingpu" }),
    ]),
    pills,
  ]));
  card.append(el("p", { class: "blurb", text: `Served model ${gw.served_model || "qwen-local"} · context ${gw.context_length || "—"} · ${gw.active_requests ?? 0} active` }));
  const models = (catalog.models || []).map((item) => item.name);
  const fields = el("div", { class: "fields" });
  fields.append(
    field("Model", select("model", models, gw.selected_model)),
    field("Lane", select("lane", catalog.lanes || [], gw.selected_runtime)),
    field("KV K", select("cache-k", catalog.cache_types || [], gw.cache_k)),
    field("KV V", select("cache-v", catalog.cache_types || [], gw.cache_v)),
  );
  card.append(fields);
  const row = el("div", { class: "btn-row" });
  row.append(
    button("Apply and restart", "btn-primary", () => act("llama-apply", {
      model: $("model").value,
      lane: $("lane").value,
      cache_k: $("cache-k").value,
      cache_v: $("cache-v").value,
    })),
    button("Offload", "btn-ghost", () => act("llama-offload")),
    button("Stop", "btn-ghost", () => act("llama-stop")),
    button("Restart gateway", "btn-ghost", () => act("gateway-restart")),
  );
  card.append(row);
}

function serviceCard(key, spec) {
  const card = el("article", { class: `card svc${spec.holds_gpu ? " card-hold" : ""}` });
  const pills = el("div", { class: "pills" });
  const kind = spec.state === "running" || spec.state === "loaded" ? "ok" : "idle";
  pills.append(pill(spec.state || "unknown", kind));
  card.append(el("div", { class: "svc-head" }, [
    el("div", {}, [
      el("h2", { text: spec.title || key }),
      el("p", { class: "command", text: spec.command || "" }),
    ]),
    pills,
  ]));
  card.append(el("p", { class: "blurb", text: spec.blurb || "" }));
  if (key === "asr") {
    const engines = ["whisper", "qwen3", "both"];
    card.append(el("div", { class: "field-block" }, [field("Engine", select("asr-engine", engines, spec.engine || "whisper"))]));
  }
  if (key === "mineru") {
    const input = el("input", { id: "mineru-pdf", type: "text", placeholder: "/Users/you/scan.pdf" });
    card.append(el("div", { class: "field-block" }, [field("PDF", input)]));
  }
  const row = el("div", { class: "btn-row" });
  if (key === "mineru") {
    row.append(
      button("Run PDF", "btn-primary", () => act("mineru-run", { pdf: $("mineru-pdf").value })),
      button("Status", "btn-ghost", () => act("mineru-status")),
      button("Logs", "btn-ghost", () => showLog("mineru")),
    );
  } else {
    const prefix = key;
    row.append(
      button("Start", "btn-primary", () => act(`${prefix}-start`, key === "asr" ? { engine: $("asr-engine").value } : {})),
      button("Stop", "btn-ghost", () => act(`${prefix}-stop`, key === "asr" ? { engine: $("asr-engine").value } : {})),
      button("Restart", "btn-ghost", () => act(`${prefix}-restart`, key === "asr" ? { engine: $("asr-engine").value } : {})),
      button("Logs", "btn-ghost", () => showLog(key)),
    );
  }
  card.append(row);
  return card;
}

function paintServices() {
  const grid = $("service-grid");
  if (editing(grid)) return;
  clear(grid);
  ["ocr", "locate", "asr", "mineru"].forEach((key) => {
    const spec = (state.data.services || {})[key];
    if (spec) grid.append(serviceCard(key, spec));
  });
}

async function showLog(source) {
  state.logSource = source;
  const selectNode = $("log-source");
  if (selectNode) selectNode.value = source;
  await loadLog();
  $("log-card").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function loadLog() {
  const view = $("log-view");
  if (!view) return;
  try {
    const payload = await api(`/api/logs/${state.logSource}?n=80`);
    view.textContent = payload.text || "(empty)";
  } catch (error) {
    view.textContent = error.message;
  }
}

function paintLogs() {
  const card = $("log-card");
  if (editing(card)) return;
  clear(card);
  const sources = ["llama", "gateway", "watchdog", "ocr", "locate", "asr", "mineru"];
  const picker = select("log-source", sources, state.logSource);
  picker.addEventListener("change", () => {
    state.logSource = picker.value;
    loadLog();
  });
  card.append(
    el("div", { class: "logbar" }, [
      el("div", {}, [el("div", { class: "kicker", text: "Logs" }), el("h2", { id: "log-title", text: "Output" })]),
      picker,
    ]),
    el("pre", { id: "log-view", class: "logview", text: "Loading…" }),
  );
  loadLog();
}

function paint() {
  if (!state.data) return;
  paintHeader();
  paintGpu();
  paintLease();
  paintIdle();
  paintEvents();
  paintLlama();
  paintServices();
  if (!$("log-view")) paintLogs();
}

async function refresh() {
  state.data = await api("/api/status");
  paint();
}

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("gpu-dashboard-theme", theme);
  if (state.data) paintHeader();
}

$("theme").addEventListener("click", () => {
  const current = document.documentElement.dataset.theme || "dark";
  setTheme(current === "dark" ? "light" : "dark");
});

refresh().catch((error) => toast(error.message));
setInterval(() => {
  if (state.busy || state.evictArmed) return;
  refresh().catch(() => {});
}, 5000);
