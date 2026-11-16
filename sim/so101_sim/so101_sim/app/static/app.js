// Browser side of the SO-101 simulator. Everything here is a thin remote control:
// key presses and button clicks go to /api/cmd, the page redraws from /api/status.
"use strict";

const $ = (id) => document.getElementById(id);
const send = (cmd) =>
  fetch("/api/cmd", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(cmd) })
    .catch(() => {});

let status = null;
let pendingSub = null;       // leader/policy tab opened but not started yet
let lastMode = "", lastSeed = null;

// ---------------------------------------------------------------- keyboard teleop

const KEYMAP = {
  KeyW: "fwd", KeyS: "back", KeyA: "left", KeyD: "right", KeyR: "up", KeyF: "down",
  KeyT: "pitch_up", KeyG: "pitch_down", KeyQ: "roll_left", KeyE: "roll_right",
  ShiftLeft: "fine", ShiftRight: "fine",
};
const held = new Set();
const typing = () => ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName);
const sendKeys = () => send({ type: "keys", keys: [...held] });

document.addEventListener("keydown", (e) => {
  if (typing()) return;
  const a = KEYMAP[e.code];
  if (a) {
    e.preventDefault();
    if (!held.has(a)) { held.add(a); sendKeys(); }
    return;
  }
  if (e.repeat) return;
  if (e.code === "Space") { e.preventDefault(); send({ type: "toggle_grip" }); }
  else if (e.code === "KeyN") send({ type: "reset" });
  else if (e.code === "ArrowRight") { e.preventDefault(); send({ type: "save" }); }
  else if (e.code === "ArrowLeft") { e.preventDefault(); send({ type: "discard" }); }
  else if (e.code === "Enter" && document.activeElement?.tagName !== "BUTTON") send({ type: "record" });
});
document.addEventListener("keyup", (e) => {
  const a = KEYMAP[e.code];
  if (a && held.delete(a)) sendKeys();
});
window.addEventListener("blur", () => { held.clear(); sendKeys(); });
// Heartbeat: the simulator drops held keys it has not heard about for 0.6 s.
setInterval(() => { if (held.size) sendKeys(); }, 200);

// ---------------------------------------------------------------- 3D view orbit

const wrap = $("view-wrap");
let drag = null, acc = { dx: 0, dy: 0 }, zoomAcc = 0;
wrap.addEventListener("mousedown", (e) => { drag = { x: e.clientX, y: e.clientY }; wrap.focus(); });
window.addEventListener("mouseup", () => { drag = null; });
window.addEventListener("mousemove", (e) => {
  if (!drag) return;
  acc.dx += e.clientX - drag.x; acc.dy += e.clientY - drag.y;
  drag = { x: e.clientX, y: e.clientY };
});
wrap.addEventListener("wheel", (e) => { e.preventDefault(); zoomAcc += e.deltaY; }, { passive: false });
setInterval(() => {
  if (acc.dx || acc.dy) { send({ type: "orbit", ...acc }); acc = { dx: 0, dy: 0 }; }
  if (zoomAcc) { send({ type: "zoom", d: zoomAcc }); zoomAcc = 0; }
}, 40);
$("view-reset").onclick = (e) => { e.preventDefault(); send({ type: "view_reset" }); };

// ---------------------------------------------------------------- controls

document.querySelectorAll("#modes button").forEach((b) => {
  b.onclick = () => {
    const mode = b.dataset.mode;
    if (mode === "leader" || mode === "policy") { pendingSub = mode; showSub(mode); return; }  // settings first
    pendingSub = null;
    send({ type: "mode", mode });
  };
});
function showSub(mode) {
  document.querySelectorAll(".sub").forEach((s) => (s.hidden = s.dataset.for !== mode));
  document.querySelectorAll("#modes button").forEach((b) => b.classList.toggle("on", b.dataset.mode === mode));
}
$("leader-go").onclick = () =>
  send({ type: "mode", mode: "leader", leader_port: $("leader-port").value.trim(), leader_id: $("leader-id").value.trim() });
$("expert-go").onclick = () => send({ type: "mode", mode: "expert" });
$("policy-go").onclick = () =>
  send({
    type: "mode", mode: "policy", policy: $("policy-path").value.trim(),
    latency_ms: Number($("policy-latency").value || 0), policy_mode: $("policy-mode").value,
    threshold: Number($("policy-threshold").value || 0.5), num_steps: $("policy-steps").value || null,
  });

$("btn-new").onclick = () => send({ type: "reset" });
$("btn-pause").onclick = () => send({ type: "pause", value: !(status && status.paused) });
$("btn-seed").onclick = () => send({ type: "reset", seed: Number($("seed").value) });
$("randomize").onchange = (e) => send({ type: "randomize", level: e.target.value });
$("ds-open").onclick = () => send({ type: "dataset", name: $("ds-name").value, task: $("ds-task").value });
$("btn-record").onclick = () => send({ type: "record" });
$("btn-save").onclick = () => send({ type: "save" });
$("btn-discard").onclick = () => send({ type: "discard" });
$("batch-go").onclick = () => send({ type: "batch", n: Number($("batch-n").value || 1) });
$("batch-stop").onclick = () => send({ type: "stop_batch" });
$("fast").onchange = (e) => send({ type: "fast", value: e.target.checked });
$("btn-quit").onclick = () => {
  if (confirm("Stop the simulator? Saved episodes stay on disk; an unsaved episode is discarded.")) {
    send({ type: "quit" }).then(() => setTimeout(() => ($("gone").hidden = false), 500));
  }
};

// ---------------------------------------------------------------- sliders

function buildSliders(st) {
  const box = $("sliders");
  box.innerHTML = "";
  st.joint_names.forEach((name, i) => {
    const [lo, hi] = st.limits[i];
    const row = document.createElement("div");
    row.className = "slider";
    row.innerHTML = `<span>${name.replace(".pos", "")}</span>
      <input type="range" min="${lo}" max="${hi}" step="0.5" value="${st.state[i]}">
      <span>${st.state[i].toFixed(1)}</span>`;
    const input = row.querySelector("input");
    input.addEventListener("input", () => {
      row.lastElementChild.textContent = Number(input.value).toFixed(1);
      send({ type: "joints", values: [...box.querySelectorAll("input")].map((x) => Number(x.value)) });
    });
    box.appendChild(row);
  });
}

// ---------------------------------------------------------------- status

let firstStatus = true, misses = 0;
async function poll() {
  try {
    const r = await fetch("/api/status", { cache: "no-store" });
    status = await r.json();
    misses = 0;
    $("gone").hidden = true;
    render(status);
  } catch {
    if (++misses > 25) $("gone").hidden = false;     // about 4 s without an answer
  }
  setTimeout(poll, 150);
}

function render(st) {
  if (!st || !st.mode) return;
  if (firstStatus) {
    firstStatus = false;
    $("ds-name").value = st.dataset.name || "sim_cube_box";
    $("ds-task").value = st.task;
    $("seed").value = st.seed;
    $("randomize").value = st.randomize;
    if (st.leader.port) $("leader-port").value = st.leader.port;
    $("leader-id").value = st.leader.id || "";
    if (st.policy.spec) $("policy-path").value = st.policy.spec;
    $("policy-latency").value = st.policy.latency_ms;
    $("policy-mode").value = st.policy.mode;
    $("policy-threshold").value = st.policy.threshold;
    $("no-lerobot").hidden = st.lerobot;
  }
  $("task").textContent = `“${st.task}”`;
  $("chip-mode").textContent = st.mode;
  $("chip-time").textContent = `${st.sim_time.toFixed(1)} s · seed ${st.seed}`;
  $("chip-fps").textContent = `${st.fps_actual.toFixed(0)}/${st.fps_target} Hz`;
  $("chip-rec").hidden = !st.recording;
  $("chip-success").hidden = !st.success;
  $("paused-badge").hidden = !st.paused;
  $("btn-pause").textContent = st.paused ? "Resume" : "Pause";

  // Mode tabs follow the simulator, except while leader/policy settings are being filled in.
  if (pendingSub && st.mode === pendingSub) pendingSub = null;
  if (st.mode !== lastMode && lastMode !== "") pendingSub = null;
  showSub(pendingSub || (st.mode === "idle" ? "keyboard" : st.mode));
  if (st.mode === "sliders" && (lastMode !== "sliders" || lastSeed !== st.seed)) buildSliders(st);
  lastMode = st.mode;
  lastSeed = st.seed;

  const tb = $("joints").tBodies[0];
  if (tb.rows.length !== st.joint_names.length) {
    tb.innerHTML = st.joint_names.map((n) => `<tr><td>${n.replace(".pos", "")}</td><td></td><td></td></tr>`).join("");
  }
  st.joint_names.forEach((_, i) => {
    tb.rows[i].cells[1].textContent = st.state[i].toFixed(1);
    tb.rows[i].cells[2].textContent = st.action[i].toFixed(1);
  });

  if (st.keyboard) {
    const [x, y, z] = st.keyboard.pos;
    $("kb-readout").textContent =
      `jaws at x ${(x * 100).toFixed(1)} · y ${(y * 100).toFixed(1)} · z ${(z * 100).toFixed(1)} cm, ` +
      `tilt ${st.keyboard.pitch_deg.toFixed(0)}° below horizontal, gripper ${st.keyboard.closed ? "closed" : "open"}`;
  }

  const p = st.policy;
  if (st.mode === "policy") {
    $("policy-readout").textContent = `${p.inferences} chunks requested, ${p.stalls} stalled steps` +
      (p.waiting ? " · waiting for a chunk" : "");
  }

  const ds = st.dataset;
  $("ds-readout").textContent = ds.name
    ? `${ds.name}: ${ds.episodes} episode${ds.episodes === 1 ? "" : "s"} saved` + (ds.root ? ` · ${ds.root}` : "")
    : "";
  $("btn-record").disabled = !st.lerobot || st.recording;
  $("btn-save").disabled = !st.recording;
  $("btn-discard").disabled = !st.recording;
  $("batch-go").disabled = !st.lerobot || st.batch_left > 0;
  $("batch-stop").disabled = st.batch_left === 0;
  $("ep-bar").style.width = st.recording ? `${(100 * st.ep_frames) / st.ep_limit}%` : "0";
  $("ep-readout").textContent = st.recording
    ? `recording: ${(st.ep_frames / st.fps_target).toFixed(1)} s of ${(st.ep_limit / st.fps_target).toFixed(0)} s`
    : "Recording starts from a fresh scene and saves itself at the time limit, like lerobot-record.";
  $("batch-readout").textContent = st.batch_left > 0
    ? `${st.batch_left} to go · ${st.batch_stats[0]} saved of ${st.batch_stats[1]} tries`
    : st.batch_stats[1] ? `last batch: ${st.batch_stats[0]} saved of ${st.batch_stats[1]} tries` : "";
  $("fast").checked = st.fast;

  const log = $("log");
  const text = st.log.join("\n");
  if (log.textContent !== text) { log.textContent = text; log.scrollTop = log.scrollHeight; }
}

poll();
