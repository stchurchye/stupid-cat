// el() and getJSON() live in util.js (loaded first).

function formatTime(iso) {
  if (!iso) return "—";
  try {
    const d = new Date(iso);
    return d.toLocaleString("zh-CN", {
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
  } catch {
    return iso;
  }
}

function formatDuration(sec) {
  if (sec == null || sec === "") return "—";
  const n = Number(sec);
  if (Number.isNaN(n)) return String(sec);
  return `${n} 秒`;
}

function showToast(message, kind = "error") {
  let host = document.getElementById("toast-host");
  if (!host) {
    host = document.createElement("div");
    host.id = "toast-host";
    host.style.cssText =
      "position:fixed;top:12px;right:12px;z-index:9999;display:flex;flex-direction:column;gap:6px";
    document.body.appendChild(host);
  }
  const t = el("div", message);
  t.style.cssText =
    "padding:8px 12px;border-radius:6px;color:#fff;font-size:0.9em;" +
    "box-shadow:0 2px 6px rgba(0,0,0,.2);background:" +
    (kind === "error" ? "#dc2626" : "#16a34a");
  host.appendChild(t);
  setTimeout(() => t.remove(), 4000);
}

function renderTableMessage(html, { retry = false } = {}) {
  const tbody = document.querySelector("#visits tbody");
  tbody.innerHTML = "";
  const tr = document.createElement("tr");
  const td = el("td");
  td.colSpan = 7;
  td.className = "empty-state";
  td.innerHTML = html;
  if (retry) {
    const btn = el("button", "重试");
    btn.className = "text-link";
    btn.style.marginTop = "6px";
    btn.addEventListener("click", loadVisits);
    td.appendChild(document.createElement("br"));
    td.appendChild(btn);
  }
  tr.appendChild(td);
  tbody.appendChild(tr);
}

async function fetchCatMap() {
  const cats = await getJSON("/api/v1/cats");
  const map = new Map();
  for (const c of cats) {
    map.set(c.id, c.name || c.id);
  }
  map.set("unknown", "未识别");
  return map;
}

async function loadCats(selects) {
  const cats = await getJSON("/api/v1/cats");
  for (const sel of selects) {
    const current = sel.value;
    sel.innerHTML = "";
    sel.appendChild(el("option", "选择猫咪…"));
    sel.options[0].value = "";
    const unknown = el("option", "未识别");
    unknown.value = "unknown";
    sel.appendChild(unknown);
    for (const c of cats) {
      if (c.id === "unknown") continue;
      const opt = el("option", c.name || c.id);
      opt.value = c.id;
      sel.appendChild(opt);
    }
    if (current) sel.value = current;
  }
}

async function fetchCameraMap() {
  const cameras = await getJSON("/api/v1/cameras");
  const map = new Map();
  for (const c of cameras) {
    map.set(c.id, c.name || c.id);
  }
  return map;
}

function formatCameraIds(ids, cameraMap) {
  if (!ids || ids.length === 0) return "—";
  return ids.map((id) => cameraMap.get(id) || id).join("、");
}

function visitRecordings(visit) {
  if (visit.recordings && visit.recordings.length > 0) {
    return visit.recordings;
  }
  if (visit.recording_path) {
    return [{ camera_id: "cam1", name: "主摄", url: `/recordings/${visit.id}.mp4` }];
  }
  return [];
}

async function loadVisits() {
  let visits;
  let catMap;
  let cameraMap;
  try {
    [visits, catMap, cameraMap] = await Promise.all([
      getJSON("/api/v1/visits"),
      fetchCatMap(),
      fetchCameraMap(),
    ]);
  } catch (err) {
    // Network down / API error / bad JSON — distinct from "no visits yet".
    renderTableMessage(
      `<p>加载失败</p><small>${err.message || "请检查服务是否在运行"}</small>`,
      { retry: true }
    );
    return;
  }

  const tbody = document.querySelector("#visits tbody");
  tbody.innerHTML = "";
  const selects = [];
  const wasteSelects = [];

  if (visits.length === 0) {
    renderTableMessage("<p>暂无 visit 记录</p><small>猫咪进入 ROI 后会自动出现在这里</small>");
    return;
  }

  for (const v of visits) {
    const tr = document.createElement("tr");
    tr.appendChild(el("td", formatTime(v.started_at)));

    const catTd = el("td");
    const catLabel = el(
      "span",
      catMap.get(v.cat_id) || v.cat_id,
      `cat-label${v.cat_id === "unknown" ? " unknown" : ""}`
    );
    catTd.appendChild(catLabel);
    if (v.multi_cat) {
      const badge = el("span", `👥 ${v.max_cats || 2}`, "multi-cat-badge");
      badge.title = "本次有多只猫同时在盆里,身份识别不可靠";
      badge.style.marginLeft = "6px";
      badge.style.color = "#c2410c";
      badge.style.fontSize = "0.85em";
      catTd.appendChild(badge);
    }
    tr.appendChild(catTd);

    const durTd = el("td");
    durTd.appendChild(el("span", formatDuration(v.duration_sec)));
    const wasteLabel = { poop: "💩 屎", pee: "💧 尿" }[v.waste_type];
    if (wasteLabel) {
      const wb = el("span", wasteLabel, "waste-badge");
      wb.style.marginLeft = "6px";
      wb.style.fontSize = "0.85em";
      // Surface the signals behind the guess so the heuristic can be eyeballed/tuned.
      const conf = typeof v.waste_confidence === "number" ? v.waste_confidence.toFixed(2) : "?";
      const dig = typeof v.digging_motion === "number" ? v.digging_motion.toFixed(1) : "?";
      wb.title = `启发式判断 置信度 ${conf} · 后段刨砂强度 ${dig} · 时长 ${v.duration_sec ?? "?"}s`;
      durTd.appendChild(wb);
    }
    tr.appendChild(durTd);

    tr.appendChild(el("td", formatCameraIds(v.camera_ids, cameraMap), "conf-value"));

    const conf =
      typeof v.confidence === "number" ? v.confidence.toFixed(2) : String(v.confidence ?? "");
    tr.appendChild(el("td", conf, "conf-value"));

    const recTd = el("td");
    recTd.className = "recording";
    const recordings = visitRecordings(v);
    if (recordings.length === 0) {
      recTd.appendChild(el("span", "—", "conf-value"));
    } else {
      for (const rec of recordings) {
        const block = el("div", null, "recording-block");
        block.appendChild(el("div", rec.name, "recording-label"));
        const video = document.createElement("video");
        video.controls = true;
        video.preload = "metadata";
        video.src = rec.url;
        block.appendChild(video);
        const link = document.createElement("a");
        link.href = rec.url;
        link.target = "_blank";
        link.rel = "noopener";
        link.className = "text-link";
        link.textContent = "新标签页打开";
        block.appendChild(link);
        recTd.appendChild(block);
      }
    }
    tr.appendChild(recTd);

    const corrTd = el("td");
    const sel = document.createElement("select");
    sel.className = "cat-select";
    sel.dataset.visitId = v.id;
    corrTd.appendChild(sel);

    const wsel = document.createElement("select");
    wsel.className = "waste-select";
    wsel.dataset.visitId = v.id;
    for (const [val, label] of [["", "屎/尿…"], ["poop", "💩 屎"], ["pee", "💧 尿"], ["unknown", "未知"]]) {
      const opt = el("option", label);
      opt.value = val;
      wsel.appendChild(opt);
    }
    wsel.value = v.waste_type === "poop" || v.waste_type === "pee" ? v.waste_type : "";
    corrTd.appendChild(wsel);
    tr.appendChild(corrTd);

    tbody.appendChild(tr);
    selects.push(sel);
    wasteSelects.push(wsel);
  }

  await loadCats(selects);
  selects.forEach((sel) => {
    sel.dataset.prev = sel.value || "";
    sel.addEventListener("change", () =>
      submitChange(sel, `/api/v1/visits/${sel.dataset.visitId}/correct`, { cat_id: sel.value }, "已更新猫咪")
    );
  });
  wasteSelects.forEach((wsel) => {
    wsel.dataset.prev = wsel.value || "";
    wsel.addEventListener("change", () =>
      submitChange(wsel, `/api/v1/visits/${wsel.dataset.visitId}/waste`, { waste_type: wsel.value }, "已更新屎/尿")
    );
  });
}

async function submitChange(select, url, body, okMsg) {
  if (!select.value) return;
  const prev = select.dataset.prev || "";
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (res.status === 401) {
      window.location.href = "/login";
      return;
    }
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `HTTP ${res.status}`);
    }
    showToast(okMsg, "ok");
    loadVisits(); // refresh on success only
  } catch (err) {
    showToast(`更新失败:${err.message}`, "error");
    select.value = prev; // revert the dropdown; don't reload over the user's view
  }
}

loadVisits();
