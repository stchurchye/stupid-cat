function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text != null) node.textContent = text;
  if (className) node.className = className;
  return node;
}

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

async function fetchCatMap() {
  const res = await fetch("/api/v1/cats");
  const cats = await res.json();
  const map = new Map();
  for (const c of cats) {
    map.set(c.id, c.name || c.id);
  }
  map.set("unknown", "未识别");
  return map;
}

async function loadCats(selects) {
  const res = await fetch("/api/v1/cats");
  const cats = await res.json();
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

async function loadVisits() {
  const [visitsRes, catMap] = await Promise.all([fetch("/api/v1/visits"), fetchCatMap()]);
  const visits = await visitsRes.json();
  const tbody = document.querySelector("#visits tbody");
  tbody.innerHTML = "";
  const selects = [];

  if (visits.length === 0) {
    const tr = document.createElement("tr");
    const td = el("td");
    td.colSpan = 6;
    td.className = "empty-state";
    td.innerHTML = "<p>暂无 visit 记录</p><small>猫咪进入 ROI 后会自动出现在这里</small>";
    tr.appendChild(td);
    tbody.appendChild(tr);
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
    tr.appendChild(catTd);

    tr.appendChild(el("td", formatDuration(v.duration_sec)));

    const conf =
      typeof v.confidence === "number" ? v.confidence.toFixed(2) : String(v.confidence ?? "");
    tr.appendChild(el("td", conf, "conf-value"));

    const recTd = el("td");
    recTd.className = "recording";
    if (v.recording_path) {
      const url = `/recordings/${v.id}.mp4`;
      const video = document.createElement("video");
      video.controls = true;
      video.preload = "metadata";
      video.src = url;
      recTd.appendChild(video);
      const link = document.createElement("a");
      link.href = url;
      link.target = "_blank";
      link.rel = "noopener";
      link.className = "text-link";
      link.textContent = "新标签页打开";
      recTd.appendChild(link);
    } else {
      recTd.appendChild(el("span", "—", "conf-value"));
    }
    tr.appendChild(recTd);

    const corrTd = el("td");
    const sel = document.createElement("select");
    sel.className = "cat-select";
    sel.dataset.visitId = v.id;
    corrTd.appendChild(sel);
    tr.appendChild(corrTd);

    tbody.appendChild(tr);
    selects.push(sel);
  }

  await loadCats(selects);
  selects.forEach((sel) => {
    sel.addEventListener("change", async () => {
      if (!sel.value) return;
      await fetch(`/api/v1/visits/${sel.dataset.visitId}/correct`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cat_id: sel.value }),
      });
      loadVisits();
    });
  });
}

loadVisits();
