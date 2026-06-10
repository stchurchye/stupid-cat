function el(tag, text) {
  const node = document.createElement(tag);
  if (text != null) node.textContent = text;
  return node;
}

async function loadCats(selects) {
  const res = await fetch("/api/v1/cats");
  const cats = await res.json();
  for (const sel of selects) {
    const current = sel.value;
    sel.innerHTML = "";
    sel.appendChild(el("option", "纠错…"));
    sel.options[0].value = "";
    const unknown = el("option", "unknown");
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
  const res = await fetch("/api/v1/visits");
  const visits = await res.json();
  const tbody = document.querySelector("#visits tbody");
  tbody.innerHTML = "";
  const selects = [];

  for (const v of visits) {
    const tr = document.createElement("tr");
    tr.appendChild(el("td", v.started_at));
    tr.appendChild(el("td", v.cat_id));
    tr.appendChild(el("td", String(v.duration_sec ?? "")));
    const conf =
      typeof v.confidence === "number" ? v.confidence.toFixed(2) : String(v.confidence ?? "");
    tr.appendChild(el("td", conf));

    const recTd = el("td");
    recTd.className = "recording";
    if (v.recording_path) {
      const url = `/recordings/${v.id}.mp4`;
      const video = document.createElement("video");
      video.controls = true;
      video.preload = "metadata";
      video.width = 240;
      video.src = url;
      recTd.appendChild(video);
      const link = document.createElement("a");
      link.href = url;
      link.target = "_blank";
      link.rel = "noopener";
      link.textContent = "新标签打开";
      recTd.appendChild(document.createElement("br"));
      recTd.appendChild(link);
    }
    tr.appendChild(recTd);

    const corrTd = el("td");
    const sel = document.createElement("select");
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
