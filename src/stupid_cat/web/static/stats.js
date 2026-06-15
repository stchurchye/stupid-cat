let currentDays = 7;

// el() and getJSON() live in util.js (loaded first).

function formatDuration(sec) {
  if (!sec) return "0 秒";
  if (sec < 60) return `${sec} 秒`;
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return s ? `${m} 分 ${s} 秒` : `${m} 分`;
}

function formatTime(iso) {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString("zh-CN", {
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    });
  } catch {
    return iso;
  }
}

function formatDayLabel(dateStr) {
  if (!dateStr) return "";
  const d = new Date(`${dateStr}T12:00:00`);
  return d.toLocaleDateString("zh-CN", { month: "numeric", day: "numeric", weekday: "short" });
}

function renderSummary(stats) {
  const root = document.getElementById("summary-cards");
  root.innerHTML = "";
  const avg =
    stats.total_visits > 0
      ? Math.round(stats.total_duration_sec / stats.total_visits)
      : 0;
  const items = [
    { label: "总次数", value: String(stats.total_visits) },
    { label: "总时长", value: formatDuration(stats.total_duration_sec) },
    { label: "平均时长", value: formatDuration(avg) },
    { label: "有记录猫咪", value: String(stats.by_cat.filter((c) => c.visits > 0).length) },
  ];
  for (const item of items) {
    const card = el("div", null, "stat-card");
    card.appendChild(el("div", item.value, "stat-value"));
    card.appendChild(el("div", item.label, "stat-label"));
    root.appendChild(card);
  }
}

function renderDayChart(byDay) {
  const root = document.getElementById("day-chart");
  root.innerHTML = "";
  if (!byDay.length) {
    root.appendChild(el("div", "暂无数据", "empty-state"));
    return;
  }
  const maxVisits = Math.max(...byDay.map((d) => d.visits), 1);
  for (const day of byDay) {
    const col = el("div", null, "day-col");
    const barWrap = el("div", null, "day-bar-wrap");
    const bar = el("div", null, "day-bar");
    bar.style.height = `${Math.max(8, (day.visits / maxVisits) * 100)}%`;
    bar.title = `${day.date}: ${day.visits} 次`;
    barWrap.appendChild(bar);
    col.appendChild(barWrap);
    col.appendChild(el("div", String(day.visits), "day-count"));
    col.appendChild(el("div", formatDayLabel(day.date), "day-label"));
    root.appendChild(col);
  }
}

function mergeCatRows(stats, cats) {
  const byId = new Map(stats.by_cat.map((c) => [c.cat_id, c]));
  const rows = cats.map((cat) => {
    const hit = byId.get(cat.id);
    return (
      hit || {
        cat_id: cat.id,
        name: cat.name,
        visits: 0,
        duration_sec: 0,
        avg_duration_sec: 0,
        last_visit_at: null,
      }
    );
  });
  const unknown = byId.get("unknown");
  if (unknown) rows.push(unknown);
  rows.sort((a, b) => b.visits - a.visits || a.name.localeCompare(b.name, "zh-CN"));
  return rows;
}

function renderCatTable(rows, totalVisits) {
  const tbody = document.querySelector("#cat-stats tbody");
  tbody.innerHTML = "";
  if (!rows.length) {
    const tr = document.createElement("tr");
    const td = el("td");
    td.colSpan = 6;
    td.className = "empty-state";
    td.innerHTML = "<p>暂无统计数据</p><small>有 visit 记录后会出现在这里</small>";
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }
  for (const row of rows) {
    const tr = document.createElement("tr");
    tr.appendChild(el("td", row.name, row.visits ? "cat-label" : "cat-label unknown"));
    tr.appendChild(el("td", String(row.visits), "conf-value"));
    tr.appendChild(el("td", formatDuration(row.duration_sec), "conf-value"));
    tr.appendChild(el("td", row.visits ? `${row.avg_duration_sec} 秒` : "—", "conf-value"));
    tr.appendChild(el("td", formatTime(row.last_visit_at), "conf-value"));

    const shareTd = el("td");
    if (totalVisits > 0 && row.visits > 0) {
      const pct = Math.round((row.visits / totalVisits) * 100);
      const barRow = el("div", null, "share-bar-row");
      const track = el("div", null, "share-track");
      const fill = el("div", null, "share-fill");
      fill.style.width = `${pct}%`;
      track.appendChild(fill);
      barRow.appendChild(track);
      barRow.appendChild(el("span", `${pct}%`, "share-pct"));
      shareTd.appendChild(barRow);
    } else {
      shareTd.appendChild(el("span", "—", "conf-value"));
    }
    tr.appendChild(shareTd);
    tbody.appendChild(tr);
  }
}

function renderStatsError(message) {
  const tbody = document.querySelector("#cat-stats tbody");
  tbody.innerHTML = "";
  const tr = document.createElement("tr");
  const td = el("td");
  td.colSpan = 6;
  td.className = "empty-state";
  td.innerHTML = `<p>加载失败</p><small>${message || "请检查服务是否在运行"}</small>`;
  const btn = el("button", "重试");
  btn.className = "text-link";
  btn.style.marginTop = "6px";
  btn.addEventListener("click", () => loadStats(currentDays));
  td.appendChild(document.createElement("br"));
  td.appendChild(btn);
  tr.appendChild(td);
  tbody.appendChild(tr);
}

async function loadStats(days) {
  currentDays = days;
  const qs = days > 0 ? `?days=${days}` : "?days=0";
  let stats;
  let cats;
  try {
    [stats, cats] = await Promise.all([getJSON(`/api/v1/stats${qs}`), getJSON("/api/v1/cats")]);
  } catch (err) {
    renderStatsError(err.message);
    return;
  }
  renderSummary(stats);
  renderDayChart(stats.by_day);
  renderCatTable(mergeCatRows(stats, cats), stats.total_visits);
}

document.getElementById("period-tabs").addEventListener("click", (e) => {
  const btn = e.target.closest(".period-tab");
  if (!btn) return;
  document.querySelectorAll(".period-tab").forEach((b) => b.classList.remove("active"));
  btn.classList.add("active");
  loadStats(Number(btn.dataset.days));
});

loadStats(currentDays);
