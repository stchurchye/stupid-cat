const REFRESH_MS = 800;

// el() lives in util.js (loaded first).

function startPreview(cameraId, statusEl, imgEl) {
  const tick = () => {
    imgEl.src = `/api/v1/cameras/${encodeURIComponent(cameraId)}/preview.jpg?t=${Date.now()}`;
  };
  imgEl.addEventListener("load", () => {
    statusEl.textContent = "已连接";
    statusEl.className = "badge ok";
  });
  imgEl.addEventListener("error", () => {
    statusEl.textContent = "等待画面";
    statusEl.className = "badge wait";
  });
  tick();
  return window.setInterval(tick, REFRESH_MS);
}

async function init() {
  const root = document.getElementById("camera-list");
  let cameras = [];
  try {
    const res = await fetch("/api/v1/cameras");
    cameras = await res.json();
  } catch {
    try {
      const health = await fetch("/api/v1/health").then((r) => r.json());
      cameras = (health.cameras || []).map((c) => ({ id: c.id, name: c.name || c.id }));
    } catch {
      cameras = [{ id: "cam1", name: "cam1" }];
    }
  }
  if (cameras.length === 0) {
    const empty = el("div", null, "card card-body empty-state");
    empty.innerHTML = "<p>没有配置的摄像头</p>";
    root.appendChild(empty);
    return;
  }
  for (const cam of cameras) {
    const id = cam.id;
    const label = cam.name || id;
    const card = el("section", null, "camera-card");
    const header = el("div", null, "camera-header");
    header.appendChild(el("h2", label));
    if (label !== id) {
      header.appendChild(el("span", id, "cam-id"));
    }
    const status = el("span", "连接中", "badge wait");
    header.appendChild(status);
    card.appendChild(header);

    const wrap = el("div", null, "preview-wrap");
    const img = document.createElement("img");
    img.alt = `${label} 实时预览`;
    img.className = "preview-img";
    wrap.appendChild(img);
    card.appendChild(wrap);

    root.appendChild(card);
    startPreview(id, status, img);
  }
}

init();
