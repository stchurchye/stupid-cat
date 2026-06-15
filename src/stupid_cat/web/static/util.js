// Shared helpers for the dashboard pages (loaded before each page script).

function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text != null) node.textContent = text;
  if (className) node.className = className;
  return node;
}

async function getJSON(url) {
  const res = await fetch(url);
  if (res.status === 401) {
    window.location.href = "/login"; // session expired / key rotated -> re-auth
    throw new Error("unauthorized");
  }
  if (!res.ok) throw new Error(`${url} → HTTP ${res.status}`);
  return res.json();
}
