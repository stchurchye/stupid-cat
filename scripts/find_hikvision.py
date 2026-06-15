"""Scan LAN for Hikvision-style devices (ports 80/554/8000)."""
from __future__ import annotations

import re
import socket
import subprocess

SUBNET = "192.168.31"
HIK_PORTS = [80, 443, 554, 8000, 8080]


def ping(ip: str) -> bool:
    try:
        r = subprocess.run(
            ["ping", "-n", "1", "-w", "400", ip],
            capture_output=True,
            text=True,
            timeout=4,
        )
        return "TTL=" in r.stdout.upper()
    except Exception:
        return False


def port_open(ip: str, port: int, timeout: float = 1.0) -> bool:
    s = socket.socket()
    s.settimeout(timeout)
    try:
        return s.connect_ex((ip, port)) == 0
    finally:
        s.close()


def probe_http(ip: str, port: int = 80) -> str:
    try:
        s = socket.create_connection((ip, port), timeout=2)
        s.sendall(f"GET / HTTP/1.0\r\nHost: {ip}\r\n\r\n".encode())
        data = s.recv(4096).decode("latin-1", "replace")
        s.close()
        return data
    except Exception:
        return ""


def main() -> None:
    ips = [f"{SUBNET}.{i}" for i in range(1, 255)]
    alive = [ip for ip in ips if ping(ip)]
    print(f"Alive: {len(alive)}")

    for ip in alive:
        if ip.endswith(".87"):
            continue
        open_ports = [p for p in HIK_PORTS if port_open(ip, p)]
        if not open_ports:
            continue
        http = ""
        for p in (80, 8000, 8080, 443):
            if p in open_ports:
                http = probe_http(ip, p)
                if http:
                    break
        low = http.lower()
        is_hik = any(
            k in low
            for k in ("hikvision", "doc/page/login", "webcomponents", "/isapi/")
        )
        server = re.search(r"Server:\s*(.+)", http, re.I)
        title = re.search(r"<title>([^<]+)</title>", http, re.I)
        srv = (server.group(1).strip() if server else "")[:60]
        ttl = (title.group(1).strip() if title else "")[:40]
        tag = " *** HIKVISION? ***" if is_hik or 8000 in open_ports else ""
        print(f"{ip:16} ports={open_ports} server={srv!r} title={ttl!r}{tag}")


if __name__ == "__main__":
    main()
