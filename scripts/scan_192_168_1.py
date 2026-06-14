"""Scan 192.168.1.x for Hikvision cameras (when PC ethernet is on 1.x)."""
from __future__ import annotations

import re
import socket
import subprocess

SUBNET = "192.168.1"
SKIP = {100}  # this PC


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


def port_open(ip: str, port: int) -> bool:
    s = socket.socket()
    s.settimeout(0.8)
    try:
        return s.connect_ex((ip, port)) == 0
    finally:
        s.close()


def probe_http(ip: str) -> str:
    try:
        s = socket.create_connection((ip, 80), timeout=2)
        s.sendall(f"GET / HTTP/1.0\r\nHost: {ip}\r\n\r\n".encode())
        data = s.recv(4096).decode("latin-1", "replace")
        s.close()
        return data
    except OSError:
        return ""


def main() -> None:
    found = 0
    for i in range(1, 255):
        if i in SKIP:
            continue
        ip = f"{SUBNET}.{i}"
        if not ping(ip):
            continue
        found += 1
        ports = [p for p in (80, 554, 8000) if port_open(ip, p)]
        http = probe_http(ip) if 80 in ports else ""
        is_hik = any(
            k in http.lower()
            for k in ("hikvision", "doc/page/login", "/isapi/")
        )
        server = re.search(r"Server:\s*(.+)", http, re.I)
        srv = (server.group(1).strip() if server else "")[:55]
        tag = " << CAMERA?" if is_hik or 554 in ports or 8000 in ports else ""
        print(f"{ip} ports={ports} hik={is_hik} server={srv}{tag}")
    print(f"alive (excl. this PC): {found}")


if __name__ == "__main__":
    main()
