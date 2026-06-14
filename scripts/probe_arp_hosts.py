"""Quick probe of ARP-known hosts for Hikvision/camera ports."""
from __future__ import annotations

import re
import socket

IPS = [
    "192.168.31.32", "192.168.31.112", "192.168.31.149", "192.168.31.181",
    "192.168.31.14", "192.168.31.25", "192.168.31.37", "192.168.31.38",
    "192.168.31.73", "192.168.31.79", "192.168.31.94", "192.168.31.111",
    "192.168.31.116", "192.168.31.122", "192.168.31.131", "192.168.31.133",
    "192.168.31.143", "192.168.31.155", "192.168.31.161", "192.168.31.180",
    "192.168.31.192", "192.168.31.209", "192.168.31.226", "192.168.31.237",
    "192.168.31.241", "192.168.31.242",
]
PORTS = [80, 443, 554, 8000, 8080, 8443]


def port_open(ip: str, port: int) -> bool:
    s = socket.socket()
    s.settimeout(0.9)
    try:
        return s.connect_ex((ip, port)) == 0
    finally:
        s.close()


def probe_http(ip: str, port: int) -> str:
    try:
        s = socket.create_connection((ip, port), timeout=2)
        s.sendall(f"GET / HTTP/1.0\r\nHost: {ip}\r\n\r\n".encode())
        data = s.recv(4096).decode("latin-1", "replace")
        s.close()
        return data
    except OSError:
        return ""


def main() -> None:
    for ip in IPS:
        open_ports = [p for p in PORTS if port_open(ip, p)]
        if not open_ports:
            print(f"{ip:16} no camera ports")
            continue
        http = ""
        for p in (80, 8000, 8080):
            if p in open_ports:
                http = probe_http(ip, p)
                if http:
                    break
        is_hik = any(
            k in http.lower()
            for k in ("hikvision", "doc/page/login", "/isapi/", "webcomponents")
        )
        server = re.search(r"Server:\s*(.+)", http, re.I)
        srv = (server.group(1).strip() if server else "")[:55]
        tag = " <<<< HIKVISION?" if is_hik or 8000 in open_ports or 554 in open_ports else ""
        print(f"{ip:16} ports={open_ports} server={srv}{tag}")


if __name__ == "__main__":
    main()
