"""Configure Hikvision camera device name, time sync, and OSD overlays."""
from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
from requests.auth import HTTPDigestAuth

NS = {"h": "http://www.hikvision.com/ver20/XMLSchema"}
TZ = "CST-8:00:00"
NTP_SERVERS = ("192.168.31.1", "cn.pool.ntp.org")


@dataclass
class CameraTarget:
    ip: str
    password: str
    device_name: str
    osd_name: str
    username: str = "admin"


def _session(username: str, password: str) -> requests.Session:
    s = requests.Session()
    s.auth = HTTPDigestAuth(username, password)
    s.headers["Content-Type"] = "application/xml"
    return s


def _tag(root: ET.Element, name: str) -> ET.Element | None:
    el = root.find(f"h:{name}", NS)
    if el is None:
        el = root.find(name)
    return el


def _set_text(parent: ET.Element, tag: str, value: str) -> None:
    el = _tag(parent, tag)
    if el is None:
        el = ET.SubElement(parent, tag)
    el.text = value


def parse_rtsp_cameras(config_path: Path) -> list[CameraTarget]:
    import yaml

    with config_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cameras = data.get("cameras") or []
    out: list[CameraTarget] = []
    for cam in cameras:
        url = cam.get("rtsp_url", "")
        parsed = urlparse(url)
        ip = parsed.hostname
        # RTSP URLs percent-encode credentials (@, :, / are URL delimiters), so
        # decode before using them for HTTP digest auth, or auth will 401.
        password = unquote(parsed.password or "")
        username = unquote(parsed.username or "admin")
        if not ip or not password:
            continue
        cam_id = cam.get("id", ip)
        display = cam.get("name") or cam_id
        out.append(
            CameraTarget(
                ip=ip,
                password=password,
                device_name=display,
                osd_name=display,
                username=username,
            )
        )
    return out


def _replace_tag(text: str, tag: str, value: str) -> str:
    pat = rf"(<{tag}>)([^<]*)(</{tag}>)"
    if re.search(pat, text):
        # Use a function replacement so backslashes / \1-like sequences in `value`
        # (e.g. a camera name) aren't interpreted as regex group references.
        return re.sub(pat, lambda m: f"{m.group(1)}{value}{m.group(3)}", text, count=1)
    return text


def _replace_enabled_for_text(text: str, display: str, enabled: bool) -> str:
    flag = "true" if enabled else "false"
    pattern = (
        rf"(<TextOverlay>\s*<id>\d+</id>\s*<enabled>)(true|false)(</enabled>\s*"
        rf"(?:.*?)<displayText>{re.escape(display)}</displayText>)"
    )
    return re.sub(pattern, rf"\1{flag}\3", text, flags=re.DOTALL)


def set_device_name(session: requests.Session, ip: str, name: str) -> None:
    get = session.get(f"http://{ip}/ISAPI/System/deviceInfo", timeout=8)
    get.raise_for_status()
    body = _replace_tag(get.text, "deviceName", name)
    put = session.put(f"http://{ip}/ISAPI/System/deviceInfo", data=body.encode("utf-8"), timeout=8)
    put.raise_for_status()


def set_time(session: requests.Session, ip: str) -> None:
    ntp_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<NTPServerList version="2.0" xmlns="http://www.hikvision.com/ver20/XMLSchema">
{''.join(
    f'''<NTPServer version="2.0" xmlns="http://www.hikvision.com/ver20/XMLSchema">
<id>{idx}</id>
<addressingFormatType>hostname</addressingFormatType>
<hostName>{host}</hostName>
<portNo>123</portNo>
<synchronizeInterval>60</synchronizeInterval>
</NTPServer>'''
    for idx, host in enumerate(NTP_SERVERS, start=1)
)}
</NTPServerList>"""
    session.put(f"http://{ip}/ISAPI/System/time/ntpServers", data=ntp_xml, timeout=8).raise_for_status()

    cst = timezone(timedelta(hours=8))
    now = datetime.now(cst).strftime("%Y-%m-%dT%H:%M:%S+08:00")
    time_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Time version="2.0" xmlns="http://www.hikvision.com/ver20/XMLSchema">
<timeMode>manual</timeMode>
<localTime>{now}</localTime>
<timeZone>{TZ}</timeZone>
</Time>"""
    session.put(f"http://{ip}/ISAPI/System/time", data=time_xml, timeout=8).raise_for_status()

    ntp_mode = f"""<?xml version="1.0" encoding="UTF-8"?>
<Time version="2.0" xmlns="http://www.hikvision.com/ver20/XMLSchema">
<timeMode>NTP</timeMode>
<timeZone>{TZ}</timeZone>
</Time>"""
    session.put(f"http://{ip}/ISAPI/System/time", data=ntp_mode, timeout=8).raise_for_status()


def _overlay_items(root: ET.Element, list_tag: str) -> list[ET.Element]:
    lst = _tag(root, list_tag)
    if lst is None:
        return []
    return list(lst)


def set_overlays(session: requests.Session, ip: str, osd_name: str) -> None:
    url = f"http://{ip}/ISAPI/System/Video/inputs/channels/1/overlays"
    get = session.get(url, timeout=8)
    get.raise_for_status()
    body = get.text
    for junk in ("温度检测", "实时温度", "呼救识别"):
        body = _replace_enabled_for_text(body, junk, False)

    if "<channelNameOverlay>" in body:
        body = re.sub(
            r"(<channelNameOverlay>\s*<enabled>)(true|false)(</enabled>)",
            r"\1true\3",
            body,
            count=1,
        )
        if re.search(r"<channelName>[^<]*</channelName>", body):
            body = _replace_tag(body, "channelName", osd_name)
    else:
        body = re.sub(
            r"(<TextOverlay>\s*<id>4</id>\s*<enabled>)(true|false)(</enabled>)",
            r"\1true\3",
            body,
            count=1,
        )
        body = re.sub(
            rf"(<TextOverlay>\s*<id>4</id>.*?<displayText>)([^<]*)(</displayText>)",
            rf"\1{re.escape(osd_name)}\3",
            body,
            count=1,
            flags=re.DOTALL,
        )

    body = re.sub(
        r"(<dateTimeOverlay>\s*<enabled>)(true|false)(</enabled>)",
        r"\1true\3",
        body,
        count=1,
    )

    put = session.put(url, data=body.encode("utf-8"), timeout=8)
    put.raise_for_status()


def configure_camera(cam: CameraTarget) -> None:
    session = _session(cam.username, cam.password)
    set_device_name(session, cam.ip, cam.device_name)
    set_time(session, cam.ip)
    set_overlays(session, cam.ip, cam.osd_name)
    verify = session.get(f"http://{cam.ip}/ISAPI/System/time", timeout=8)
    verify.raise_for_status()
    local = re.search(r"<localTime>([^<]+)</localTime>", verify.text)
    print(f"{cam.ip}\t{cam.device_name}\ttime={local.group(1) if local else '?'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.local.yaml"),
        help="Local config with cameras[].rtsp_url and optional name",
    )
    args = parser.parse_args()
    cameras = parse_rtsp_cameras(args.config)
    if not cameras:
        print("No cameras found in config", file=sys.stderr)
        return 1
    for cam in cameras:
        configure_camera(cam)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
