"""Optional MQTT alerting (spec §11).

paho-mqtt is an optional dependency (``pip install paho-mqtt`` or the ``mqtt``
extra). When mqtt is disabled or the lib is missing, every method is a safe no-op,
and a publish failure can never propagate into the pipeline.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from stupid_cat.config import MqttConfig

logger = logging.getLogger(__name__)


class MqttPublisher:
    def __init__(self, cfg: MqttConfig, *, client: Any | None = None) -> None:
        self.cfg = cfg
        self._client = client
        self._enabled = cfg.enabled

    def start(self) -> None:
        if not self._enabled or self._client is not None:
            return
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            logger.warning("mqtt.enabled but paho-mqtt is not installed; alerts disabled")
            self._enabled = False
            return
        client = mqtt.Client()
        if self.cfg.username:
            client.username_pw_set(self.cfg.username, self.cfg.password)
        try:
            # async + loop_start so a broker that's down at startup (or drops) keeps
            # retrying in the background instead of blocking/raising here.
            client.connect_async(self.cfg.broker_host, self.cfg.broker_port)
            client.loop_start()
        except Exception:  # noqa: BLE001 - never let alerting break startup
            logger.exception("mqtt connect failed; alerts disabled")
            self._enabled = False
            return
        self._client = client

    def publish(self, subtopic: str, payload: dict[str, Any]) -> None:
        # Fire-and-forget: paho's publish() only enqueues onto the loop_start
        # background thread, so this is non-blocking and safe to call while the
        # pipeline holds its lock (e.g. from _on_visit_end / _disk_ok).
        if not self._enabled or self._client is None:
            return
        topic = f"{self.cfg.topic_prefix}/{subtopic}"
        try:
            self._client.publish(topic, json.dumps(payload))
        except Exception:  # noqa: BLE001 - alerting must never break the pipeline
            logger.exception("mqtt publish to %s failed", topic)

    def close(self) -> None:
        if self._client is None:
            return
        try:
            self._client.loop_stop()
            self._client.disconnect()
        except Exception:  # noqa: BLE001
            pass
        self._client = None
