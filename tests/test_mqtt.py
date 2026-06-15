import json

from stupid_cat.config import MqttConfig
from stupid_cat.mqtt import MqttPublisher


class _FakeClient:
    def __init__(self) -> None:
        self.msgs: list[tuple[str, str]] = []

    def publish(self, topic: str, payload: str) -> None:
        self.msgs.append((topic, payload))


def test_publisher_publishes_to_prefixed_topic() -> None:
    fake = _FakeClient()
    pub = MqttPublisher(MqttConfig(enabled=True, topic_prefix="sc"), client=fake)
    pub.publish("visit_ended", {"a": 1})
    assert fake.msgs == [("sc/visit_ended", json.dumps({"a": 1}))]


def test_publisher_noop_when_disabled() -> None:
    fake = _FakeClient()
    pub = MqttPublisher(MqttConfig(enabled=False, topic_prefix="sc"), client=fake)
    pub.publish("visit_ended", {"a": 1})
    assert fake.msgs == []


def test_publish_swallows_client_errors() -> None:
    class _Boom:
        def publish(self, *a: object) -> None:
            raise RuntimeError("broker down")

    pub = MqttPublisher(MqttConfig(enabled=True), client=_Boom())
    pub.publish("x", {"a": 1})  # must not raise
