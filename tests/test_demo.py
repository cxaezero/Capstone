import time

import numpy as np
import pytest

from capstone import config
from demo.app import create_app
from demo.streams import FrameSource, Latest, LogBus, PassthroughStream, STREAM_KEYS, mjpeg_frames, no_signal_jpeg

FAKE_JPEG = b"\xff\xd8fake-jpeg\xff\xd9"


class FakeStream:
    def __init__(self):
        self.jpeg = Latest()
        self.jpeg.publish(FAKE_JPEG)


class FakeRegistry:
    def __init__(self):
        self.streams = {k: FakeStream() for k in STREAM_KEYS}

    def keys(self):
        return STREAM_KEYS

    def get(self, key):
        return self.streams[key]


@pytest.fixture
def client():
    app = create_app(registry=FakeRegistry(), log_bus=LogBus())
    app.testing = True
    return app.test_client()


def _first_chunk(resp):
    try:
        return next(iter(resp.response))
    finally:
        resp.close()


def test_dashboard_renders_all_streams(client):
    resp = client.get("/")
    assert resp.status_code == 200
    for key in STREAM_KEYS:
        assert f"/stream/{key}".encode() in resp.data
    assert f"const maxLogs = {config.MAX_LOGS};".encode() in resp.data
    assert b"evtSource.close()" not in resp.data


def test_stream_returns_mjpeg_part(client):
    resp = client.get("/stream/stream0")
    assert resp.status_code == 200
    assert resp.mimetype == "multipart/x-mixed-replace"
    chunk = _first_chunk(resp)
    assert chunk.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n") and FAKE_JPEG in chunk


def test_unknown_stream_is_404(client):
    assert client.get("/stream/nope").status_code == 404


def test_log_stream_sends_backlog_as_sse(client):
    client.application.extensions["log_bus"].publish("hello")
    resp = client.get("/log_stream")
    assert resp.mimetype == "text/event-stream"
    assert _first_chunk(resp) == b"data: hello\n\n"
    health = client.get("/healthz").get_json()
    assert health["status"] == "ok" and health["recent_logs"] == ["hello"]


def test_latest_wait_and_timeout():
    slot = Latest()
    assert slot.wait(0, timeout=0.01) is None
    slot.publish("a")
    assert slot.wait(0, timeout=0.01) == (1, "a")
    assert slot.wait(1, timeout=0.01) is None


def test_log_bus_backlog_then_keepalive():
    bus = LogBus(maxlen=3)
    for m in "abcd":
        bus.publish(m)
    assert bus.recent() == ["d", "c", "b"]
    sub = bus.subscribe(keepalive=0.01)
    assert [next(sub) for _ in range(3)] == ["b", "c", "d"]
    assert next(sub) is None  # idle -> keep-alive
    bus.alert("stream9")
    assert "stream9" in next(sub)


def test_frame_source_backs_off_on_missing_file(tmp_path):
    src = FrameSource(tmp_path / "missing.mp4", backoff=(0.02, 0.05))
    src.start()
    time.sleep(0.15)
    assert src.is_alive() and src.latest.seq == 0
    src.stop()
    src.join(timeout=1)
    assert not src.is_alive()


@pytest.mark.skipif(not config.DEMO_VIDEO.is_file(), reason="demo video not present")
def test_passthrough_stream_publishes_jpeg_from_demo_video():
    src = FrameSource(config.DEMO_VIDEO, loop=True, pace=False)
    stream = PassthroughStream("stream2", src)
    stream.start()
    got = stream.jpeg.wait(0, timeout=10)
    assert got is not None and got[1].startswith(b"\xff\xd8")
    src.stop()


def test_mjpeg_sends_placeholder_while_no_frames():
    gen = mjpeg_frames(Latest(), timeout=0.01)
    part = next(gen)
    gen.close()
    assert part.startswith(b"--frame\r\n") and no_signal_jpeg() in part
    assert no_signal_jpeg().startswith(b"\xff\xd8")
