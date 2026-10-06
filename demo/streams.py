"""Threaded video sources, per-stream processing and broadcasting for the dashboard.

Design (REFACTORING_PLAN B2/B3/B4):
  FrameSource (thread)  -> Latest[frame]  -> EnhancedStream / PassthroughStream (thread) -> Latest[jpeg]
Every HTTP client of a stream only *reads* the latest JPEG, so any number of clients can watch
the same key without competing for frames. Sources reconnect with back-off instead of spinning,
and processing errors are logged without killing the thread.
"""
from __future__ import annotations

import datetime
import logging
import threading
import time
from collections import deque
from typing import Iterator

import cv2
import numpy as np

from capstone import config
from capstone.models import ModelBundle, load_models
from capstone.pipeline import AnomalyPipeline
from capstone.preprocess import frame_to_tensor, tensor_to_bgr

log = logging.getLogger(__name__)

ENHANCED_KEYS = ("stream0", "stream1")      # de-weathered + anomaly detection
PASSTHROUGH_KEYS = ("stream2", "stream3")   # raw RTMP relay
STREAM_KEYS = ENHANCED_KEYS + PASSTHROUGH_KEYS


class Latest:
    """Single-slot mailbox: a writer replaces the value; readers wait for a newer sequence number."""

    def __init__(self):
        self._cond = threading.Condition()
        self._seq = 0
        self._value = None

    @property
    def seq(self) -> int:
        return self._seq

    def publish(self, value) -> None:
        with self._cond:
            self._value = value
            self._seq += 1
            self._cond.notify_all()

    def wait(self, last_seq: int, timeout: float | None = None):
        """Return ``(seq, value)`` newer than ``last_seq``, or ``None`` on timeout."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq > last_seq, timeout):
                return None
            return self._seq, self._value


class LogBus:
    """Bounded alert log; subscribers receive the backlog and then new entries (no polling)."""

    def __init__(self, maxlen: int = config.MAX_LOGS):
        self._cond = threading.Condition()
        self._logs: deque[tuple[int, str]] = deque(maxlen=maxlen)
        self._next_id = 1

    def publish(self, message: str) -> None:
        with self._cond:
            self._logs.appendleft((self._next_id, message))
            self._next_id += 1
            self._cond.notify_all()

    def alert(self, key: str) -> None:
        now = datetime.datetime.now().strftime("%H:%M:%S")
        self.publish(f"[ {now} - {key} ] 이상 상황 발생")

    def recent(self) -> list[str]:
        with self._cond:
            return [m for _, m in self._logs]

    def subscribe(self, keepalive: float = 15.0) -> Iterator[str | None]:
        """Yield messages oldest-first; yields ``None`` every ``keepalive`` seconds when idle."""
        last_id = 0
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._logs and self._logs[0][0] > last_id, keepalive)
                fresh = [(i, m) for i, m in reversed(self._logs) if i > last_id]
            if not fresh:
                yield None
                continue
            for i, m in fresh:
                last_id = i
                yield m


class FrameSource(threading.Thread):
    """Reads frames from a file (looped, paced at its fps) or a live URL into a ``Latest`` slot.

    When the source cannot be opened or drops, it retries with exponential back-off.
    """

    def __init__(self, uri, *, loop: bool = False, pace: bool = False, name: str | None = None,
                 backoff: tuple[float, float] = (1.0, 30.0)):
        super().__init__(name=name or f"source:{uri}", daemon=True)
        self.uri = str(uri)
        self.loop = loop
        self.pace = pace
        self.backoff = backoff
        self.latest: Latest = Latest()
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        delay = self.backoff[0]
        while not self._stop.is_set():
            cap = cv2.VideoCapture(self.uri)
            if not cap.isOpened():
                cap.release()
                log.warning("%s: cannot open, retrying in %.0fs", self.uri, delay)
                self._stop.wait(delay)
                delay = min(delay * 2, self.backoff[1])
                continue
            delay = self.backoff[0]
            try:
                self._read_loop(cap)
            except Exception:
                log.exception("%s: reader failed", self.uri)
            finally:
                cap.release()
            if not self._stop.is_set():
                log.warning("%s: source ended/dropped, reconnecting in %.0fs", self.uri, delay)
                self._stop.wait(delay)

    def _read_loop(self, cap: cv2.VideoCapture) -> None:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_delay = 1.0 / fps if fps > 0 else 1.0 / 30
        failures = 0
        while not self._stop.is_set():
            started = time.monotonic()
            ok, frame = cap.read()
            if not ok:
                failures += 1
                if self.loop and failures == 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # end of file: start over
                    continue
                return  # live source dropped (or file unreadable): reconnect with back-off
            failures = 0
            self.latest.publish(frame)
            if self.pace:
                remaining = frame_delay - (time.monotonic() - started)
                if remaining > 0:
                    self._stop.wait(remaining)


class BaseStream(threading.Thread):
    """Takes the newest frame from a source, processes it, publishes a JPEG."""

    def __init__(self, key: str, source: FrameSource):
        super().__init__(name=f"stream:{key}", daemon=True)
        self.key = key
        self.source = source
        self.jpeg: Latest = Latest()

    def process(self, frame: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def start(self) -> None:
        if not self.source.is_alive():
            self.source.start()
        super().start()

    def run(self) -> None:
        last_seq = 0
        while True:
            got = self.source.latest.wait(last_seq, timeout=1.0)
            if got is None:
                continue  # no new frame (source down); wait() blocks, no busy loop
            last_seq, frame = got
            try:
                out = self.process(frame)
            except Exception:
                log.exception("%s: frame processing failed", self.key)
                time.sleep(0.5)
                continue
            ok, buf = cv2.imencode(".jpg", out)
            if ok:
                self.jpeg.publish(buf.tobytes())


class PassthroughStream(BaseStream):
    def process(self, frame: np.ndarray) -> np.ndarray:
        return frame


class EnhancedStream(BaseStream):
    """ESDNet enhancement + anomaly scoring; draws a red border and logs while an alert is active."""

    def __init__(self, key: str, source: FrameSource, pipeline: AnomalyPipeline, log_bus: LogBus, *,
                 alert_frames: int = config.ALERT_FRAMES, log_interval: float = config.LOG_INTERVAL,
                 display_size: tuple[int, int] = config.DISPLAY_SIZE, img_size: int = config.IMG_SIZE):
        super().__init__(key, source)
        self.pipeline = pipeline
        self.log_bus = log_bus
        self.alert_frames = alert_frames
        self.log_interval = log_interval
        self.display_size = display_size
        self.img_size = img_size
        self._last_log_time = 0.0

    def process(self, frame: np.ndarray) -> np.ndarray:
        result = self.pipeline.step(frame_to_tensor(frame, self.img_size))
        out = tensor_to_bgr(result.clean, self.display_size)
        if result.streak >= self.alert_frames:
            h, w = out.shape[:2]
            cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 0, 255), 3)
            now = time.monotonic()
            if now - self._last_log_time >= self.log_interval:
                self.log_bus.alert(self.key)
                self._last_log_time = now
        return out


class StreamRegistry:
    """Creates and starts the dashboard's streams on first use; one pipeline per enhanced stream."""

    def __init__(self, device, log_bus: LogBus, *, share_models: bool = config.SHARE_MODELS,
                 demo_video=config.DEMO_VIDEO, rtmp_base_url: str = config.RTMP_BASE_URL):
        self.device = device
        self.log_bus = log_bus
        self.share_models = share_models
        self.demo_video = demo_video
        self.rtmp_base_url = rtmp_base_url
        self._streams: dict[str, BaseStream] = {}
        self._bundles: list[ModelBundle] = []
        self._lock = threading.Lock()

    def keys(self) -> tuple[str, ...]:
        return STREAM_KEYS

    def warm_up(self) -> None:
        """Load model weights now so the first client does not wait for them."""
        with self._lock:
            needed = 1 if self.share_models else len(ENHANCED_KEYS)
            while len(self._bundles) < needed:
                self._bundles.append(load_models(self.device))

    def _bundle(self, index: int) -> ModelBundle:
        if self.share_models:
            index = 0
        while len(self._bundles) <= index:
            self._bundles.append(load_models(self.device))
        return self._bundles[index]

    def get(self, key: str) -> BaseStream:
        if key not in STREAM_KEYS:
            raise KeyError(key)
        with self._lock:
            stream = self._streams.get(key)
            if stream is None:
                stream = self._create(key)
                stream.start()
                self._streams[key] = stream
                log.info("stream %s started (%s)", key, stream.source.uri)
            return stream

    def _create(self, key: str) -> BaseStream:
        if key == "stream0":
            source = FrameSource(self.demo_video, loop=True, pace=True)
        else:
            source = FrameSource(f"{self.rtmp_base_url}/{key}")
        if key in ENHANCED_KEYS:
            bundle = self._bundle(ENHANCED_KEYS.index(key))
            pipeline = AnomalyPipeline(bundle.deweather, bundle.feature_extractor, bundle.classifier,
                                       device=self.device)
            return EnhancedStream(key, source, pipeline, self.log_bus)
        return PassthroughStream(key, source)


_NO_SIGNAL_CACHE: dict[tuple, bytes] = {}


def no_signal_jpeg(size: tuple[int, int] = config.DISPLAY_SIZE, text: str = "NO SIGNAL") -> bytes:
    """A dark placeholder frame shown while a source has not delivered anything."""
    key = (size, text)
    if key not in _NO_SIGNAL_CACHE:
        w, h = size
        img = np.full((h, w, 3), 20, dtype=np.uint8)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(img, text, ((w - tw) // 2, (h + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (160, 160, 160), 2)
        _NO_SIGNAL_CACHE[key] = cv2.imencode(".jpg", img)[1].tobytes()
    return _NO_SIGNAL_CACHE[key]


def _mjpeg_part(jpeg: bytes) -> bytes:
    return (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode()
            + b"\r\n\r\n" + jpeg + b"\r\n")


def mjpeg_frames(latest: Latest, timeout: float = 5.0, placeholder: bytes | None = None) -> Iterator[bytes]:
    """multipart/x-mixed-replace body for one stream.

    While no frame arrives within ``timeout`` seconds a placeholder image is sent instead, so the
    viewer sees "NO SIGNAL" and the server notices disconnected clients (a generator that never
    yields can never be closed by the WSGI server).
    """
    placeholder = placeholder if placeholder is not None else no_signal_jpeg()
    last_seq = 0
    while True:
        got = latest.wait(last_seq, timeout)
        if got is None:
            yield _mjpeg_part(placeholder)
            continue
        last_seq, jpeg = got
        yield _mjpeg_part(jpeg)


def sse_events(log_bus: LogBus) -> Iterator[str]:
    """text/event-stream body: one ``data:`` line per log message, comments as keep-alive."""
    for message in log_bus.subscribe():
        yield ": keep-alive\n\n" if message is None else f"data: {message}\n\n"
