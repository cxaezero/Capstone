"""Flask dashboard: four MJPEG streams + an SSE alert log.

Run:  python demo/app.py [--host 0.0.0.0] [--port 5050] [--device cpu|cuda]
Keys: stream0 = looped demo video (enhanced + detection), stream1 = RTMP enhanced + detection,
      stream2/stream3 = RTMP relayed as-is. RTMP base URL and paths come from capstone.config.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow running without `pip install -e .`

from flask import Flask, Response, abort, jsonify, render_template  # noqa: E402

from capstone import config  # noqa: E402
from demo.streams import LogBus, StreamRegistry, mjpeg_frames, sse_events  # noqa: E402

log = logging.getLogger(__name__)

DEMO_LOCATION = os.environ.get("CAPSTONE_DEMO_LOCATION", "Seoul, South Korea")


def create_app(registry=None, log_bus: LogBus | None = None, device=None) -> Flask:
    """Build the app. ``registry`` only needs ``get(key)`` / ``keys()`` (injectable for tests)."""
    app = Flask(__name__)
    log_bus = log_bus or LogBus()
    if registry is None:
        device = device or config.get_device()
        log.info("loading models on %s", device)
        registry = StreamRegistry(device, log_bus)
        registry.warm_up()
    app.extensions["registry"] = registry
    app.extensions["log_bus"] = log_bus

    @app.route("/")
    def dashboard():
        return render_template("dashboard.html", location=DEMO_LOCATION, max_logs=config.MAX_LOGS,
                               stream_keys=registry.keys())

    @app.route("/stream/<key>")
    def stream_video(key):
        try:
            stream = registry.get(key)
        except KeyError:
            abort(404)
        return Response(mjpeg_frames(stream.jpeg), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/log_stream")
    def log_stream():
        return Response(sse_events(log_bus), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.route("/healthz")
    def healthz():
        return jsonify(status="ok", streams=list(registry.keys()), recent_logs=log_bus.recent())

    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=config.DEMO_PORT)
    parser.add_argument("--device", default=None, help="cpu / cuda / mps (default: auto, or CAPSTONE_DEVICE)")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = create_app(device=config.get_device(args.device))
    app.run(host=args.host, port=args.port, threaded=True, debug=args.debug, use_reloader=False)


if __name__ == "__main__":
    main()
