#!/usr/bin/env python3
"""Surround VLM service for Jetson Orin Nano 8 GB.

Default: dummy backend (no 7 GB download). On the Jetson:

    export VLM_BACKEND=qwen
    export YOLO_WEIGHTS=yolov8n.pt   # or yolov8n.engine
    python -m vlm_camera.main --source http://127.0.0.1:9090/stitch
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import uvicorn

from vlm_camera.api.server import app, get_pipeline
from vlm_camera.camera.camera_capture import frames, open_capture


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO + Qwen2.5-VL-3B surround service")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--source",
        default="",
        help="Optional live source: CSI index, RTSP, file, or Java /stitch URL",
    )
    p.add_argument("--gstreamer", action="store_true")
    p.add_argument("--serve-only", action="store_true", help="Only FastAPI, no capture loop")
    return p.parse_args()


def capture_loop(source: str, use_gst: bool) -> None:
    cap = open_capture(int(source) if source.isdigit() else source, use_gstreamer=use_gst)
    if not cap.isOpened():
        print("Could not open source:", source, file=sys.stderr)
        sys.exit(1)
    pipe = get_pipeline()
    try:
        for frame in frames(cap):
            result = pipe.analyze_bgr(frame, force_vlm=False)
            if result.get("vlm_ran"):
                print(result.get("description") or result.get("scene"), flush=True)
    finally:
        cap.release()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("VLM_BACKEND", os.environ.get("VLM_BACKEND", "dummy"))
    if args.source and not args.serve_only:
        # Run API in a thread and capture in the foreground.
        import threading

        t = threading.Thread(
            target=lambda: uvicorn.run(app, host=args.host, port=args.port, log_level="info"),
            daemon=True,
        )
        t.start()
        capture_loop(args.source, args.gstreamer)
        return
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
