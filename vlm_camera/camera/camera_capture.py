from __future__ import annotations

from typing import Iterator, Optional

import cv2
import numpy as np

from .gstreamer_pipeline import (
    jetson_csi_pipeline,
    jetson_rtsp_pipeline,
    mjpeg_http_pipeline,
)


def open_capture(
    source: str | int = 0,
    *,
    use_gstreamer: bool = False,
) -> cv2.VideoCapture:
    """Open CSI / RTSP / MJPEG / file / device index."""
    if isinstance(source, int):
        if use_gstreamer:
            return cv2.VideoCapture(jetson_csi_pipeline(source), cv2.CAP_GSTREAMER)
        return cv2.VideoCapture(source)

    src = str(source)
    if use_gstreamer and src.startswith("rtsp://"):
        return cv2.VideoCapture(jetson_rtsp_pipeline(src), cv2.CAP_GSTREAMER)
    if use_gstreamer and src.startswith("http"):
        return cv2.VideoCapture(mjpeg_http_pipeline(src), cv2.CAP_GSTREAMER)
    return cv2.VideoCapture(src)


def frames(cap: cv2.VideoCapture) -> Iterator[np.ndarray]:
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        yield frame


def read_bgr(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img
