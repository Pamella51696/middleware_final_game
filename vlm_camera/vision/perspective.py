from __future__ import annotations

import cv2
import numpy as np

VLM_SIZE = (768, 384)  # width, height — short wide crop for Qwen2.5-VL


def resize_for_vlm(bgr: np.ndarray, size: tuple[int, int] = VLM_SIZE) -> np.ndarray:
    w, h = size
    if bgr is None or bgr.size == 0:
        return np.zeros((h, w, 3), dtype=np.uint8)
    return cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
