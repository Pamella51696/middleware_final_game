"""Three perspective ROIs from a wide stitched strip (left / center / right)."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .perspective import resize_for_vlm

ROI_SIZE = (384, 384)


def split_triple_rois(
    bgr: np.ndarray,
    overlap_frac: float = 0.08,
) -> Dict[str, np.ndarray]:
    h, w = bgr.shape[:2]
    third = w / 3.0
    overlap = third * overlap_frac
    spans = {
        "left": (0.0, third + overlap),
        "center": (third - overlap, 2 * third + overlap),
        "right": (2 * third - overlap, float(w)),
    }
    out: Dict[str, np.ndarray] = {}
    for name, (x0, x1) in spans.items():
        a = max(0, int(x0))
        b = min(w, int(x1))
        crop = bgr[:, a:b]
        out[name] = resize_for_vlm(crop, ROI_SIZE)
    return out
