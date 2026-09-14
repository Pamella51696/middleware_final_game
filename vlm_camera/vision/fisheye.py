"""Drop the unused letterbox around the stitched strip."""

from __future__ import annotations

import numpy as np


def crop_black_bands(
    bgr: np.ndarray,
    luma_min: float = 16.0,
    min_keep_frac: float = 0.18,
) -> np.ndarray:
    if bgr.ndim != 3 or bgr.size == 0:
        return bgr
    gray = bgr.mean(axis=2)
    row_mean = gray.mean(axis=1)
    col_mean = gray.mean(axis=0)
    rows = np.where(row_mean > luma_min)[0]
    cols = np.where(col_mean > luma_min)[0]
    h, w = gray.shape
    if rows.size < h * min_keep_frac or cols.size < w * min_keep_frac:
        return bgr
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    return bgr[y0:y1, x0:x1]
