from __future__ import annotations

import time


class FrameSelector:
    """Keep VLM load at ~1 FPS on an 8 GB Orin Nano."""

    def __init__(self, interval_sec: float = 1.0):
        self.interval_sec = max(0.2, float(interval_sec))
        self._last: float | None = None

    def should_run(self, now: float | None = None) -> bool:
        t = time.monotonic() if now is None else now
        if self._last is None or t - self._last >= self.interval_sec:
            self._last = t
            return True
        return False

    def reset(self) -> None:
        self._last = None
