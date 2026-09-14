"""Keep last N VLM snapshots so alerts are not frame-flicker."""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List


class TemporalMemory:
    def __init__(self, maxlen: int = 8):
        self._items: Deque[Dict[str, Any]] = deque(maxlen=maxlen)

    def push(self, scene: Dict[str, Any]) -> None:
        self._items.append(dict(scene))

    def recent_vehicles(self) -> List[str]:
        seen: List[str] = []
        for item in self._items:
            for v in item.get("vehicles") or []:
                if v not in seen:
                    seen.append(v)
        return seen

    def __len__(self) -> int:
        return len(self._items)
