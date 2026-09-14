"""Structured JSON prompt for Qwen2.5-VL-3B (and the dummy backend)."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

SCENE_PROMPT = """Analyze these camera views as one surround scene
(left, center, and right from a vehicle-mounted fisheye stitch).

Identify:
1. vehicles
2. people
3. road condition
4. objects of interest
5. possible hazards

Return JSON only with this exact schema:
{
  "vehicles": ["delivery_truck"],
  "people": [],
  "hazards": [],
  "scene": "residential road",
  "description": "short sentence",
  "confidence": 0.0
}

No markdown. No extra keys. JSON only.
"""

EMPTY_SCENE: Dict[str, Any] = {
    "vehicles": [],
    "people": [],
    "hazards": [],
    "scene": "unknown",
    "description": "",
    "confidence": 0.0,
}


def parse_scene_json(text: str) -> Dict[str, Any]:
    if not text:
        return dict(EMPTY_SCENE)
    raw = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1)
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return dict(EMPTY_SCENE)
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return dict(EMPTY_SCENE)
    return normalize_scene(data)


def normalize_scene(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(EMPTY_SCENE)
    if not isinstance(data, dict):
        return out

    def _as_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value else []
        if isinstance(value, list):
            return [str(v) for v in value if v]
        return []

    out["vehicles"] = _as_list(data.get("vehicles"))
    out["people"] = _as_list(data.get("people"))
    out["hazards"] = _as_list(data.get("hazards"))
    out["scene"] = str(data.get("scene") or "unknown")
    out["description"] = str(data.get("description") or "")
    try:
        out["confidence"] = float(data.get("confidence") or 0.0)
    except (TypeError, ValueError):
        out["confidence"] = 0.0
    out["confidence"] = max(0.0, min(1.0, out["confidence"]))
    return out
