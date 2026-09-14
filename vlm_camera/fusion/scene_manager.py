from __future__ import annotations

from typing import Any, Dict, List, Optional

from vlm_camera.detector.detector import DetectedObject
from vlm_camera.vlm.prompt import normalize_scene

from .temporal_memory import TemporalMemory


class SceneManager:
    def __init__(self) -> None:
        self.memory = TemporalMemory()
        self.latest: Dict[str, Any] = normalize_scene(None)

    def fuse(
        self,
        detections: List[DetectedObject],
        vlm: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        scene = normalize_scene(vlm)
        det_vehicles = []
        det_people = []
        objects = []
        for d in detections:
            objects.append(d.to_dict())
            if d.cls == "person":
                det_people.append("person")
            elif d.cls:
                det_vehicles.append(d.cls.replace(" ", "_"))

        if not scene["vehicles"] and det_vehicles:
            scene["vehicles"] = list(dict.fromkeys(det_vehicles))
        if not scene["people"] and det_people:
            scene["people"] = list(dict.fromkeys(det_people))

        scene["objects"] = objects
        scene["hazard"] = bool(scene["hazards"])
        self.memory.push(scene)
        scene["recent_vehicles"] = self.memory.recent_vehicles()
        self.latest = scene
        return scene
