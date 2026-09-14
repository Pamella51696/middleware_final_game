"""Fast detector: YOLOv8n / TensorRT on Jetson, dummy boxes otherwise."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, List, Optional

import numpy as np


COCO_VEHICLES = {"car", "truck", "bus", "motorcycle", "bicycle"}
COCO_PEOPLE = {"person"}


@dataclass
class DetectedObject:
    cls: str
    confidence: float
    bbox: List[int]  # x1, y1, x2, y2

    def to_dict(self) -> dict:
        d = asdict(self)
        d["class"] = d.pop("cls")
        return d


class ObjectDetector:
    def __init__(self, weights: Optional[str] = None, conf: float = 0.35):
        self.conf = conf
        self._model = None
        self.backend = "dummy"
        if weights:
            self._try_load(weights)

    def _try_load(self, weights: str) -> None:
        try:
            from ultralytics import YOLO  # type: ignore

            self._model = YOLO(weights)
            self.backend = "yolo"
        except Exception:
            self._model = None
            self.backend = "dummy"

    def detect(self, bgr: np.ndarray) -> List[DetectedObject]:
        if self._model is None or bgr is None or bgr.size == 0:
            return []
        results = self._model.predict(bgr, conf=self.conf, verbose=False)
        out: List[DetectedObject] = []
        for r in results:
            names = r.names
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                name = str(names.get(cls_id, cls_id))
                xyxy = [int(v) for v in box.xyxy[0].tolist()]
                out.append(
                    DetectedObject(
                        cls=name,
                        confidence=float(box.conf[0]),
                        bbox=xyxy,
                    )
                )
        return out
