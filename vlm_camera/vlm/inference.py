from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

from .prompt import SCENE_PROMPT
from .qwen_vlm import QwenVlm, dummy_describe


class VlmEngine:
    def __init__(self, backend: Optional[str] = None, load: bool = True):
        self.backend = (backend or os.environ.get("VLM_BACKEND", "auto")).lower()
        self.qwen = QwenVlm(load_4bit=True)
        if load and self.backend in {"auto", "qwen", "transformers"}:
            try:
                self.qwen.load()
            except Exception:
                self.qwen.backend = "dummy"
        if self.backend == "dummy":
            self.qwen.backend = "dummy"

    def run(
        self,
        rois: Dict[str, np.ndarray],
        detections: Optional[List[dict]] = None,
        prompt: str = SCENE_PROMPT,
    ) -> Dict[str, Any]:
        order = ["left", "center", "right"]
        images = [rois[k] for k in order if k in rois]
        if self.qwen.available() and self.backend != "dummy":
            result = self.qwen.describe(images, prompt=prompt, detections=detections)
        else:
            result = dummy_describe(images, detections)
        result["backend"] = self.qwen.backend if self.qwen.available() else "dummy"
        return result
