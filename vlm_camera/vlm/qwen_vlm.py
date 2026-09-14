"""Qwen2.5-VL-3B-Instruct (INT4) with a CPU/dummy fallback for development."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

from .prompt import SCENE_PROMPT, normalize_scene, parse_scene_json

QWEN_MODEL_ID = os.environ.get("VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")


def _bgr_to_pil(bgr: np.ndarray):
    from PIL import Image

    rgb = bgr[:, :, ::-1]
    return Image.fromarray(rgb)


class QwenVlm:
    def __init__(self, model_id: str = QWEN_MODEL_ID, load_4bit: bool = True):
        self.model_id = model_id
        self.load_4bit = load_4bit
        self.backend = "unloaded"
        self._model = None
        self._processor = None

    def load(self) -> str:
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except Exception:
            self.backend = "dummy"
            return self.backend

        kwargs: Dict[str, Any] = {"device_map": "auto"}
        if self.load_4bit:
            try:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=getattr(torch, "float16", torch.float32),
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            except Exception:
                kwargs["torch_dtype"] = torch.float16
        else:
            kwargs["torch_dtype"] = torch.float16

        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, trust_remote_code=True, **kwargs
        )
        self.backend = "qwen4bit" if self.load_4bit else "qwen"
        return self.backend

    def available(self) -> bool:
        return self._model is not None and self._processor is not None

    def describe(
        self,
        images: List[np.ndarray],
        prompt: str = SCENE_PROMPT,
        detections: Optional[List[dict]] = None,
    ) -> Dict[str, Any]:
        if not self.available():
            return dummy_describe(images, detections)

        pil_images = [_bgr_to_pil(im) for im in images if im is not None and im.size]
        det_txt = ""
        if detections:
            det_txt = "\nDetected objects: " + ", ".join(
                f"{d.get('class')}@{d.get('confidence', 0):.2f}" for d in detections
            )
        messages = [
            {
                "role": "user",
                "content": (
                    [{"type": "image", "image": im} for im in pil_images]
                    + [{"type": "text", "text": prompt + det_txt}]
                ),
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        try:
            from qwen_vl_utils import process_vision_info

            image_inputs, video_inputs = process_vision_info(messages)
        except Exception:
            image_inputs, video_inputs = pil_images, None
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)
        out_ids = self._model.generate(**inputs, max_new_tokens=192)
        trimmed = [o[len(i) :] for i, o in zip(inputs.input_ids, out_ids)]
        text_out = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return parse_scene_json(text_out)


def dummy_describe(
    images: List[np.ndarray],
    detections: Optional[List[dict]] = None,
) -> Dict[str, Any]:
    """Deterministic JSON when Qwen weights are not on this machine."""
    vehicles: List[str] = []
    people: List[str] = []
    for d in detections or []:
        name = str(d.get("class") or d.get("cls") or "")
        if name in {"person"}:
            people.append("person")
        elif name:
            vehicles.append(name.replace(" ", "_"))
    # De-dupe preserving order
    vehicles = list(dict.fromkeys(vehicles))
    people = list(dict.fromkeys(people))
    n = sum(1 for im in images if im is not None and getattr(im, "size", 0))
    scene = "residential road" if n else "unknown"
    desc = (
        f"{len(vehicles)} vehicle type(s) visible across {n} view(s)."
        if vehicles
        else f"No detector hits; {n} camera view(s) received."
    )
    return normalize_scene(
        {
            "vehicles": vehicles,
            "people": people,
            "hazards": [],
            "scene": scene,
            "description": desc,
            "confidence": 0.55 if vehicles else 0.25,
        }
    )
