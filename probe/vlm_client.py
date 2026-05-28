"""
Local VLM client for the faithfulness probe.

Uses Ollama for fully local, zero-cost inference with vision-language
models. No API keys, no internet, no billing.
"""

import base64
import io
import json
import re
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import requests
from PIL import Image


VALID_CLASSES = {
    "center", "edge-ring", "edge-loc", "scratch",
    "random", "near-full", "none"
}

OLLAMA_URL = "http://localhost:11434/api/chat"


@dataclass
class VLMResponse:
    classification: str
    root_causes: list
    cited_features: list
    raw_text: str


def encode_wafer_image(wafer_rgb: np.ndarray, scale: int = 4) -> str:
    """Encode an RGB wafer image as a base64 PNG string."""
    h, w = wafer_rgb.shape[:2]
    img = Image.fromarray(wafer_rgb)
    img = img.resize((w * scale, h * scale), Image.NEAREST)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def parse_classification(text: str) -> str:
    """Extract the defect class from the VLM response text."""
    text_lower = text.lower()

    compound_map = {
        "edge-ring": "Edge-Ring",
        "edge ring": "Edge-Ring",
        "edge-loc": "Edge-Loc",
        "edge loc": "Edge-Loc",
        "near-full": "Near-Full",
        "near full": "Near-Full",
    }
    for pattern_str, label in compound_map.items():
        if pattern_str in text_lower:
            return label

    single_word_map = {
        "center": "Center",
        "scratch": "Scratch",
        "random": "Random",
        "none": "None",
    }
    for key, val in single_word_map.items():
        pattern = rf'\b{key}\b'
        if re.search(pattern, text_lower):
            return val

    for line in text.split("\n"):
        line_stripped = line.strip().lower()
        if line_stripped.startswith("classification:") or line_stripped.startswith("class:"):
            after_colon = line_stripped.split(":", 1)[1].strip()
            for cls in VALID_CLASSES:
                if cls in after_colon:
                    return cls.replace("-", " ").title().replace(" ", "-")

    return "Unknown"


def parse_cited_features(text: str) -> list:
    """Extract which geometric features the VLM cited in its reasoning."""
    feature_keywords = {
        "radial density": "radial_density",
        "radial profile": "radial_density",
        "edge concentration": "edge_concentration",
        "edge ratio": "edge_concentration",
        "ring score": "ring_score",
        "ring pattern": "ring_score",
        "cluster": "clusters",
        "connected component": "clusters",
        "eccentricity": "clusters",
        "scratch": "scratch_linearity",
        "linearity": "scratch_linearity",
        "linear feature": "scratch_linearity",
        "radon": "scratch_linearity",
        "gradient": "density_gradient",
        "density gradient": "density_gradient",
        "angular": "angular_histogram",
        "sector": "angular_histogram",
    }

    cited = set()
    text_lower = text.lower()
    for keyword, feature in feature_keywords.items():
        if keyword in text_lower:
            cited.add(feature)

    return sorted(cited)


def parse_root_causes(text: str) -> list:
    """Extract root cause proposals from the VLM response."""
    causes = []
    lines = text.split("\n")

    for line in lines:
        stripped = line.strip()
        if re.match(r'^[\d]+[\.\)]\s', stripped) or stripped.startswith("- "):
            cause_text = re.sub(r'^[\d]+[\.\)]\s*', '', stripped)
            cause_text = re.sub(r'^-\s*', '', cause_text)
            if len(cause_text) > 10:
                causes.append(cause_text)

    return causes[:3]


class ProbeVLMClient:
    """Ollama-based local VLM client. Zero cost, full privacy."""

    def __init__(self, model: str = "qwen2.5vl:7b"):
        self.model = model
        self._verify_connection()

    def _verify_connection(self):
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=5)
            if resp.status_code != 200:
                raise ConnectionError("Ollama returned non-200 status")
            models = [m["name"] for m in resp.json().get("models", [])]
            if self.model not in models:
                available = ", ".join(models) if models else "none"
                raise ValueError(
                    f"Model '{self.model}' not found in Ollama. "
                    f"Available: {available}. "
                    f"Pull it with: ollama pull {self.model}"
                )
        except requests.ConnectionError:
            raise ConnectionError(
                "Cannot connect to Ollama at localhost:11434. "
                "Start it with: ollama serve"
            )

    def query(self, system_message: str, user_message: str,
              image_b64: Optional[str] = None) -> VLMResponse:
        messages = [{"role": "system", "content": system_message}]

        user_msg = {"role": "user", "content": user_message}
        if image_b64:
            user_msg["images"] = [image_b64]

        messages.append(user_msg)

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.0, "num_ctx": 2048},
        }

        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()

        raw_text = resp.json().get("message", {}).get("content", "")

        return VLMResponse(
            classification=parse_classification(raw_text),
            root_causes=parse_root_causes(raw_text),
            cited_features=parse_cited_features(raw_text),
            raw_text=raw_text
        )
