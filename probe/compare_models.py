"""
Head-to-head comparison of local VLMs on wafer map classification.

Tests each model on the same set of synthetic wafer maps and reports
classification accuracy, response quality, and inference speed.

Usage:
    python -m probe.compare_models
"""

import base64
import io
import json
import time

import numpy as np
import requests
from PIL import Image


OLLAMA_URL = "http://localhost:11434/api/chat"

MODELS = ["qwen2.5vl:7b", "gemma3:4b", "minicpm-v"]

SYSTEM_PROMPT = (
    "You are a semiconductor process engineer analyzing wafer map defect patterns. "
    "The image shows a wafer map where dark gray = off-wafer, green = normal die, "
    "red = defective die."
)

PROMPT = (
    "Classify the defect pattern in this wafer map. "
    "Choose exactly one: Center, Edge-Ring, Edge-Loc, Scratch, Random, Near-Full, None. "
    "Reply with ONLY the class name on the first line, then a brief explanation."
)


def make_wafer_image(wafer: np.ndarray, scale: int = 4) -> str:
    colormap = {0: [40, 40, 40], 1: [100, 180, 100], 2: [220, 60, 60]}
    h, w = wafer.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for val, color in colormap.items():
        img[wafer == val] = color

    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((w * scale, h * scale), Image.NEAREST)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_test_wafers():
    wafers = []
    size = 64
    cr, cc = 31.5, 31.5
    rr, ccarr = np.mgrid[:size, :size]
    dist = np.sqrt((rr - cr) ** 2 + (ccarr - cc) ** 2)
    max_r = np.sqrt(cr ** 2 + cc ** 2)

    w = np.ones((size, size), dtype=np.uint8)
    w[dist <= 8] = 2
    wafers.append(("Center", w.copy()))

    w = np.ones((size, size), dtype=np.uint8)
    w[dist >= 0.85 * max_r] = 2
    wafers.append(("Edge-Ring", w.copy()))

    w = np.ones((size, size), dtype=np.uint8)
    w[32, :] = 2
    w[33, :] = 2
    wafers.append(("Scratch", w.copy()))

    rng = np.random.default_rng(42)
    w = np.ones((size, size), dtype=np.uint8)
    w[rng.random((size, size)) < 0.05] = 2
    wafers.append(("Random", w.copy()))

    w = np.ones((size, size), dtype=np.uint8)
    wafers.append(("None", w.copy()))

    return wafers


def query_ollama(model: str, image_b64: str) -> tuple:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": PROMPT,
                "images": [image_b64],
            },
        ],
        "stream": False,
        "options": {"temperature": 0.0},
    }

    start = time.perf_counter()
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        elapsed = time.perf_counter() - start
        data = resp.json()
        text = data.get("message", {}).get("content", "")
        return text, elapsed
    except Exception as e:
        elapsed = time.perf_counter() - start
        return f"ERROR: {e}", elapsed


def extract_class(text: str) -> str:
    compound = {
        "edge-ring": "Edge-Ring", "edge ring": "Edge-Ring",
        "edge-loc": "Edge-Loc", "edge loc": "Edge-Loc",
        "near-full": "Near-Full", "near full": "Near-Full",
    }
    text_lower = text.lower()
    for pattern, label in compound.items():
        if pattern in text_lower:
            return label

    simple = {"center": "Center", "scratch": "Scratch",
              "random": "Random", "none": "None"}
    first_line = text.split("\n")[0].strip().lower()
    for key, val in simple.items():
        if key in first_line:
            return val

    for key, val in simple.items():
        if key in text_lower:
            return val

    return "Unknown"


def main():
    test_wafers = generate_test_wafers()
    available_models = []

    print("Checking available models...")
    for model in MODELS:
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={"model": model, "messages": [{"role": "user", "content": "hi"}], "stream": False},
                timeout=30
            )
            if resp.status_code == 200:
                available_models.append(model)
                print(f"  {model}: available")
            else:
                print(f"  {model}: not available (HTTP {resp.status_code})")
        except Exception as e:
            print(f"  {model}: not available ({e})")

    if not available_models:
        print("No models available. Pull at least one with: ollama pull <model>")
        return

    print(f"\nRunning comparison on {len(test_wafers)} wafer patterns x {len(available_models)} models")
    print("=" * 70)

    results = {m: {"correct": 0, "total": 0, "times": [], "responses": []} for m in available_models}

    for gt_label, wafer in test_wafers:
        image_b64 = make_wafer_image(wafer)
        print(f"\nPattern: {gt_label} (defects: {np.sum(wafer == 2)})")
        print("-" * 70)

        for model in available_models:
            text, elapsed = query_ollama(model, image_b64)
            predicted = extract_class(text)
            correct = predicted.lower() == gt_label.lower()

            results[model]["total"] += 1
            results[model]["times"].append(elapsed)
            results[model]["responses"].append(text[:200])
            if correct:
                results[model]["correct"] += 1

            status = "CORRECT" if correct else "WRONG"
            first_line = text.split("\n")[0][:60]
            print(f"  {model:20s} | {elapsed:5.1f}s | {predicted:10s} | {status} | {first_line}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':20s} | {'Accuracy':>8s} | {'Avg Time':>8s} | {'Total Time':>10s}")
    print("-" * 60)

    for model in available_models:
        r = results[model]
        acc = r["correct"] / r["total"] if r["total"] > 0 else 0
        avg_t = np.mean(r["times"])
        total_t = np.sum(r["times"])
        print(f"{model:20s} | {acc:7.0%}  | {avg_t:7.1f}s | {total_t:9.1f}s")

    print()
    print("For the full probe, estimated time per model:")
    for model in available_models:
        avg_t = np.mean(results[model]["times"])
        est_50 = avg_t * 350
        print(f"  {model}: ~{est_50/60:.0f} minutes for 50 wafers")


if __name__ == "__main__":
    main()
