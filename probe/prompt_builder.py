"""
Prompt construction for the VLM faithfulness probe.

Generates three prompt variants per wafer map:
  1. Image-only: the wafer map visualization with no feature annotations.
  2. Raw features: image + the full WaferFeatures struct as a JSON dump.
  3. Grounded summary: image + a natural-language description derived from
     the geometric features, written to match how a process engineer would
     describe the defect pattern.
"""

import json
from dataclasses import dataclass


@dataclass
class PromptVariant:
    variant: str
    system_message: str
    user_message: str
    feature_context: str


SYSTEM_PROMPT = (
    "You are a semiconductor process engineer analyzing wafer map defect patterns. "
    "Given a wafer map image and optional geometric measurements, classify the "
    "defect pattern and propose likely root causes. Cite specific evidence from "
    "the measurements when available."
)

CLASSIFICATION_QUESTION = (
    "Classify the defect pattern shown in this wafer map. "
    "Choose from: Center, Edge-Ring, Edge-Loc, Scratch, Random, Near-Full, None. "
    "Then propose up to three likely root causes. "
    "For each root cause, cite the specific evidence that supports it."
)


def build_image_only_prompt() -> PromptVariant:
    return PromptVariant(
        variant="image_only",
        system_message=SYSTEM_PROMPT,
        user_message=CLASSIFICATION_QUESTION,
        feature_context=""
    )


def build_raw_features_prompt(features_dict: dict) -> PromptVariant:
    context = json.dumps(features_dict, indent=2)
    message = (
        f"Geometric measurements for this wafer:\n```json\n{context}\n```\n\n"
        f"{CLASSIFICATION_QUESTION}"
    )
    return PromptVariant(
        variant="raw_features",
        system_message=SYSTEM_PROMPT,
        user_message=message,
        feature_context=context
    )


def build_grounded_summary_prompt(features_dict: dict) -> PromptVariant:
    summary_lines = []

    rd = features_dict.get("radial_density", [])
    if rd:
        peak_idx = max(range(len(rd)), key=lambda i: rd[i])
        num_bins = len(rd)
        peak_frac = peak_idx / num_bins
        summary_lines.append(
            f"Radial density peaks at {peak_frac:.2f} of the wafer radius "
            f"(bin {peak_idx}/{num_bins})"
        )

    ec = features_dict.get("edge_concentration", 0)
    summary_lines.append(
        f"Edge concentration ratio: {ec:.3f} "
        f"({'high -- defects cluster at the edge' if ec > 0.5 else 'low -- defects are interior'})"
    )

    rs = features_dict.get("ring_score", 1)
    summary_lines.append(
        f"Ring score: {rs:.2f} "
        f"({'strong ring pattern' if rs > 3.0 else 'moderate' if rs > 1.5 else 'near-uniform radial distribution'})"
    )

    clusters = features_dict.get("clusters", [])
    if clusters:
        areas = [c["area"] for c in clusters]
        eccs = [c["eccentricity"] for c in clusters]
        summary_lines.append(
            f"{len(clusters)} defect cluster(s): "
            f"areas={areas}, eccentricities=[{', '.join(f'{e:.2f}' for e in eccs)}]"
        )
    else:
        summary_lines.append("No defect clusters detected")

    sl = features_dict.get("scratch_linearity", 0)
    sa = features_dict.get("scratch_angle", 0)
    if sl > 0.3:
        summary_lines.append(
            f"Linear feature detected: linearity={sl:.3f}, angle={sa:.1f} degrees"
        )
    else:
        summary_lines.append(f"No strong linear features (linearity={sl:.3f})")

    summary = "\n".join(f"- {line}" for line in summary_lines)
    message = (
        f"Geometric analysis of this wafer:\n{summary}\n\n"
        f"{CLASSIFICATION_QUESTION}"
    )

    return PromptVariant(
        variant="grounded_summary",
        system_message=SYSTEM_PROMPT,
        user_message=message,
        feature_context=summary
    )
