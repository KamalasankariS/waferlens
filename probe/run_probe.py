"""
Main entry point for the VLM faithfulness probe.

Runs the three-variant prompt protocol against a local VLM (Qwen2.5-VL
via Ollama) on synthetic wafer maps and produces a faithfulness report
with ablation, counterfactual, and consistency metrics.

Usage:
    python -m probe.run_probe [--num-wafers 50] [--output probe_results.json]

Requires Ollama running locally with a vision model pulled.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

from probe.prompt_builder import (
    build_image_only_prompt,
    build_raw_features_prompt,
    build_grounded_summary_prompt,
)
from probe.faithfulness import (
    FaithfulnessResult,
    compute_ablation_score,
    compute_counterfactual_score,
    compute_consistency_score,
    aggregate_faithfulness,
)
from probe.vlm_client import ProbeVLMClient, encode_wafer_image, VLMResponse
from probe.perturbations import get_counterfactual


REPHRASINGS = [
    (
        "What type of defect pattern is visible on this wafer map? "
        "Select from: Center, Edge-Ring, Edge-Loc, Scratch, Random, Near-Full, None. "
        "Explain your reasoning with evidence."
    ),
    (
        "Analyze the spatial distribution of defects in this wafer image. "
        "Classify as one of: Center, Edge-Ring, Edge-Loc, Scratch, Random, Near-Full, None. "
        "Justify your classification."
    ),
    (
        "Examine this semiconductor wafer map. The red cells indicate defective die. "
        "Determine the defect pattern category (Center, Edge-Ring, Edge-Loc, Scratch, "
        "Random, Near-Full, or None) and state your evidence."
    ),
]


def generate_probe_wafers(n: int, size: int = 64, seed: int = 42) -> list:
    """Generate labeled synthetic wafers for probing."""
    rng = np.random.default_rng(seed)
    wafers = []

    patterns = ["center", "edge_ring", "scratch", "random", "none"]
    per_pattern = n // len(patterns)
    remainder = n % len(patterns)

    for pi, pattern in enumerate(patterns):
        count = per_pattern + (1 if pi < remainder else 0)
        for i in range(count):
            wafer = np.ones((size, size), dtype=np.uint8)
            cr, cc = (size - 1) / 2.0, (size - 1) / 2.0
            rr, ccarr = np.mgrid[:size, :size]
            dist = np.sqrt((rr - cr) ** 2 + (ccarr - cc) ** 2)
            max_r = np.sqrt(cr ** 2 + cc ** 2)

            if pattern == "center":
                radius = rng.uniform(4, size // 4)
                wafer[dist <= radius] = 2

            elif pattern == "edge_ring":
                threshold = rng.uniform(0.78, 0.88) * max_r
                wafer[dist >= threshold] = 2

            elif pattern == "scratch":
                angle = rng.uniform(0, np.pi)
                proj = (rr - cr) * np.cos(angle) + (ccarr - cc) * np.sin(angle)
                wafer[np.abs(proj) < 1.5] = 2

            elif pattern == "random":
                mask = rng.random((size, size)) < rng.uniform(0.02, 0.08)
                wafer[mask] = 2

            ground_truth = {
                "center": "Center",
                "edge_ring": "Edge-Ring",
                "scratch": "Scratch",
                "random": "Random",
                "none": "None",
            }[pattern]

            wafers.append({
                "id": f"{pattern}_{i:03d}",
                "wafer": wafer,
                "ground_truth": ground_truth,
            })

    rng.shuffle(wafers)
    return wafers


def wafer_to_rgb(wafer: np.ndarray) -> np.ndarray:
    colormap = {0: [40, 40, 40], 1: [100, 180, 100], 2: [220, 60, 60]}
    h, w = wafer.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for val, color in colormap.items():
        img[wafer == val] = color
    return img


def features_to_dict(feat) -> dict:
    return {
        "radial_density": list(feat.radial_density),
        "angular_histogram": list(feat.angular_histogram),
        "edge_concentration": feat.edge_concentration,
        "ring_score": feat.ring_score,
        "peak_ring_index": feat.peak_ring_index,
        "clusters": [
            {
                "label": c.label,
                "area": c.area,
                "centroid_row": c.centroid_row,
                "centroid_col": c.centroid_col,
                "eccentricity": c.eccentricity,
            }
            for c in feat.clusters
        ],
        "scratch_linearity": feat.scratch_linearity,
        "scratch_angle": feat.scratch_angle,
        "total_defects": feat.total_defects,
        "total_normal": feat.total_normal,
    }


def run_single_wafer(client, analyzer, wafer_entry, feat_dict, image_b64):
    """Run the full 3-variant protocol plus ablation/counterfactual for one wafer."""
    wid = wafer_entry["id"]
    wafer = wafer_entry["wafer"]
    results = []

    prompt_variants = [
        build_image_only_prompt(),
        build_raw_features_prompt(feat_dict),
        build_grounded_summary_prompt(feat_dict),
    ]

    for prompt in prompt_variants:
        resp = client.query(prompt.system_message, prompt.user_message, image_b64)
        original_class = resp.classification

        ablation_class = None
        if prompt.variant in ("raw_features", "grounded_summary"):
            ablation_prompt_fn = (build_raw_features_prompt
                                  if prompt.variant == "raw_features"
                                  else build_grounded_summary_prompt)
            ablated_dict = {k: v for k, v in feat_dict.items()}
            for cited in resp.cited_features[:1]:
                if cited in ablated_dict:
                    if isinstance(ablated_dict[cited], list):
                        ablated_dict[cited] = []
                    elif isinstance(ablated_dict[cited], (int, float)):
                        ablated_dict[cited] = 0
            ablation_prompt = ablation_prompt_fn(ablated_dict)
            ablation_resp = client.query(
                ablation_prompt.system_message,
                ablation_prompt.user_message,
                image_b64
            )
            ablation_class = ablation_resp.classification

        counterfactual_class = None
        if resp.cited_features and prompt.variant != "image_only":
            primary_cited = resp.cited_features[0]
            cf_wafer = get_counterfactual(wafer, primary_cited)
            cf_rgb = wafer_to_rgb(cf_wafer)
            cf_b64 = encode_wafer_image(cf_rgb)
            cf_feat = analyzer.compute(cf_wafer)
            cf_dict = features_to_dict(cf_feat)
            cf_prompt_fn = (build_raw_features_prompt
                            if prompt.variant == "raw_features"
                            else build_grounded_summary_prompt)
            cf_prompt = cf_prompt_fn(cf_dict)
            cf_resp = client.query(
                cf_prompt.system_message, cf_prompt.user_message, cf_b64
            )
            counterfactual_class = cf_resp.classification

        rephrased_classes = []
        if prompt.variant == "image_only":
            for rephrase in REPHRASINGS:
                reph_resp = client.query(prompt.system_message, rephrase, image_b64)
                rephrased_classes.append(reph_resp.classification)

        all_classes = [original_class] + rephrased_classes
        consistency = compute_consistency_score(all_classes) if rephrased_classes else 1.0

        result = FaithfulnessResult(
            wafer_id=wid,
            variant=prompt.variant,
            original_class=original_class,
            ablation_class=ablation_class,
            counterfactual_class=counterfactual_class,
            rephrased_classes=rephrased_classes,
            ablation_changed=(ablation_class is not None
                              and ablation_class != original_class),
            counterfactual_changed=(counterfactual_class is not None
                                    and counterfactual_class != original_class),
            consistency_score=consistency,
        )
        results.append(result)

    return results


def format_report(all_results, wafer_entries):
    """Generate a human-readable summary report."""
    summary = aggregate_faithfulness(all_results)

    lines = []
    lines.append("=" * 60)
    lines.append("WAFERLENS FAITHFULNESS PROBE REPORT")
    lines.append("=" * 60)
    lines.append(f"Wafers evaluated: {len(wafer_entries)}")
    lines.append(f"Total VLM queries: {len(all_results)}")
    lines.append("")

    for variant, metrics in sorted(summary.items()):
        lines.append(f"--- {variant} ---")
        for metric, stats in sorted(metrics.items()):
            lines.append(f"  {metric}: {stats['mean']:.3f} (n={stats['count']})")
        lines.append("")

    gt_match = {v: {"correct": 0, "total": 0} for v in ["image_only", "raw_features", "grounded_summary"]}
    for r in all_results:
        entry = next((w for w in wafer_entries if w["id"] == r.wafer_id), None)
        if entry:
            gt_match[r.variant]["total"] += 1
            if r.original_class.lower().replace("-", "_") == entry["ground_truth"].lower().replace("-", "_"):
                gt_match[r.variant]["correct"] += 1

    lines.append("--- Classification accuracy (vs ground truth) ---")
    for variant, counts in gt_match.items():
        if counts["total"] > 0:
            acc = counts["correct"] / counts["total"]
            lines.append(f"  {variant}: {acc:.3f} ({counts['correct']}/{counts['total']})")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="waferlens VLM faithfulness probe")
    parser.add_argument("--num-wafers", type=int, default=50)
    parser.add_argument("--wafer-size", type=int, default=64)
    parser.add_argument("--model", type=str, default="qwen2.5vl:7b")
    parser.add_argument("--output", type=str, default="probe_results.json")
    args = parser.parse_args()

    try:
        import waferlens
    except ImportError:
        print("Error: waferlens package not installed.")
        sys.exit(1)

    print("waferlens-probe: VLM faithfulness evaluation (local, zero cost)")
    print(f"  Model:       {args.model}")
    print(f"  Wafers:      {args.num_wafers}")
    print(f"  Wafer size:  {args.wafer_size}x{args.wafer_size}")
    print(f"  Output:      {args.output}")
    print()

    client = ProbeVLMClient(model=args.model)
    analyzer = waferlens.WaferAnalyzer()

    print("Generating synthetic wafer maps...")
    wafer_entries = generate_probe_wafers(args.num_wafers, args.wafer_size)
    print(f"  Generated {len(wafer_entries)} wafers across 5 pattern types")
    print()

    print("Extracting geometric features...")
    feature_cache = {}
    image_cache = {}
    for entry in wafer_entries:
        feat = analyzer.compute(entry["wafer"])
        feature_cache[entry["id"]] = features_to_dict(feat)
        rgb = wafer_to_rgb(entry["wafer"])
        image_cache[entry["id"]] = encode_wafer_image(rgb)
    print("  Done.")
    print()

    all_results = []
    total = len(wafer_entries)

    print("Running probe...")
    start_time = time.time()

    for i, entry in enumerate(wafer_entries):
        wid = entry["id"]
        feat_dict = feature_cache[wid]
        image_b64 = image_cache[wid]

        try:
            results = run_single_wafer(client, analyzer, entry, feat_dict, image_b64)
            all_results.extend(results)
        except Exception as e:
            print(f"  [{i+1}/{total}] {wid}: ERROR - {e}")
            continue

        elapsed = time.time() - start_time
        avg_per_wafer = elapsed / (i + 1)
        remaining = avg_per_wafer * (total - i - 1)

        print(f"  [{i+1}/{total}] {wid}: "
              f"{results[0].original_class} | "
              f"{results[1].original_class} | "
              f"{results[2].original_class}  "
              f"({remaining:.0f}s remaining)")

    elapsed_total = time.time() - start_time
    print()
    print(f"Probe complete in {elapsed_total:.0f}s")
    print()

    report = format_report(all_results, wafer_entries)
    print(report)

    output_data = {
        "config": {
            "model": args.model,
            "num_wafers": args.num_wafers,
            "wafer_size": args.wafer_size,
        },
        "results": [
            {
                "wafer_id": r.wafer_id,
                "variant": r.variant,
                "original_class": r.original_class,
                "ablation_class": r.ablation_class,
                "counterfactual_class": r.counterfactual_class,
                "rephrased_classes": r.rephrased_classes,
                "ablation_changed": r.ablation_changed,
                "counterfactual_changed": r.counterfactual_changed,
                "consistency_score": r.consistency_score,
            }
            for r in all_results
        ],
        "summary": aggregate_faithfulness(all_results),
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Full results written to {args.output}")


if __name__ == "__main__":
    main()
