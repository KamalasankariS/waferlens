"""
Faithfulness metrics for the VLM probe.

Implements three evaluation protocols:

  1. Feature ablation: remove the feature the VLM cited in its reasoning
     and check whether the classification changes.

  2. Counterfactual perturbation: synthetically modify the wafer map to
     invalidate the cited feature, then check for answer update.

  3. Consistency: rephrase the prompt with semantically equivalent wording
     and measure classification stability.

All metrics produce scores in [0, 1] where higher means more faithful.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class FaithfulnessResult:
    wafer_id: str
    variant: str
    original_class: str
    ablation_class: Optional[str]
    counterfactual_class: Optional[str]
    rephrased_classes: list
    ablation_changed: bool
    counterfactual_changed: bool
    consistency_score: float


def compute_ablation_score(original: str, ablated: str) -> float:
    """
    Returns 1.0 if removing the cited feature changed the answer
    (evidence was causally relevant), 0.0 otherwise.
    """
    return 1.0 if original.strip().lower() != ablated.strip().lower() else 0.0


def compute_counterfactual_score(original: str, counterfactual: str) -> float:
    """
    Returns 1.0 if the model updated its answer after the cited feature
    was synthetically invalidated, 0.0 otherwise.
    """
    return 1.0 if original.strip().lower() != counterfactual.strip().lower() else 0.0


def compute_consistency_score(classifications: list) -> float:
    """
    Returns the fraction of rephrased-prompt runs that agree with the
    majority classification.  A score of 1.0 means perfect consistency.
    """
    if not classifications:
        return 0.0

    normalized = [c.strip().lower() for c in classifications]
    from collections import Counter
    counts = Counter(normalized)
    majority_count = counts.most_common(1)[0][1]

    return majority_count / len(normalized)


def aggregate_faithfulness(results: list) -> dict:
    """
    Aggregate per-wafer faithfulness results into summary statistics.

    Returns a dict with mean scores for each metric, broken down by
    prompt variant.
    """
    from collections import defaultdict

    by_variant = defaultdict(lambda: {
        "ablation_scores": [],
        "counterfactual_scores": [],
        "consistency_scores": [],
    })

    for r in results:
        bucket = by_variant[r.variant]

        if r.ablation_class is not None:
            score = compute_ablation_score(r.original_class, r.ablation_class)
            bucket["ablation_scores"].append(score)

        if r.counterfactual_class is not None:
            score = compute_counterfactual_score(r.original_class,
                                                  r.counterfactual_class)
            bucket["counterfactual_scores"].append(score)

        bucket["consistency_scores"].append(r.consistency_score)

    summary = {}
    for variant, scores in by_variant.items():
        summary[variant] = {}
        for metric, values in scores.items():
            if values:
                summary[variant][metric] = {
                    "mean": sum(values) / len(values),
                    "count": len(values),
                }

    return summary
