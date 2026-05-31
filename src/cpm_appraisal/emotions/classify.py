"""Turning an appraisal vector into an emotion distribution, and measuring it.

Pipeline (report Section 3.3, Figure 3):
  appraisal vector --Manhattan distance--> distances to 13 prototypes
                   --softmin / normalise--> probability distribution
                   --Shannon entropy--> scalar uncertainty (convergence signal)

Partial vectors: when only some SECs are done, we compare ONLY on the
dimensions present in the vector (the report excludes unspecified dims from the
distance computation).
"""
from __future__ import annotations

import math

from ..types import AppraisalVector, EmotionDistribution
from .prototypes import EMOTION_LABELS


def manhattan(vec: AppraisalVector, proto: dict[str, float]) -> float:
    """L1 distance over the dimensions PRESENT in `vec` only."""
    return sum(abs(vec[d] - proto[d]) for d in vec)


def distances_to_distribution(
    distances: dict[str, float], temperature: float = 1.0
) -> EmotionDistribution:
    """Smaller distance -> higher probability, via a softmin.

    P(e) ∝ exp(-distance(e) / temperature)
    """
    # numerically stable softmin
    neg = {e: -d / temperature for e, d in distances.items()}
    m = max(neg.values())
    exps = {e: math.exp(v - m) for e, v in neg.items()}
    z = sum(exps.values())
    return EmotionDistribution({e: v / z for e, v in exps.items()})


def appraisal_to_distribution(
    vec: AppraisalVector,
    prototypes: dict[str, dict[str, float]],
    temperature: float = 1.0,
) -> EmotionDistribution:
    if not vec:
        # nothing processed yet -> uniform distribution (maximum uncertainty)
        u = 1.0 / len(EMOTION_LABELS)
        return EmotionDistribution({e: u for e in EMOTION_LABELS})
    distances = {e: manhattan(vec, prototypes[e]) for e in prototypes}
    return distances_to_distribution(distances, temperature)


def entropy(dist: EmotionDistribution, normalise: bool = True) -> float:
    """Shannon entropy in bits. If normalise, scale to [0, 1] by log2(n)."""
    h = -sum(p * math.log2(p) for p in dist.probabilities.values() if p > 0)
    if normalise:
        h /= math.log2(len(dist.probabilities))
    return h
