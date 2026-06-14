"""Turning an appraisal vector into an emotion distribution, and measuring it.

Pipeline:
  appraisal vector --Manhattan distance--> distances to 13 prototypes
                   --softmin / normalise--> probability distribution
                   --Shannon entropy--> scalar uncertainty (convergence signal)

Partial vectors: when only some SECs are done, we compare ONLY on the
dimensions present in the vector.
"""
from __future__ import annotations

import math

from ..types import AppraisalVector, EmotionDistribution
from .prototypes import EMOTION_LABELS


def manhattan(
    vec: AppraisalVector, proto: dict[str, float], weights: dict[str, float] | None = None
) -> float:
    """Weighted L1 distance over the dimensions PRESENT in `vec` only."""
    if weights is None:
        return sum(abs(vec[d] - proto[d]) for d in vec)

    return sum(abs(weights[d] * vec[d] - proto[d]) for d in vec)


def distances_to_distribution(
    distances: dict[str, float], temperature: float = 1.0, verbose: bool = False
) -> EmotionDistribution:
    """Smaller distance -> higher probability, via a softmin.

    P(e) ∝ exp(-distance(e) / temperature)
    """
    # numerically stable softmin
    neg = {e: -d / temperature for e, d in distances.items()}
    m = max(neg.values())
    exps = {e: math.exp(v - m) for e, v in neg.items()}
    z = sum(exps.values())
    probs = {e: v / z for e, v in exps.items()}

    if verbose:
        print("\nFinal Probabilities (Softmin):")
        for e in sorted(probs, key=probs.get, reverse=True):
            print(f"  {e:12}: prob={probs[e]:.4f}")
        print("---------------------------------\n")

    return EmotionDistribution(probs)


def appraisal_to_distribution(
    vec: AppraisalVector,
    prototypes: dict[str, dict[str, float]],
    weights: dict[str, float] | None = None,
    temperature: float = 0.3,
    verbose: bool = False,
) -> EmotionDistribution:
    if not vec:
        if verbose:
            print("\n--- Emotion Calculation Debug (Empty Vector) ---")
            u = 1.0 / len(EMOTION_LABELS)
            print(f"  Uniform probability for all {len(EMOTION_LABELS)} emotions: {u:.4f}")
            print("------------------------------------------------\n")
        return EmotionDistribution({e: 1.0 / len(EMOTION_LABELS) for e in EMOTION_LABELS})

    # 1. STANDARDIZE THE INPUT VECTOR (Convert 1-5 scale to Z-scores)
    vals = list(vec.values())
    mean_val = sum(vals) / len(vals)
    variance = sum((v - mean_val) ** 2 for v in vals) / len(vals)
    std_dev = math.sqrt(variance) if variance > 0 else 1.0

    standardized_vec = {d: (v - mean_val) / std_dev for d, v in vec.items()}

    if verbose:
        print(f"\n--- Emotion Calculation Debug (Dimensions: {list(vec.keys())}) ---")
        print("Responses (Appraisal Vector) and Weights:")
        for d, raw_v in vec.items():
            w = weights[d] if weights and d in weights else 1.0
            z_v = standardized_vec[d]
            print(f"  {d:12}: raw={raw_v:.2f}, z-score={z_v:.2f}, weight={w:.2f}")

    distances = {}
    for e, proto in prototypes.items():
        d_sum = 0.0
        w_sum = 0.0
        if verbose:
            print(f"\nDistance calculation for emotion: {e}")

        for d, z_val in standardized_vec.items():
            p_val = proto.get(d)
            if p_val is None or p_val == -100:
                continue

            w = weights[d] if weights and d in weights else 1.0
            weighted_val = w * z_val
            weighted_proto_val = w * p_val
            diff = abs(weighted_val - weighted_proto_val)
            d_sum += diff
            w_sum += w

            if verbose:
                print(f"    {d:12} | (weight {w:.2f} * resp_z {z_val:.2f}) = {weighted_val:.2f} vs proto {weighted_proto_val:.2f} -> diff {diff:.4f}")

        normalized_distance = d_sum / w_sum if w_sum > 0 else float('inf')
        if verbose:
            print(f"  Total raw distance: {d_sum:.4f} | Total weight: {w_sum:.2f} | Normalized distance to {e}: {normalized_distance:.4f}")
        distances[e] = normalized_distance

    return distances_to_distribution(distances, temperature, verbose=verbose)



def entropy(dist: EmotionDistribution, normalise: bool = True) -> float:
    """Shannon entropy in bits. If normalise, scale to [0, 1] by log2(n)."""
    h = -sum(p * math.log2(p) for p in dist.probabilities.values() if p > 0)
    if normalise:
        h /= math.log2(len(dist.probabilities))
    return h
