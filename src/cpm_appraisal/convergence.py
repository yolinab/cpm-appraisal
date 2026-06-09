"""Detecting convergence from an appraisal trajectory.

For each AppraisalStep we:
  1. build the emotion distribution from the (partial) appraisal vector,
  2. compute its entropy,
giving an entropy trace over the appraisal timeline.

Convergence = the point at which entropy stops changing
meaningfully: |H(tau) - H(tau-1)| < threshold, sustained. We report the first
tau where the change stays below threshold for `patience` consecutive steps.
"""
from __future__ import annotations

from .emotions.classify import appraisal_to_distribution, entropy
from .types import AppraisalStep, ConvergencePoint, EmotionDistribution


def analyse_trajectory(
    scenario_id: str,
    complexity_level: int,
    trajectory: list[AppraisalStep],
    prototypes: dict[str, dict[str, float]],
    weights: dict[str, float] | None = None,
    threshold: float = 0.02,
    patience: int = 2,
    temperature: float = 1.0,
) -> ConvergencePoint:
    """ 
    Takes the list of AppraisalStep snapshots and turns it into a ConvergencePoint. For each 
    AppraisalStep:
    1) computes the Manhattan distance from the current appraisal vector to each of the 13 emotion prototypes,
    2) applies softmin normalization (smaller distance = higher probability) to convert those distances into
       a probability distribution over emotions
    3) computes the Shannon entropy via entropy(dist) of that distribution

    _first_stable finds the step where the appraisal converged. It returns a ConvergencePoint with the appraisal
    step (tau) where it converged, the final distribution over emotions and the entropy trace.
    """
    entropy_trace: list[float] = []
    dists: list[EmotionDistribution] = []
    for step in trajectory:
        dist = appraisal_to_distribution(
            step.vector, prototypes, weights=weights, temperature=temperature
        )
        dists.append(dist)
        entropy_trace.append(entropy(dist))

    converged_at = _first_stable(entropy_trace, threshold, patience)

    final = dists[-1] if dists else appraisal_to_distribution({}, prototypes, weights=weights)
    return ConvergencePoint(
        scenario_id=scenario_id,
        complexity_level=complexity_level,
        converged=converged_at is not None,
        converged_at_tau=trajectory[converged_at].tau if converged_at is not None else None,
        final_distribution=final,
        entropy_trace=entropy_trace,
    )


def _first_stable(trace: list[float], threshold: float, patience: int) -> int | None:
    if len(trace) < patience + 1:
        return None
    stable = 0
    for i in range(1, len(trace)):
        if abs(trace[i] - trace[i - 1]) < threshold:
            stable += 1
            if stable >= patience:
                return i  # index of the step at which stability was reached
        else:
            stable = 0
    return None
