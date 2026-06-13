"""Turn an appraisal trajectory into an informative, analysis-ready record.

This is the reporting / serialisation layer. It does not run the LLM or the
scheduler; it consumes the AppraisalStep trajectory they produce and derives the
quantities the three research questions actually need:

  CONVERGENCE (RQ1, RQ3)
    - per-step emotion distribution + Shannon entropy (raw bits and normalised)
    - Jensen-Shannon divergence between CONSECUTIVE steps
      (stability: has the distribution stopped moving?)
    - Jensen-Shannon divergence of each step from the FINAL distribution
      (validity risk: how far were early annotations from the eventual one?)
    - convergence point: earliest tau from which consecutive JS stays below a
      threshold, plus the fraction of total processing elapsed by then

  TIMING (RQ2, and the supervisor's question)
    - per-SEC reported latency (ms) and step cost
    - total processing time for the event, in steps (tau) and ms
    - whether the budget was exhausted before all four checks completed

It reuses the caller's distance->distribution function (inject `to_distribution`)
so the probability model stays IDENTICAL to convergence.py rather than being
re-implemented here. A Manhattan-distance default is provided only as a fallback.

NOTE on entropy: `entropy` is normalised to [0, 1] (divided by log2 of the number
of emotions), which matches the H(X) in [0,1] stated in the report. `entropy_bits`
is the raw Shannon value, kept for completeness.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

from .types import AppraisalStep, AppraisalVector, Scenario, SEC

EmotionDist = dict[str, float]


# --------------------------------------------------------------------------- #
# Information-theoretic helpers
# --------------------------------------------------------------------------- #
def shannon_entropy(dist: EmotionDist) -> float:
    """Raw Shannon entropy in bits."""
    return -sum(p * math.log2(p) for p in dist.values() if p > 0.0)


def normalised_entropy(dist: EmotionDist) -> float:
    """Shannon entropy divided by log2(n) -> [0, 1]. 1 = uniform/undecided."""
    n = len(dist)
    if n <= 1:
        return 0.0
    return shannon_entropy(dist) / math.log2(n)


def _kl(p: EmotionDist, q: EmotionDist) -> float:
    # caller guarantees q[k] > 0 wherever p[k] > 0 (q is a mixture m)
    total = 0.0
    for k, pk in p.items():
        if pk > 0.0:
            total += pk * math.log2(pk / q[k])
    return total


def js_divergence(p: EmotionDist, q: EmotionDist) -> float:
    """Jensen-Shannon divergence (base-2 logs, so in [0, 1])."""
    keys = set(p) | set(q)
    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in keys}
    p = {k: p.get(k, 0.0) for k in keys}
    q = {k: q.get(k, 0.0) for k in keys}
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _topk(dist: EmotionDist, k: int) -> list[tuple[str, float]]:
    return sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:k]


def _first_stable_step(js_from_prev: list[Optional[float]], threshold: float) -> Optional[int]:
    """Earliest index i such that every consecutive JS at j > i is < threshold.

    Interpretation: from step i onward the distribution no longer moves more
    than `threshold` between checks -- i.e. it has settled.
    """
    n = len(js_from_prev)
    for i in range(n):
        if all((js_from_prev[j] is not None and js_from_prev[j] < threshold)
               for j in range(i + 1, n)):
            return i
    return None


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def build_run_record(
    scenario: Scenario,
    trajectory: list[AppraisalStep],
    to_distribution: Callable[[AppraisalVector], EmotionDist],
    *,
    js_threshold: float = 0.01,
    top_k: int = 2,
    include_full_distribution: bool = False,
) -> dict:
    """Produce the enriched, analysis-ready record for one scenario/event.

    `to_distribution` maps an appraisal vector to an emotion distribution; pass
    convergence.py's function so the probability model is identical there.
    """
    dists = [to_distribution(step.vector) for step in trajectory]
    final = dists[-1] if dists else {}

    js_prev: list[Optional[float]] = []
    steps_out: list[dict] = []
    for i, (step, dist) in enumerate(zip(trajectory, dists)):
        jp = js_divergence(dists[i - 1], dist) if i > 0 else None
        js_prev.append(jp)
        rec = {
            "sec": i,
            "tau": step.tau,
            "sec_completed": step.sec.value if step.sec else None,
            "completed_secs": [s.value for s in step.completed_secs],
            "latency_ms": step.latency_ms,
            "step_cost": step.step_cost,
            "entropy": normalised_entropy(dist),
            "entropy_bits": shannon_entropy(dist),
            "js_from_prev": jp,
            "js_from_final": js_divergence(dist, final),
            "top": _topk(dist, top_k),
        }
        if include_full_distribution:
            rec["distribution"] = dist
        steps_out.append(rec)

    n = len(steps_out)
    conv_step = _first_stable_step(js_prev, js_threshold)
    total_tau = trajectory[-1].tau if trajectory else 0
    converged_at_tau = steps_out[conv_step]["tau"] if conv_step is not None else None

    total_ms = sum((step.latency_ms or 0.0) for step in trajectory)
    all_secs_done = bool(trajectory) and len(trajectory[-1].completed_secs) == len(SEC.ordered())

    divs_from_final = [s["js_from_final"] for s in steps_out]
    event = scenario.events[0] if scenario.events else None

    return {
        "scenario": {
            "scenario_id": scenario.scenario_id,
            "complexity_level": scenario.complexity_level,
            "base_seed": scenario.base_seed,
            "narrative": scenario.narrative,
            "event": ({"id": event.id, "description": event.description} if event else None),
        },
        "timing": {
            "total_tau": total_tau,
            "total_ms": total_ms,
            "budget_exhausted": not all_secs_done,
            "per_sec": [
                {
                    "sec": step.sec.value if step.sec else None,
                    "latency_ms": step.latency_ms,
                    "step_cost": step.step_cost,
                    "cumulative_tau": step.tau,
                }
                for step in trajectory
            ],
        },
        "convergence": {
            "converged": conv_step is not None,
            "converged_at_sec": conv_step,
            "converged_at_tau": converged_at_tau,
            "converged_before_completion": (conv_step is not None and conv_step < n - 1),
            "secs_total": n,
            "fraction_processed_before_convergence": (
                converged_at_tau / total_tau
                if (converged_at_tau is not None and total_tau) else None
            ),
            "criterion": {"measure": "js_consecutive", "threshold": js_threshold},
            "final_entropy": normalised_entropy(final),
            "final_entropy_bits": shannon_entropy(final),
            "initial_div_from_final": (divs_from_final[0] if divs_from_final else 0.0),
            "max_div_from_final": (max(divs_from_final) if divs_from_final else 0.0),
        },
        "trajectory": steps_out,
        "final_distribution": (final if include_full_distribution else _topk(final, top_k)),
    }
