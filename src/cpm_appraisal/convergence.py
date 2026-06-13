"""Detecting convergence from an appraisal trajectory.

For each AppraisalStep we build the emotion distribution from the (partial)
appraisal vector. Convergence is then tracked by how much that DISTRIBUTION
moves between consecutive steps, measured with Jensen-Shannon divergence:

    converged_at = earliest step from which the consecutive JS divergence stays
                   below `js_threshold` for every remaining step.

This replaces the earlier entropy-delta test. The reason: entropy only measures
how *uncertain* the distribution is, not *which* emotion it favours. Two steps
can have nearly identical entropy while probability mass shifts from one emotion
to another -- same uncertainty, different answer. Because the research questions
are about whether the early emotion equals the final emotion, the criterion must
watch distribution identity, not just its spread. JS does that; entropy does not.

Entropy is still reported, as a descriptive trace of uncertainty over time, but
it is no longer the convergence test.

The JS primitive and the stable-step search are imported from reporting.py so
that the convergence point reported here is computed the SAME way as in the
enriched run record: one criterion, one definition of converged_at. (Ideally
these shared primitives would live in a small dedicated helpers module; importing
from reporting is a pragmatic stand-in and introduces no import cycle, since
reporting depends only on types.)

NOTE on the boolean: `converged` here means converged *before completion* -- the
distribution settled and at least one further check confirmed it. A run that
only goes quiet on its very last step has reached the end of processing, not
convergence, so it is reported as converged=False. This matches the
`converged_before_completion` field in reporting.build_run_record (NOT its looser
`converged` flag).
"""
from __future__ import annotations

from .types import AppraisalStep, ConvergencePoint, EmotionDistribution
from .emotions.classify import appraisal_to_distribution, entropy
from .reporting import js_divergence, _first_stable_step


def analyse_trajectory(
    scenario_id: str,
    complexity_level: int,
    trajectory: list[AppraisalStep],
    prototypes: dict[str, dict[str, float]],
    weights: dict[str, float] | None = None,
    js_threshold: float = 0.01,
    temperature: float = 1.0,
) -> ConvergencePoint:
    """Turn a list of AppraisalStep snapshots into a ConvergencePoint.

    For each step we (1) convert the partial appraisal vector to a distribution
    over the 13 emotion prototypes (Manhattan distance -> softmin), (2) record
    its Shannon entropy, and (3) measure the JS divergence from the previous
    step. Convergence is the earliest step from which that consecutive JS stays
    below `js_threshold` until the end, provided at least one further step
    confirms it.

    Signature change from the previous version: the entropy-based `threshold`
    and `patience` parameters are replaced by a single `js_threshold`.
    """
    if not trajectory:
        empty = appraisal_to_distribution(
            {}, prototypes, weights=weights, temperature=temperature
        )
        return ConvergencePoint(
            scenario_id=scenario_id,
            complexity_level=complexity_level,
            converged=False,
            converged_at_tau=None,
            final_distribution=empty,
            entropy_trace=[],
        )

    # 1-2. distribution and entropy at each step
    dists: list[EmotionDistribution] = []
    entropy_trace: list[float] = []
    for i, step in enumerate(trajectory):
        print(f"\n------- APPRAISAL STEP {i+1}, SEC {step.sec}, completed SECs {len(step.completed_secs)} --------\n")
        dist = appraisal_to_distribution(
            step.vector, prototypes, weights=weights, temperature=temperature, verbose=True
        )
        dists.append(dist)
        entropy_trace.append(entropy(dist))

    # 3. consecutive Jensen-Shannon divergence over the probability dicts;
    #    index 0 has no predecessor.
    js_from_prev: list[float | None] = [None]
    for i in range(1, len(dists)):
        js_from_prev.append(
            js_divergence(dists[i - 1].probabilities, dists[i].probabilities)
        )

    # earliest step from which the distribution stays still until the end
    stable_at = _first_stable_step(js_from_prev, js_threshold)

    # require >=1 confirming step: a run that only stabilises on its final step
    # is the end of processing, not convergence before completion.
    converged_before_completion = (
        stable_at is not None and stable_at < len(trajectory) - 1
    )
    converged_at_tau = (
        trajectory[stable_at].tau if converged_before_completion else None
    )

    return ConvergencePoint(
        scenario_id=scenario_id,
        complexity_level=complexity_level,
        converged=converged_before_completion,
        converged_at_tau=converged_at_tau,
        final_distribution=dists[-1],
        entropy_trace=entropy_trace,
    )