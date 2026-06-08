"""The temporal appraisal scheduler

The scheduler orchestrates the appraisal process across multiple events and SECs
  EVENT timeline      e1, e2, ...   objective order events become perceivable.
  APPRAISAL timeline  tau=1,2,...   subjective processing time (steps of dt).

Mechanics:
  - Each (event, SEC) pair is resolved at ONE processing LEVEL: schematic
    (fast) or conceptual (slow), assigned by levels.py.
  - The cost of a check, in appraisal steps, comes from the StepCostModel in
    durations.py. Schematic checks cost one step; conceptual checks cost a
    per-SEC number of steps grounded in the appraisal latencies reviewed by
    Smith & Lane (2015), rather than a single flat schematic:conceptual ratio.
  - A global budget t_max (in steps) bounds processing; when spent, appraisal
    halts and the current emotion state is the outcome.
  - Every time a check COMPLETES we emit a snapshot of the whole appraisal
    state so far. Those snapshots are the trajectory convergence is measured on.

This module is LLM-agnostic except for the injected `run_check` callable, so it
is unit-testable with a trivial fake.

SIMPLIFYING ASSUMPTION: appraisal accumulates across
events into ONE running vector, so "emotion so far" reflects everything
processed up to tau. Cross-event coupling (later events re-biasing in-flight
earlier ones) is OFF by default -- see SchedulerConfig.cross_event_coupling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .types import AppraisalStep, AppraisalVector, Event, SEC
from .levels import ProcessingLevelPolicy, default_policy
from .durations import StepCostModel, default_cost_model

# A check runner: (sec, event_description, prior_vector) -> ratings for that sec
RunCheck = Callable[[SEC, str, AppraisalVector], AppraisalVector]


@dataclass
class SchedulerConfig:
    # Budget in appraisal steps. At dt = 0.1 s, 100 steps = 10 s (report budget).
    t_max: int = 100

    # Per-(SEC, level) step costs, derived from neural appraisal latencies.
    # Replaces the old flat schematic_cost=1 / conceptual_cost=3 ratio.
    cost_model: StepCostModel = field(default_factory=default_cost_model)

    # The "DECISION" box from the shared-clock diagram (Fig. 2 in the readme)
    # should the running appraisal state loop back and bias events still being
    # processed (cross-event coupling / "background loop")?
    #
    # OFF (default): events accumulate sequentially into the shared vector; no
    #   later event retroactively re-biases an earlier one.
    # ON: would require cyclical re-evaluation (LangGraph loops) and a
    #   convergence guarantee we don't have time to validate. Leave for future
    #   work. NOT implemented yet -- setting this True currently does nothing.
    cross_event_coupling: bool = False


def run_appraisal_timeline(
    events: list[Event],
    run_check: RunCheck,
    policy: ProcessingLevelPolicy | None = None,
    config: SchedulerConfig | None = None,
) -> list[AppraisalStep]:
    """Process events through the four SECs along the appraisal timeline.

    Returns one AppraisalStep per COMPLETED check (a trajectory snapshot),
    until the budget t_max is spent or all events are fully appraised.
    """
    policy = policy or default_policy()
    config = config or SchedulerConfig()

    running_vector: AppraisalVector = {}
    completed_secs: list[SEC] = []
    trajectory: list[AppraisalStep] = []
    tau = 0

    for event in events:
        for sec in SEC.ordered():
            # levels.py picks the mode (schematic/conceptual) from event content;
            # durations.py turns (sec, mode) into a cost in appraisal steps.
            level = policy.level_for(event, sec)
            cost = config.cost_model.steps_for(sec, level)

            if tau + cost > config.t_max:
                return trajectory  # budget exhausted mid-process

            # advance the appraisal clock by the cost of this check
            tau += cost

            ratings = run_check(sec, event.description, dict(running_vector))
            running_vector.update(ratings)
            if sec not in completed_secs:
                completed_secs.append(sec)

            trajectory.append(
                AppraisalStep(
                    event_id=event.id,
                    tau=tau,
                    completed_secs=list(completed_secs),
                    vector=dict(running_vector),
                )
            )

    return trajectory
