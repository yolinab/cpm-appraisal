"""The temporal appraisal scheduler

The scheduler orchestrates the appraisal process across multiple events and SECs
  EVENT timeline      e1, e2, ...   objective order events become perceivable.
  APPRAISAL timeline  tau=1,2,...   subjective processing time (steps of dt).

Mechanics:
  - Each (event, SEC) pair is resolved at ONE processing LEVEL: schematic
    (fast, 1 step) or conceptual (slow, 3 steps), assigned by levels.py.
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

from dataclasses import dataclass
from typing import Callable

from ..types import AppraisalStep, AppraisalVector, Event, SEC
from .levels import ProcessingLevelPolicy, default_policy

# A check runner: (sec, event_description, prior_vector) -> ratings for that sec
RunCheck = Callable[[SEC, str, AppraisalVector], AppraisalVector]


@dataclass
class SchedulerConfig:
    t_max: int = 20          # budget in appraisal steps (report: 10s / dt=0.5s)
    schematic_cost: int = 1  # steps a schematic check occupies
    conceptual_cost: int = 3 # steps a conceptual check occupies

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
            level = policy.level_for(event, sec)
            cost = (config.conceptual_cost if level.value == "conceptual"
                    else config.schematic_cost)

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
