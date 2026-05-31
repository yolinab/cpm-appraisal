"""The temporal appraisal scheduler -- the heart of the project.

CONCEPTUAL FRAMING (three "worlds", which collapse to two clocks in code):

  Objective world : the full narrative -- everything that happens. (Produced
                    upstream by the scene generator; not a clock.)
  Perceived world : what the persona has taken in so far == the prefix of
                    events revealed up to now on the EVENT timeline.
  Appraisal world : the internal processing of what's been perceived == the
                    APPRAISAL timeline (tau), where SEC checks run.

So only two clocks exist in code:
  EVENT timeline      e1, e2, ...   objective order events become perceivable.
  APPRAISAL timeline  tau=1,2,...   subjective processing time (steps of dt).

There is no separate "subconscious thread" per event. The thing the supervisor
gestures at -- processing lagging behind arrival -- is captured by OVERLAP on a
single shared appraisal clock: a later event can start being appraised before
an earlier one finishes. See docs/diagrams/02_shared_clock_cross_event.png.

Mechanics:
  - Each (event, SEC) pair is resolved at ONE processing LEVEL: schematic
    (fast, 1 step) or conceptual (slow, 3 steps), assigned by levels.py.
    NB: schematic vs conceptual is the DEPTH of a check, not a parallel copy
    of it -- a check is resolved one way or the other, never both.
  - A global budget t_max (in steps) bounds processing; when spent, appraisal
    halts and the current emotion state is the outcome.
  - Every time a check COMPLETES we emit a snapshot of the whole appraisal
    state so far. Those snapshots are the trajectory convergence is measured on.
    See docs/diagrams/01_appraisal_module_single_event.png for one event's view.

This module is LLM-agnostic except for the injected `run_check` callable, so it
is unit-testable with a trivial fake.

SIMPLIFYING ASSUMPTION (flagged for the report): appraisal accumulates across
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

    # The "DECISION" box from the shared-clock diagram (docs/diagrams/02):
    # should the running appraisal state loop back and bias events still being
    # processed (cross-event coupling / "background loop")?
    #
    # OFF (default): events accumulate sequentially into the shared vector; no
    #   later event retroactively re-biases an earlier one. This is the safe,
    #   tractable choice for the 2-week deadline. The recursive re-appraisal it
    #   omits is already named as a LIMITATION in the report, so excluding it
    #   keeps us self-consistent.
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
