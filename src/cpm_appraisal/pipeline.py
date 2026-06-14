"""LangGraph implementation of the appraisal timeline.

Equivalent to run_appraisal_timeline in scheduler.py but structured as a
StateGraph. This enables two things the plain loop cannot do:

  1. Cross-event coupling (cycles): once cross_event_coupling is switched on
     in SchedulerConfig, a back-edge from "advance" → "run_sec" lets completed
     appraisals loop back and re-bias in-flight earlier checks (the "DECISION"
     box in Fig. 2). The hook is already in should_continue(); just uncomment
     the cross-coupling branch when the feature is ready.

  2. Checkpointing: pass a LangGraph checkpointer to build_appraisal_graph()
     and long DelftBlue runs survive job preemption — resume from the last
     completed SEC rather than restarting from scratch.

Check costs come from config.cost_model (durations.py): schematic checks cost
one step, conceptual checks cost a per-SEC number of steps derived from neural
appraisal latencies, replacing the old flat 1:3 ratio.

Usage:

    from cpm_appraisal.pipeline import run_appraisal_timeline_graph

    trajectory = run_appraisal_timeline_graph(events, run_check, config=cfg)

run_appraisal_timeline_graph is a drop-in replacement for the scheduler's
run_appraisal_timeline.
"""
from __future__ import annotations

import random
from operator import add
from typing import Annotated

from typing_extensions import TypedDict
from langgraph.graph import END, START, StateGraph

from .convergence import analyse_trajectory
from .generation import PERSONA_SYSTEM, generate_scenario
from .llm import LanguageModel
from .scheduler import RunCheck, SchedulerConfig
from .secs import run_sec
from .types import AppraisalStep, AppraisalVector, ConvergencePoint, Event, Scenario, SEC


class AppraisalState(TypedDict):
    events: list[Event]
    event_index: int          # which event we are currently appraising
    sec_index: int            # which SEC within that event (index into SEC.ordered())
    running_vector: AppraisalVector
    completed_secs: list[SEC]
    trajectory: Annotated[list[AppraisalStep], add]  # reducer appends each new step
    tau: int                  # current position on the appraisal timeline
    done: bool                # set True to signal the conditional edge to stop


def build_appraisal_graph(
    run_check: RunCheck,
    policy: ProcessingLevelPolicy | None = None,
    config: SchedulerConfig | None = None,
):
    """Compile and return a LangGraph app for the appraisal timeline.

    run_check, policy, and config are closed over so the returned app is
    self-contained. Invoke with an initial AppraisalState dict; the final
    state's "trajectory" key is the list of AppraisalStep snapshots.

    To add checkpointing:
        from langgraph.checkpoint.memory import MemorySaver
        app = build_appraisal_graph(run_check, checkpointer=MemorySaver())
    """
    config = config or SchedulerConfig()
    secs = SEC.ordered()

    def run_sec_node(state: AppraisalState) -> dict:
        event_index = state["event_index"]
        sec_index = state["sec_index"]

        # guard: all events consumed
        if event_index >= len(state["events"]):
            return {"done": True}

        event = state["events"][event_index]
        sec = secs[sec_index]

        # levels.py picks the mode; durations.py turns (sec, mode) into a step cost
        ratings, latency_ms = run_check(sec, event.description, dict(state["running_vector"]))
        cost = config.cost_model.steps_from_ms(latency_ms)
        if state["tau"] + cost > config.t_max:
            return {"done": True}
        new_vector = {**state["running_vector"], **ratings}
        new_completed = list(state["completed_secs"])
        if sec not in new_completed:
            new_completed.append(sec)
        new_tau = state["tau"] + cost

        # advance the (event, sec) cursor
        new_sec_index = sec_index + 1
        new_event_index = event_index
        if new_sec_index >= len(secs):
            new_sec_index = 0
            new_event_index += 1

        return {
            "running_vector": new_vector,
            "completed_secs": new_completed,
            # returning a single-element list; the `add` reducer appends it
            "trajectory": [AppraisalStep(
                event_id=event.id,
                tau=new_tau,
                completed_secs=new_completed,
                vector=dict(new_vector),
                sec=sec,
                latency_ms=latency_ms,
                step_cost=cost,
            )],
            "tau": new_tau,
            "event_index": new_event_index,
            "sec_index": new_sec_index,
            "done": False,
        }

    def should_continue(state: AppraisalState) -> str:
        if state["done"]:
            return END

        # cross_event_coupling back-edge goes here:
        # if config.cross_event_coupling:
        #     ... return "run_sec" targeting an earlier event's in-flight SEC

        return "run_sec"

    graph = StateGraph(AppraisalState)
    graph.add_node("run_sec", run_sec_node)
    graph.add_edge(START, "run_sec")
    graph.add_conditional_edges("run_sec", should_continue)

    return graph.compile()


def run_appraisal_timeline_graph(
    events: list[Event],
    run_check: RunCheck,
    policy: ProcessingLevelPolicy | None = None,
    config: SchedulerConfig | None = None,
) -> list[AppraisalStep]:
    """Drop-in replacement for run_appraisal_timeline using LangGraph."""
    app = build_appraisal_graph(run_check, policy, config)
    result = app.invoke({
        "events": events,
        "event_index": 0,
        "sec_index": 0,
        "running_vector": {},
        "completed_secs": [],
        "trajectory": [],
        "tau": 0,
        "done": False,
    })
    return result["trajectory"]


def appraise_scenario(
    llm: LanguageModel,
    complexity_level: int,
    prototypes: dict[str, dict[str, float]],
    weights: dict[str, float] | None = None,
    scheduler_config: SchedulerConfig | None = None,
    seed: int | None = None,
    temperature: float = 0.3,
) -> tuple[Scenario, list[AppraisalStep], ConvergencePoint]:
    """Run the full pipeline for one complexity level.

    Returns the scenario, the appraisal trajectory (needed for the enriched run
    record), and the ConvergencePoint.
    """
    scenario = generate_scenario(llm, complexity_level)

    # sample one event
    rng = random.Random(seed)
    sampled_event = rng.choice(scenario.events)
    scenario.events = [sampled_event]

    def run_check(sec, event_description, prior):
        return run_sec(llm, sec, event_description, prior, PERSONA_SYSTEM)

    trajectory = run_appraisal_timeline_graph(
        scenario.events, run_check, config=scheduler_config
    )
    result = analyse_trajectory(
        scenario.scenario_id, complexity_level, trajectory,
        prototypes, weights=weights, temperature=temperature,
    )
    return scenario, trajectory, result

