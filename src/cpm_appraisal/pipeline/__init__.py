"""End-to-end orchestration: seed -> scenario -> appraisal -> convergence.

This is the function the experiment scripts call. It wires the stages together
but contains no logic of its own beyond sequencing -- each stage lives in its
own module and is independently testable.

NOTE on LangGraph: the proposal commits to LangGraph for orchestration. The
scheduler in appraisal/scheduler.py is a plain Python loop that is a drop-in for
a LangGraph state graph (nodes = SEC checks, edges = sequential/cyclical
transitions, state = running_vector + tau). Porting to LangGraph later means
wrapping each `run_check` as a node and the budget test as the loop-exit
condition; the data types do not change. Starting plain keeps the project
runnable today and makes the LangGraph version a refactor, not a rewrite.
"""
from __future__ import annotations

from ..appraisal.scheduler import SchedulerConfig, run_appraisal_timeline
from ..appraisal.secs import run_sec
from ..convergence import analyse_trajectory
from ..generation import PERSONA_SYSTEM, generate_scenario
from ..llm import LanguageModel
from ..types import ConvergencePoint, Scenario


def run_one(
    llm: LanguageModel,
    complexity_level: int,
    prototypes: dict[str, dict[str, float]],
    scheduler_config: SchedulerConfig | None = None,
) -> tuple[Scenario, ConvergencePoint]:
    scenario = generate_scenario(llm, complexity_level)

    def run_check(sec, event_description, prior):
        return run_sec(llm, sec, event_description, prior, PERSONA_SYSTEM)

    trajectory = run_appraisal_timeline(
        scenario.events, run_check, config=scheduler_config
    )

    result = analyse_trajectory(
        scenario.scenario_id, complexity_level, trajectory, prototypes
    )
    return scenario, result
