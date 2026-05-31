"""Core domain types for the CPM appraisal pipeline.

These dataclasses are the shared vocabulary of the whole project. Everything
upstream (generation) produces them; everything downstream (convergence)
consumes them. Keeping them in one place means a teammate can read this file
and understand the data flow without touching any logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SEC(str, Enum):
    """The four Stimulus Evaluation Checks, in their canonical order.

    Order matters: each check is meant to build on the previous ones
    (Scherer's sequential-cumulative assumption).
    """

    RELEVANCE = "relevance"
    IMPLICATION = "implication"
    COPING = "coping"
    NORMATIVE = "normative"

    @classmethod
    def ordered(cls) -> list["SEC"]:
        return [cls.RELEVANCE, cls.IMPLICATION, cls.COPING, cls.NORMATIVE]


class ProcessingLevel(str, Enum):
    """Collapsed version of Scherer's processing levels (we use two).

    SCHEMATIC  -> fast, automatic, costs 1 appraisal step.
    CONCEPTUAL -> slow, deliberate, costs 3 appraisal steps.
    """

    SCHEMATIC = "schematic"
    CONCEPTUAL = "conceptual"


@dataclass
class Event:
    """One atomic happening in the decomposed timeline (e1, e2, ...)."""

    id: str
    description: str


@dataclass
class Scenario:
    """A full generated situation at one complexity level."""

    scenario_id: str
    complexity_level: int          # 1 = low, 2 = medium, 3 = high
    base_seed: str                 # the one-sentence base scenario
    narrative: str                 # the static ~350-word description
    events: list[Event] = field(default_factory=list)


# An appraisal vector is a mapping from dimension name -> 1..5 rating.
# We use a dict (not a fixed-length list) so that partially-processed events,
# where later SEC dimensions are still "unspecified", are representable simply
# by leaving those keys absent.
AppraisalVector = dict[str, float]


@dataclass
class AppraisalStep:
    """The model's appraisal state at a single point on the appraisal timeline.

    `vector` contains only the dimensions for SECs completed so far. Unfinished
    dimensions are absent (treated as unspecified in the distance computation).
    """

    event_id: str
    tau: int                                   # index on the appraisal timeline
    completed_secs: list[SEC]
    vector: AppraisalVector
    raw_llm_output: Optional[str] = None       # kept for debugging / inspection


@dataclass
class EmotionDistribution:
    """A probability distribution over the 13 discrete emotion prototypes."""

    probabilities: dict[str, float]            # emotion label -> probability

    def top(self, k: int = 2) -> list[tuple[str, float]]:
        return sorted(self.probabilities.items(), key=lambda kv: kv[1], reverse=True)[:k]


@dataclass
class ConvergencePoint:
    """The result of analysing one scenario's appraisal trajectory."""

    scenario_id: str
    complexity_level: int
    converged: bool
    converged_at_tau: Optional[int]            # None if never converged within budget
    final_distribution: EmotionDistribution
    entropy_trace: list[float] = field(default_factory=list)
