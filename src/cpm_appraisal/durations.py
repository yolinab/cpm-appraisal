"""Latency model: converting a processing level into a cost in appraisal steps.

levels.py decides WHICH mode a given (event, SEC) is resolved in -- schematic
(fast, automatic) or conceptual (slow, deliberate). This module decides HOW LONG
that mode costs, in appraisal steps (tau).

Previously the scheduler used a single flat schematic:conceptual ratio of 1:3.
That ratio was a free parameter the supervisor objected to. Here the durations
are instead grounded in the appraisal latencies reviewed by Smith & Lane (2015)
(drawing on Brosch & Sander, 2013), which assign characteristic processing times
to different appraisal mechanisms:

    fast tier (~100-140 ms): novelty, concern relevance          -> schematic
    slow tier (~340-800 ms): goal congruence, agency,            -> conceptual
                             norm/value compatibility

Two design points follow:

  1. At dt = 0.1 s, the schematic check (120 ms) rounds to 1 appraisal step and
     the conceptual check (360 ms) rounds to 4 steps. t_max therefore must be set to 
     100 steps to keep the a budget of 10 s.

  2. The conceptual cost is no longer one constant. Each SEC's conceptual mode
     engages a different slow mechanism, so it inherits that mechanism's latency.
     This is what makes processing time vary with event *content*, not with the
     scenario's complexity label (per supervisor feedback, meeting 28.05).

PROVENANCE -- which numbers are cited vs assigned. Smith & Lane do NOT give a
latency for every check:

  schematic, all SECs    ~120 ms  CITED      fast tier, novelty / concern relevance
  Implication conceptual ~700 ms  CITED      goal congruence 340-380 ms (floor) +
                                             agency 450-800 ms (operative). Agency
                                             lives in Implication; the conceptual
                                             implication checks in the report
                                             (fault / consequence attribution) are
                                             agency-type, so the upper end is used.
  Coping conceptual      ~700 ms  INFERENCE  no reviewed mechanism or latency;
                                             assigned by analogy to the agency /
                                             self-vs-situation control machinery.
  Normative conceptual   ~500 ms  ASSIGNED   norm/value compatibility is a slow
                                             cortical mechanism but no latency is
                                             reported; mid slow-tier value.
  Relevance conceptual   ~400 ms  ASSIGNED   the override case (relevance engaging
                                             identity, e.g. e3 in Table 1); no
                                             latency reported.

REMAINING SIMPLIFICATION: conceptual cost is fixed *per SEC*, not per appraisal
sub-dimension. A more granular model would charge a goal-congruence implication
(~4 steps) less than an agency implication (~7 steps). We use one representative
conceptual latency per SEC; this is the natural next refinement.
"""
from __future__ import annotations

from dataclasses import dataclass, field

@dataclass
class StepCostModel:
    """Maps (SEC, processing level) -> cost in appraisal steps.

    Latencies are stored in milliseconds; one appraisal step lasts dt_ms. Steps
    are rounded to the nearest whole step, with a floor of 1 (no check is free).
    """
    dt_ms: float = 100.0
    
    def steps_from_ms(self, latency_ms: float) -> int:
        """Convert a runtime-estimated latency (ms) into whole appraisal steps."""
        return max(1, round(latency_ms / self.dt_ms))


def default_cost_model() -> StepCostModel:
    return StepCostModel()
