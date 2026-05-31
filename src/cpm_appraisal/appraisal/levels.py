"""Deciding whether a given (event, SEC) is processed schematically or
conceptually.

Report Table 1 assigns levels per check based on whether the check engages an
automatic cue (schematic, fast) or requires reasoning about identity, history,
or moral responsibility (conceptual, slow).

The default policy here is a transparent, rule-based stand-in: keyword cues in
the event description bump a check to conceptual. This is deliberately simple
and easy to defend in the report ("we used a fixed rule-based level assignment")
and easy to swap for an LLM-judged version later.

IMPORTANT (flag for the report): this is a *design choice*, not measured human
appraisal time. The supervisor's feedback (meeting 28.05) said: do NOT assume
more complex events take longer -- so the policy must be driven by event content,
not hard-coded by complexity level. This implementation respects that: level is
decided per (event, check) from cues, not from the scenario's complexity label.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..types import Event, ProcessingLevel, SEC

# Cue words that, if present in an event description, make a given check
# conceptual (require deliberate reasoning). Extend as needed.
CONCEPTUAL_CUES: dict[SEC, tuple[str, ...]] = {
    SEC.RELEVANCE: ("father", "identity", "accus", "fail"),
    SEC.IMPLICATION: ("blame", "fault", "history", "moved out", "consequence"),
    SEC.COPING: ("divorc", "relationship", "uncertain", "depends"),
    SEC.NORMATIVE: ("norm", "moral", "judge", "shame", "wrong", "child present"),
}


@dataclass
class ProcessingLevelPolicy:
    cues: dict[SEC, tuple[str, ...]] = field(default_factory=lambda: CONCEPTUAL_CUES)

    def level_for(self, event: Event, sec: SEC) -> ProcessingLevel:
        text = event.description.lower()
        if any(cue in text for cue in self.cues.get(sec, ())):
            return ProcessingLevel.CONCEPTUAL
        return ProcessingLevel.SCHEMATIC


def default_policy() -> ProcessingLevelPolicy:
    return ProcessingLevelPolicy()
