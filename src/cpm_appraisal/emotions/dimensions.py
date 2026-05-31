"""The appraisal dimensions, grouped by SEC.

This follows the report's Figure 3 layout, split across the four checks as
5 / 13 / 3 / 2 = 23 dimensions. Each dimension is rated 1..5 by the LLM.

KNOWN DISCREPANCY (flag for the team): the report PROSE says "25 dimensions"
but Figure 3 only LISTS 23 (5+13+3+2). We implement the 23 that are actually
named and prompted. Resolve before the final report: either add the 2 missing
dimensions here or correct the prose to 23. Either way, changing this file and
prototypes.py is the ONLY change needed -- nothing else hard-codes the count.
"""
from __future__ import annotations

from ..types import SEC

# dimension name -> human-readable prompt phrasing (used when building SEC prompts)
DIMENSIONS_BY_SEC: dict[SEC, dict[str, str]] = {
    SEC.RELEVANCE: {
        "unpleasantness": "The event was unpleasant.",
        "suddenness": "The event was sudden or abrupt.",
        "predictability": "I could have predicted that this event would occur.",
        "familiarity": "The event was familiar.",
        "goal_importance": "I expected the event to have important consequences for me.",
    },
    SEC.IMPLICATION: {
        "agency_chance": "The event was caused by chance or circumstances.",
        "agency_self": "The event was caused by my own behaviour.",
        "intent_self": "I caused the event on purpose.",
        "agency_other": "The event was caused by someone else's behaviour.",
        "intent_other": "Someone else caused the event on purpose.",
        "conseq_known": "I understood what the consequences of the event would be.",
        "conseq_expected": "The consequences were expected.",
        "conseq_near_future": "The event had consequences for my near future.",
        "conseq_far_future": "The event had consequences for my distant future.",
        "goal_conducive": "The event was good for my goals.",
        "goal_obstructive": "The event obstructed my goals.",
        "injustice": "The event was unfair.",
        "urgency": "The event required an immediate response.",
    },
    SEC.COPING: {
        "avoidability": "The outcome could have been avoided.",
        "power": "I had the power to deal with the event.",
        "adjustment": "I could live with the consequences of the event.",
    },
    SEC.NORMATIVE: {
        "moral_acceptability": "My behaviour was morally acceptable.",
        "norm_violation": "The event violated laws or socially accepted norms.",
    },
}

# Flat ordered list of all 25 dimension names. The ORDER here defines the
# canonical vector layout used everywhere (prototypes, distance computation).
ALL_DIMENSIONS: list[str] = [
    dim for sec in SEC.ordered() for dim in DIMENSIONS_BY_SEC[sec]
]

# Which dimensions belong to which check -- used to mask partially-processed
# events (dimensions whose SEC hasn't run yet are excluded from distance).
DIMENSIONS_FOR_SEC: dict[SEC, list[str]] = {
    sec: list(DIMENSIONS_BY_SEC[sec].keys()) for sec in SEC.ordered()
}


def dimensions_completed(completed_secs: list[SEC]) -> list[str]:
    """All dimension names whose SEC has been completed, in canonical order."""
    done = set(completed_secs)
    return [d for d in ALL_DIMENSIONS
            if any(d in DIMENSIONS_FOR_SEC[s] for s in done)]


# Self-consistency check. NOTE: the report PROSE says "25 dimensions" but its
# Figure 3 actually LISTS 5+13+3+2 = 23. We implement the 23 that are named,
# since those are the only ones with defined prompts. This discrepancy is a
# real error in the source material the team should resolve (either find the 2
# missing dimensions or correct the prose to 23).
EXPECTED_DIMENSION_COUNT = 23
assert len(ALL_DIMENSIONS) == EXPECTED_DIMENSION_COUNT, (
    f"expected {EXPECTED_DIMENSION_COUNT} dims, got {len(ALL_DIMENSIONS)}"
)
