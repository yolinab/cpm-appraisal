"""The appraisal dimensions, grouped by SEC.

Split across the four checks as
5 / 13 / 3 / 2 = 23 dimensions. Each dimension is rated 1..5 by the LLM.
"""
from __future__ import annotations

from ..types import SEC

# dimension name -> human-readable prompt phrasing (used when building SEC prompts)
DIMENSIONS_BY_SEC: dict[SEC, dict[str, str]] = {
    SEC.RELEVANCE: {
        "Pleas": "Was this event pleasent?",
        "Unpleas": "Was this event unpleasant?",
        "Sudden": "The event happened very suddenly and abruptly?",
        "Predict": "You could have predicted the occurrence of the event?",
        "Famil": "You are familiar with this type of event?",
        "Impcons": "The event would have very important consequences for you?"
    },
    SEC.IMPLICATION: {
        "Chance": "Chance, special circumstances, or natural forces?",
        "Ownbehav": "Your own behavior?",
        "Intent": "— If so, did you cause the event intentionally?",
        "Othbehav": "The behavior of one or more other person(s)?",
        "Othint": "— If so, did (this) these other person(s) cause the event intentionally?",
        "Consfelt": "At the time of experiencing the emotion, did you think that real or potential consequences of the event… had already been felt by you or were completely predictable?",
        "Consexp": "At the time of experiencing the emotion, did you think that real or potential consequences of the event… had been expected to occur at that time and in that specific form?",
        "Conearfu": "At the time of experiencing the emotion, did you think that real or potential consequences of the event… could be clearly envisaged and might occur in the near future with a fairly high probability?",
        "Cofarfu": "At the time of experiencing the emotion, did you think that real or potential consequences of the event… were somewhat unpredictable but might occur in the distant future (with uncertain probability)?",
        "Posoutc": "At the time of experiencing the emotion, did you think that real or potential consequences of the event… did or would bring about positive, desirable outcomes for you (e.g., helping you to reach a goal, giving pleasure, or terminating an unpleasant situation)?",
        "Negoutc": "At the time of experiencing the emotion, did you think that real or potential consequences of the event… did or would bring about negative, undesirable outcomes for you (e.g., preventing you from reaching a goal or satisfying a need, resulting in bodily harm, or producing unpleasant feelings)?",
        "Unjust": "At the time of experiencing the emotion, did you think that real or potential consequences of the event… were or would be unjust or unfair?",
        "Urgact": "That it was urgent to act immediately?"
    },
    SEC.COPING: {
        "Avoidabl": "At the time of experiencing the emotion, did you think that real or potential consequences of the event… could have been or could still be avoided or modified by appropriate human action?",
        "Modifcon": "That you would be able to avoid the consequences or modify them to your advantage (through your own power or helped by others)?",
        "Adjustcon": "That you could live with, and adjust to, the consequences of the event that could not possibly be avoided or modified?"
    },
    SEC.NORMATIVE: {
        "Moralacc": "The actions that produced the event were morally and ethically acceptable?",
        "Violnorm": "The actions that produced the event violated laws or social norms?",
        "Consimag": "— If so, was your behavior consistent with the image you have of yourself?"
    }
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


EXPECTED_DIMENSION_COUNT = 25 
assert len(ALL_DIMENSIONS) == EXPECTED_DIMENSION_COUNT, (
    f"expected {EXPECTED_DIMENSION_COUNT} dims, got {len(ALL_DIMENSIONS)}"
)
