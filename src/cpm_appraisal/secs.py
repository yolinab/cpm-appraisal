"""The appraisal module: running one SEC on one event.

Each SEC asks the LLM (as Alex) to rate just that check's dimensions on a 1..5
scale, given the event and the ratings already produced by earlier checks
(sequential-cumulative: later checks see earlier answers).
"""
from __future__ import annotations

import json

from .emotions.dimensions import DIMENSIONS_BY_SEC
from .llm import LanguageModel
from .types import AppraisalVector, SEC
from .utils import extract_json

LATENCY_INTERVALS: dict[SEC, tuple[int, int]] = {
    SEC.RELEVANCE:   (100,  250),
    SEC.IMPLICATION: (450, 800),
    SEC.COPING:      (500, 800),
    SEC.NORMATIVE:   (500, 800),
}


def sec_prompt(sec: SEC, event_description: str, prior: AppraisalVector) -> str:
    dims = DIMENSIONS_BY_SEC[sec]
    lines = "\n".join(f'- "{name}": {phrasing}' for name, phrasing in dims.items())
    prior_block = (
        f"\nYour earlier ratings so far: {json.dumps(prior)}\n" if prior else "\n"
    )
    interval = LATENCY_INTERVALS[sec]
    return f"""You are appraising this event as Alex.

EVENT: {event_description}
{prior_block}
Now perform the {sec.value.upper()} check. Rate each item below from 1 (does not
apply at all) to 5 (applies extremely). Build on your earlier ratings; stay
consistent.

{lines}

Also estimate how long this check would take to process, in milliseconds.
Consider only the content and demands of this specific event and check.
Use a value between {interval[0]} and {interval[1]}.
Return this as "_latency_ms" in your response.

Return ONLY a JSON object mapping each item name to its integer rating,
plus "_latency_ms" to your millisecond estimate."""


def run_sec(
    llm: LanguageModel,
    sec: SEC,
    event_description: str,
    prior: AppraisalVector,
    persona_system: str,
) -> tuple[AppraisalVector, float]:
    """
    Builds a prompt and calls the LLM.
    """
    raw = llm.generate(
        system=persona_system, user=sec_prompt(sec, event_description, prior)
    ).text
    ratings = _parse_ratings(raw)
    latency_ms = float(ratings.pop("_latency_ms", LATENCY_INTERVALS[sec][0]))
    valid = set(DIMENSIONS_BY_SEC[sec].keys())
    return {k: _clamp(v) for k, v in ratings.items() if k in valid}, latency_ms


def _parse_ratings(raw: str) -> dict[str, float]:
    try:
        return {k: float(v) for k, v in json.loads(extract_json(raw)).items()}
    except json.JSONDecodeError:
        print(f"[_parse_ratings] raw LLM output:\n{raw}\n")
        raise


def _clamp(v: float) -> float:
    return max(1.0, min(5.0, v))
