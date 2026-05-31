"""The appraisal module: running one SEC on one event.

Each SEC asks the LLM (as Alex) to rate just that check's dimensions on a 1..5
scale, given the event and the ratings already produced by earlier checks
(sequential-cumulative: later checks see earlier answers).
"""
from __future__ import annotations

import json

from ..emotions.dimensions import DIMENSIONS_BY_SEC
from ..llm import LanguageModel
from ..types import AppraisalVector, SEC


def sec_prompt(sec: SEC, event_description: str, prior: AppraisalVector) -> str:
    dims = DIMENSIONS_BY_SEC[sec]
    lines = "\n".join(f'- "{name}": {phrasing}' for name, phrasing in dims.items())
    prior_block = (
        f"\nYour earlier ratings so far: {json.dumps(prior)}\n" if prior else "\n"
    )
    return f"""You are appraising this event as Alex.

EVENT: {event_description}
{prior_block}
Now perform the {sec.value.upper()} check. Rate each item below from 1 (does not
apply at all) to 5 (applies extremely). Build on your earlier ratings; stay
consistent.

{lines}

Return ONLY a JSON object mapping each item name to its integer rating."""


def run_sec(
    llm: LanguageModel,
    sec: SEC,
    event_description: str,
    prior: AppraisalVector,
    persona_system: str,
) -> AppraisalVector:
    raw = llm.generate(
        system=persona_system, user=sec_prompt(sec, event_description, prior)
    ).text
    ratings = _parse_ratings(raw)
    # keep only this SEC's dimensions, clamp to 1..5
    valid = set(DIMENSIONS_BY_SEC[sec].keys())
    return {k: _clamp(v) for k, v in ratings.items() if k in valid}


def _parse_ratings(raw: str) -> dict[str, float]:
    raw = raw.strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start : end + 1]
    return {k: float(v) for k, v in json.loads(raw).items()}


def _clamp(v: float) -> float:
    return max(1.0, min(5.0, v))
