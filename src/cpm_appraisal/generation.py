"""Stage 1-2 of the pipeline: scenario generation.

  seed + complexity level  --scene generator (LLM)-->  static narrative
  static narrative         --timeline generator (LLM)--> ordered events e1..en

The persona is injected into the scene generator's system prompt so the
narrative is written from Alex's first-person point of view.
"""
from __future__ import annotations

import json

from .llm import LanguageModel
from .types import Event, Scenario
from .utils import extract_json

# The three base scenarios + appraisal-conflict structure, from the shared doc.
COMPLEXITY_SITUATIONS: dict[int, str] = {
    1: ("Alex and Sara are together and on good terms. They calmly settle which "
        "of them collects Mia from nursery each day next week. Goals are aligned; "
        "no disagreement. Every appraisal admits a clean, uncontested answer."),
    2: ("Alex and Sara are together but face a trade-off: one of them must reduce "
        "work to care for Mia. Implication is contested (good for Mia, bad for a "
        "career) and coping is contested (control depends on what Sara will do)."),
    3: ("Alex and Sara are divorced. Alex returns to the old apartment; the talk "
        "becomes an accusation that he has failed as a father. Mia is present and "
        "distressed. All four checks are contested."),
}

BASE_SEED = "Two parents meet to discuss childcare arrangements for their young child."


def narrative_prompt(complexity_level: int) -> str:
    situation = COMPLEXITY_SITUATIONS[complexity_level]
    return f"""Write a detailed, static description of a single social situation,
told in the first person from Alex's point of view.

BASE SCENARIO: {BASE_SEED}
SITUATION (complexity level {complexity_level}): {situation}

REQUIRED ELEMENTS: who is present (Alex 35, Sara 34, Mia 4), where and when
(a residential apartment; specify time of day), concrete observable actions and
speech, and sensory/relational detail.

STYLE: first person, present tense, ~350 words, static description (not a
numbered list). Do NOT name or label emotions. Show behaviour and events only.
Return only the narrative text."""


def timeline_prompt(narrative: str) -> str:
    return f'''Decompose the following first-person narrative into an ordered
sequence of discrete events as perceived by Alex.

NARRATIVE:
"""
{narrative}
"""

An event is a single atomic occurrence (one action, one expression change, one
utterance, one thing Alex notices) that can be appraised on its own. Keep
chronological order. Produce between 6 and 12 events. Describe observable
happenings only; do NOT name emotions.

Return ONLY a JSON array of objects with keys "id" and "description":
[{{"id": "e1", "description": "..."}}]'''


PERSONA_SYSTEM = """You are Alex, a 35-year-old man, the father in the situation.
Personality (stable): high conscientiousness, high self-efficacy, high power and
agency, high life satisfaction. Stay in character. Keep reasoning consistent
across steps; do not drift toward a lower-control or more helpless perspective."""


def generate_scenario(llm: LanguageModel, complexity_level: int) -> Scenario:
    narrative = llm.generate(
        system=PERSONA_SYSTEM, user=narrative_prompt(complexity_level)
    ).text

    raw = llm.generate(system="", user=timeline_prompt(narrative)).text
    events = [Event(**e) for e in _parse_events(raw)]

    return Scenario(
        scenario_id=f"L{complexity_level}",
        complexity_level=complexity_level,
        base_seed=BASE_SEED,
        narrative=narrative,
        events=events,
    )


def _parse_events(raw: str) -> list[dict]:
    return json.loads(extract_json(raw, "[", "]"))
