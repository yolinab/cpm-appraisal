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

# The three levels of complexity defined as a dual construct:
# objective situational shifts + subjective friction experienced during appraisal.
COMPLEXITY_SITUATIONS: dict[int, str] = {
    1: ("Low Complexity: Two parents calmly discussing child collection logistics. "
        "This is driven by an isolated change in exactly 1 situational dimension (Object: the schedule). "
        "Because the objective shift is minimal and their goals are aligned, the SECs remain mostly mutually consistent, "
        "resulting in 0 to 1 conflicting checks that admit clean answers without internal friction."),
    2: ("Medium Complexity: Two parents discussing which parent must reduce work hours. "
        "This is caused by simultaneous changes across 2 to 3 situational dimensions (Goal and Object). "
        "This overlapping objective load introduces moderate subjective friction, predictably resulting in exactly 2 contested SECs "
        "(Implication and Coping Potential), as the outcome is incongruent with career goals and depends on the other person."),
    3: ("High Complexity: A heated argument between divorced parents in front of their child. "
        "This is triggered by massive situational shifts across 4 to 5 dimensions (Character, Goal, Setting, and Causality). "
        "This severe operational load triggers deep appraisal friction, resulting in 3 to 4 contested SECs. "
        "The agent must resolve heavy relevance and normative significance loads, severely burdening coping potential and social judgment, requiring maximum processing time."),
}

BASE_SEED = "Two parents meet to discuss childcare arrangements for their young child."


def narrative_prompt(complexity_level: int) -> str:
    situation = COMPLEXITY_SITUATIONS[complexity_level]
    return f"""Write a detailed, static description of a single social situation,
told in the first person from Alex's point of view.

BASE SCENARIO: {BASE_SEED}
SITUATION (complexity level {complexity_level}): {situation}

Note: Complexity in this scenario is a dual construct, combining the objective situational shifts of the event with the subjective friction experienced during appraisal. Ensure the narrative actions implicitly reflect this specific level of complexity.

REQUIRED ELEMENTS: who is present (Alex 35, Sara 34, Mia 4), where and when
(a residential apartment; specify time of day), concrete observable actions and
speech, and sensory/relational detail.

STYLE: first person, present tense, ~350 words, static description (not a
numbered list). Do NOT name or label emotions. Show behaviour and events only.
Return only the narrative text."""


def timeline_prompt(narrative: str) -> str:
    return f'''Decompose the following first-person narrative into an ordered
sequence of discrete events as perceived by Alex, utilizing Event Segmentation Theory (EST).

NARRATIVE:
"""
{narrative}
"""

Under Event Segmentation Theory (EST), baseline operational complexity is quantified objectively. An event boundary occurs when there are shifts across situational dimensions (such as character, setting, object, or goal). 
Segment the narrative into discrete, atomic occurrences strictly based on these objective situational shifts. 
Keep chronological order. Produce between 6 and 12 events. Describe observable happenings only; do NOT name emotions.

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
    try:
        return json.loads(extract_json(raw, "[", "]"))
    except json.JSONDecodeError:
        print(f"[_parse_events] raw LLM output:\n{raw}\n")
        raise