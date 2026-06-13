"""Shared internal utilities."""
from __future__ import annotations

import re


def extract_json(text: str, open_char: str = "{", close_char: str = "}") -> str:
    """Strip surrounding prose from an LLM response, returning the JSON substring.

    Handles common LLM output patterns:
    - markdown code fences (```json ... ``` or ``` ... ```)
    - trailing commas before ] or } (invalid JSON but common LLM output)
    """
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    start, end = text.find(open_char), text.rfind(close_char)
    if start != -1 and end != -1:
        extracted = text[start : end + 1]
        # strip // line comments (invalid JSON but common LLM output)
        extracted = re.sub(r"//[^\n]*", "", extracted)
        # strip trailing commas before ] or }
        extracted = re.sub(r",\s*([}\]])", r"\1", extracted)
        return extracted
    return text
