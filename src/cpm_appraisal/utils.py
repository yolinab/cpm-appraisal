"""Shared internal utilities."""
from __future__ import annotations

import re


def extract_json(text: str, open_char: str = "{", close_char: str = "}") -> str:
    """Strip surrounding prose from an LLM response, returning the JSON substring.

    Handles common LLM output patterns:
    - markdown code fences (```json ... ``` or ``` ... ```)
    - trailing commas before ] or } (invalid JSON but common LLM output)
    """
    # strip <think>...</think> blocks (Qwen3 and other reasoning models)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    start, end = text.find(open_char), text.rfind(close_char)
    if start != -1 and end != -1:
        extracted = text[start : end + 1]
        # strip // line comments (invalid JSON but common LLM output)
        extracted = re.sub(r"//[^\n]*", "", extracted)
        # strip trailing commas before ] or }
        extracted = re.sub(r",\s*([}\]])", r"\1", extracted)
        # close truncated last object in an array (model cut off before closing })
        if open_char == "[" and extracted.endswith("]"):
            inner = extracted[:-1].rstrip()
            if inner and inner[-1] not in ("}", "]"):
                extracted = inner + "}]"
        return extracted
    return text
