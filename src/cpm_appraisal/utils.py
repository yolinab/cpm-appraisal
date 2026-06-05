"""Shared internal utilities."""
from __future__ import annotations


def extract_json(text: str, open_char: str = "{", close_char: str = "}") -> str:
    """Strip surrounding prose from an LLM response, returning the JSON substring."""
    text = text.strip()
    start, end = text.find(open_char), text.rfind(close_char)
    if start != -1 and end != -1:
        return text[start : end + 1]
    return text
