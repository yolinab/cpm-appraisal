"""The 13 discrete emotion prototypes.

Each prototype is a point in the 25-dimensional appraisal space (each dim 1..5).
An appraisal vector is classified by distance to these prototypes.

The 13 labels match the report / crowd-enVENT:
  anger, boredom, disgust, fear, guilt, joy, no_emotion, pride, relief,
  sadness, shame, surprise, trust

THE NUMBERS BELOW ARE PLACEHOLDERS. They are NOT the real Scherer/crowd-enVENT
profiles -- they exist so the pipeline runs end-to-end today. Replacing them is
a concrete, well-scoped task (see TODO). The crowd-enVENT paper's Figure 8 gives
average appraisal values per emotion; those should be transcribed here and
rescaled to the 25-dim layout in dimensions.py.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .dimensions import ALL_DIMENSIONS

EMOTION_LABELS: list[str] = [
    "anger", "boredom", "disgust", "fear", "guilt", "joy", "no_emotion",
    "pride", "relief", "sadness", "shame", "surprise", "trust",
]


def _placeholder_prototypes() -> dict[str, dict[str, float]]:
    """Deterministic placeholder profiles so distances are non-degenerate.

    Spreads the 13 emotions across the appraisal space using a simple hash so
    they aren't all identical. TODO: replace with crowd-enVENT Figure 8 values.
    """
    protos: dict[str, dict[str, float]] = {}
    for i, label in enumerate(EMOTION_LABELS):
        vec = {}
        for j, dim in enumerate(ALL_DIMENSIONS):
            # cycle values 1..5 deterministically, offset per emotion
            vec[dim] = float(((i * 7 + j * 3) % 5) + 1)
        protos[label] = vec
    return protos


def load_prototypes(path: str | Path | None = None) -> dict[str, dict[str, float]]:
    """Load prototypes from a CSV/JSON file if given, else placeholders.

    CSV format: first column 'emotion', then one column per dimension name.
    """
    if path is None:
        return _placeholder_prototypes()
    path = Path(path)
    if path.suffix == ".json":
        return json.loads(path.read_text())
    protos: dict[str, dict[str, float]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            label = row.pop("emotion")
            protos[label] = {k: float(v) for k, v in row.items()}
    return protos
