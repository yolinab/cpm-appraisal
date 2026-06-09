"""The 13 discrete emotion prototypes.

Each prototype is a point in the 23-dimensional appraisal space (each dim 1..5).
An appraisal vector is classified by distance to these prototypes.

The 13 labels:
  anger, boredom, disgust, fear, guilt, joy, no_emotion, pride, relief,
  sadness, shame, surprise, trust

THE NUMBERS BELOW ARE PLACEHOLDERS. They are NOT the real Scherer/crowd-enVENT
profiles - they exist so the pipeline runs end-to-end today.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .dimensions import ALL_DIMENSIONS

EMOTION_LABELS: list[str] = [
    "Sadness", "Joy", "Rage", "Anxiety", "Fear", "Irritation", "Shame",
    "Guilt", "Contempt", "Disgust", "Pleasure", "Despair", "Pride"
]


def _placeholder_prototypes() -> dict[str, dict[str, float]]:
    """Deterministic placeholder profiles so distances are non-degenerate.

    Spreads the 13 emotions across the appraisal space using a simple hash so
    they aren't all identical. TODO: replace with crowd-enVENT values.
    """
    protos: dict[str, dict[str, float]] = {}
    for i, label in enumerate(EMOTION_LABELS):
        vec = {}
        for j, dim in enumerate(ALL_DIMENSIONS):
            # cycle values 1..5 deterministically, offset per emotion
            vec[dim] = float(((i * 7 + j * 3) % 5) + 1)
        protos[label] = vec
    return protos


def _default_weights() -> dict[str, float]:
    """Uniform weights (1.0) for all dimensions."""
    return {dim: 1.0 for dim in ALL_DIMENSIONS}


def load_weights(path: str | Path | None = None) -> dict[str, float]:
    """Load dimension weights from a JSON file, else uniform 1.0."""
    if path is None:
        return _default_weights()
    path = Path(path)
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        return {k: float(v) for k, v in data.items()}
    # Optional: support CSV if needed, but JSON is simpler for a flat dict
    weights = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            # assume CSV has columns: dimension, weight
            weights[row["dimension"]] = float(row["weight"])
    return weights


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
