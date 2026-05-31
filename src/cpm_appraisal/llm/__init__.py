"""LLM backend abstraction.

The entire pipeline talks to `LanguageModel`, never to a concrete provider.
This is deliberate: we develop against `MockLLM` (free, deterministic, runs
anywhere, no keys, no GPU), and swap in a real backend only once the pipeline
logic is proven correct.

Three backends are provided:
  - MockLLM            : deterministic fake, for development & CI.
  - OpenAICompatLLM    : any OpenAI-compatible HTTP endpoint
                         (Groq / Together / local vLLM server). Stubbed.
  - LocalTransformersLLM : in-process HuggingFace model, for DelftBlue. Stubbed.

DelftBlue note: compute nodes have no internet, so on the cluster you must use
LocalTransformersLLM (or a vLLM server launched inside the same job), NOT an
HTTP API. Develop with MockLLM, validate the science with a hosted free tier,
scale up on DelftBlue last.
"""
from __future__ import annotations

import hashlib
import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    backend: str


class LanguageModel(ABC):
    """Minimal interface every backend implements."""

    @abstractmethod
    def generate(self, system: str, user: str, *, temperature: float = 0.7) -> LLMResponse:
        ...


class MockLLM(LanguageModel):
    """Deterministic fake model.

    Given the same (system, user) it always returns the same output, so tests
    and full pipeline runs are reproducible. It inspects the prompt for keywords
    to decide what *kind* of output to fake (a narrative, a timeline JSON, or an
    appraisal JSON), so the end-to-end pipeline actually runs.
    """

    def __init__(self, seed: int = 0):
        self.seed = seed

    def _rng(self, *parts: str) -> random.Random:
        h = hashlib.sha256(("|".join(parts) + str(self.seed)).encode()).hexdigest()
        return random.Random(int(h[:8], 16))

    def generate(self, system: str, user: str, *, temperature: float = 0.7) -> LLMResponse:
        low = (system + user).lower()
        # Order matters: check most specific intents first.
        if "json array" in low or "decompose" in low:
            return LLMResponse(self._fake_timeline(user), "mock")
        if "applies extremely" in low or "rate each item" in low:
            return LLMResponse(self._fake_appraisal(user), "mock")
        return LLMResponse(self._fake_narrative(user), "mock")

    def _fake_narrative(self, user: str) -> str:
        return (
            "It is early evening in the apartment. Sara sits across the table. "
            "She slides a calendar toward me and points at the week ahead. Mia "
            "is on the floor with a picture book. Sara's voice is even as she "
            "lists the days. I tap the table and read down the column. The light "
            "from the window is going orange. (MOCK NARRATIVE)"
        )

    def _fake_timeline(self, user: str) -> str:
        rng = self._rng(user)
        n = rng.randint(6, 9)
        events = [
            {"id": f"e{i+1}", "description": f"mock event {i+1}"} for i in range(n)
        ]
        return json.dumps(events)

    def _fake_appraisal(self, user: str) -> str:
        import re
        rng = self._rng(user)
        # SEC prompts list each dimension as: - "dim_name": phrasing
        names = re.findall(r'-\s*"([a-z_]+)":', user)
        if not names:
            from ..emotions.dimensions import ALL_DIMENSIONS
            names = ALL_DIMENSIONS
        vec = {name: rng.randint(1, 5) for name in names}
        return json.dumps(vec)


class OpenAICompatLLM(LanguageModel):
    """OpenAI-compatible HTTP backend (Groq, Together, local vLLM server).

    Stubbed for now. Fill in with the `openai` client pointed at base_url.
    """

    def __init__(self, model: str, base_url: str, api_key: str | None = None):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key

    def generate(self, system: str, user: str, *, temperature: float = 0.7) -> LLMResponse:
        raise NotImplementedError(
            "OpenAICompatLLM: wire up the openai client here. "
            "from openai import OpenAI; client = OpenAI(base_url=self.base_url, api_key=...)"
        )


class LocalTransformersLLM(LanguageModel):
    """In-process HuggingFace model for DelftBlue compute nodes. Stubbed."""

    def __init__(self, model_id: str):
        self.model_id = model_id

    def generate(self, system: str, user: str, *, temperature: float = 0.7) -> LLMResponse:
        raise NotImplementedError(
            "LocalTransformersLLM: load tokenizer + model with transformers/vLLM here."
        )


def build_llm(backend: str = "mock", **kwargs) -> LanguageModel:
    """Factory used by config / CLI so the backend is a one-line switch."""
    if backend == "mock":
        return MockLLM(**kwargs)
    if backend == "openai_compat":
        return OpenAICompatLLM(**kwargs)
    if backend == "local_transformers":
        return LocalTransformersLLM(**kwargs)
    raise ValueError(f"unknown backend: {backend}")
