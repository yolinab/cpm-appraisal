# CPM Appraisal Convergence Pipeline

DSAIT4230 — Research in Social Signal Processing & Affective Computing, TU Delft.

## What this project does (one paragraph)

We feed an LLM (playing a fixed persona, **Alex**) a social situation. Instead of
asking "what emotion is this?" in one shot, the model appraises the situation
step by step through Scherer's four **Stimulus Evaluation Checks** (Relevance →
Implication → Coping → Normative). At each step the appraisal ratings are turned
into a **probability distribution over 13 discrete emotions**, and we track the
**entropy** of that distribution. The core question: does the emotion *lock in*
(entropy stabilises) before all information is processed, and do **more complex
situations take more steps to lock in**?

## The two timelines (the important idea)

- **Event timeline** (`e1, e2, …`): the situation is revealed gradually as a
  sequence of discrete events.
- **Appraisal timeline** (`τ = 1, 2, …`): within processing, each check costs
  time — *schematic* checks cost 1 step, *conceptual* checks cost 3 — against a
  budget `t_max`. Convergence is the point on this timeline where entropy stops
  changing.

## The "three worlds" framing (and why it's only two clocks in code)

The supervisor talks about three layers. They map cleanly onto what we already
have, and crucially they do **not** require three timelines or parallel threads:

| layer (supervisor's words) | what it actually is | in code |
|---|---|---|
| objective world | the full narrative, all that happens | scene generator output (not a clock) |
| perceived world | what Alex has taken in *so far* | prefix of the event timeline |
| appraisal world | internal processing of the perceived | the appraisal clock τ |

The intuition that processing "runs as a faster background thread" is the right
*instinct* but the wrong *implementation*. What it really means is just:
**a new event can start being appraised before the previous event's appraisal
finishes** — overlap on one shared appraisal clock. No subconscious threads, no
200 ms subliminal machinery. One shared τ with overlapping processing windows.
That is what makes this doable in two weeks.

## Two diagrams (these are the canonical reference)

**1 — Inside the appraisal module (one event).** One event → four checks in
order → each check resolved schematically *or* conceptually (that's its depth,
not a parallel row) → after each check, read off the emotion distribution and
its entropy H → H falls and flattens → convergence.

![Appraisal module for a single event](docs/diagrams/01_appraisal_module_single_event.png)

**2 — The whole situation (events on one shared clock).** Each event's module
collapses to a single bar. Events are staggered on the shared τ clock and their
processing windows overlap. The situation-level emotion is read from all checks
completed so far; H(τ) is tracked until it settles or `t_max` is hit.

![Events on one shared appraisal clock](docs/diagrams/02_shared_clock_cross_event.png)

> **The one open modeling decision** (the dashed "background loop" in diagram 2):
> should the shared state loop back and re-bias events still being processed?
> This is **OFF by default** (`SchedulerConfig.cross_event_coupling = False`).
> Turning it on means cyclical re-appraisal — extra complexity and a convergence
> guarantee we can't validate in two weeks. It's already listed as a *limitation*
> in the report, so excluding it keeps the project self-consistent. Leave it for
> future work.

> **Note on the earlier single-graph draft:** an earlier hand-drawn version put
> schematic and conceptual as two *parallel rows* you travel through, with
> cross-over arrows between checks. That conflates two different axes — *which
> check* (R→I→Co→N) vs. *how deep* (schematic/conceptual). Splitting it into the
> two diagrams above fixes this: the check sequence is one axis, the processing
> level is a per-check property. The code follows the split.

## Architecture

```
seed + complexity ─▶ generation/ ─▶ scenario (narrative + events)
                                          │
                                          ▼
              appraisal/scheduler.py  (two-timeline loop, budget t_max)
                     │  per (event, SEC):
                     │    levels.py  → schematic (1 step) or conceptual (3)
                     │    secs.py    → run that check on the LLM
                     ▼
              trajectory of AppraisalStep
                     │
                     ▼
              emotions/  → distance to 13 prototypes → distribution → entropy
                     │
                     ▼
              convergence/ → first stable τ, entropy trace
```

Every stage is its own module under `src/cpm_appraisal/`, talks only through the
dataclasses in `types.py`, and is independently testable.

## Run it (no API key, no GPU, runs anywhere)

```bash
pip install -e .
pytest -q                      # smoke tests
python scripts/run_experiment.py   # full pipeline on the MockLLM
```

The default backend is a **deterministic mock LLM**, so the whole pipeline runs
end-to-end today. You develop and review logic against the mock, then swap in a
real model with `--backend`.

## Swapping in a real model

`src/cpm_appraisal/llm/__init__.py` defines three backends behind one interface:

| backend | use |
|---|---|
| `mock` | development, CI, reviewing logic (default) |
| `openai_compat` | hosted free tiers (Groq, Together) or a local vLLM server |
| `local_transformers` | in-process HuggingFace model — for **DelftBlue** |

**DelftBlue note:** compute nodes have no internet, so API backends won't work
inside a batch job. Use `local_transformers` (or a vLLM server in the same job).
Llama 3.3 70B is large; confirm the GPU partition's VRAM before committing.
Recommended path: prove the science on a hosted free tier first, scale on
DelftBlue last.

## What is real vs. placeholder (be honest in the report)

| component | status |
|---|---|
| pipeline structure, two-timeline scheduler, entropy/convergence | **real, tested** |
| SEC prompts, narrative/timeline prompts | **real** (from the shared doc) |
| 13 emotion prototype vectors | **PLACEHOLDER** — transcribe crowd-enVENT Fig. 8 → `data/prototypes/` |
| LLM backend | **mock** until a real one is wired |
| processing-level policy | rule-based stand-in (defensible, swappable) |

## Open issues flagged in code (search `TODO` / `NOTE` / `flag for the team`)

1. **Slides vs. report mismatch:** slides still say valence/arousal; the report
   and supervisor feedback say **discrete emotions + entropy**. Code follows the
   report. Update the slides.
2. **Dimension count:** report prose says 25, Figure 3 lists 23 (5+13+3+2). Code
   uses the 23 named ones. Resolve before final submission.
3. **Prototypes are placeholders** — replace before any real run.
4. **LangGraph:** proposal commits to it; the scheduler is a drop-in for a
   LangGraph state graph (see `pipeline/__init__.py`). Port is a refactor, not a
   rewrite.

## Team
Rheea, Florin, Irene, Yolina, Levi.
