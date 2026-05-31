# CPM Appraisal Convergence Pipeline

DSAIT4230 — Research in Social Signal Processing & Affective Computing, TU Delft.

## About the project

We feed an LLM (playing a fixed persona, **Alex**) a social situation. Instead of
asking "what emotion is this?" in one shot, the model appraises the situation
step by step through Scherer's four **Stimulus Evaluation Checks** (Relevance →
Implication → Coping → Normative). At each step the appraisal ratings are turned
into a **probability distribution over 13 discrete emotions**, and we track the
**entropy** of that distribution. The core question: does the emotion *lock in*
(entropy stabilises) before all information is processed, and do **more complex
situations take more steps to lock in**?

## Timelines

- **Event timeline** (`e1, e2, …`): the situation is revealed gradually as a
  sequence of discrete events.
- **Appraisal timeline** (`τ = 1, 2, …`): within processing, each check costs
  time — *schematic* checks cost 1 step, *conceptual* checks cost 3 — against a
  budget `t_max`. Convergence is the point on this timeline where entropy stops
  changing.

## Timeline Assumptions

We assume that appraisal processing may overlap across events. A new event can begin being appraised before the previous event’s appraisal has fully finished. All appraisal activity is handled on one shared appraisal clock with overlapping processing windows.

Therefore, the implementation uses only two relevant temporal structures:

1. the event timeline, which determines when events become available to Alex;
2. the appraisal clock τ, which determines how perceived events are processed internally.

This keeps the model aligned with the conceptual framing while remaining simple enough to implement within the project timeline.

## Reference diagrams

**1. Appraisal module (one event).** One event → four checks in
order → each check resolved schematically *or* conceptually → after each check, read off the emotion distribution and
its entropy H → H falls and flattens → convergence.

![Appraisal module for a single event](docs/diagrams/01_appraisal_module_single_event.png)

#### One-pass-per-event simplification. 
One pass per event is a deliberate simplification of a genuinely recursive theory, chosen for tractability.
Within a single event we run the four checks once, in order, and read off the emotion after each — giving the convergence trajectory we measure. Scherer's full theory is recursive: appraisal "is not a one-shot affair," and the organism maintains a "recursive appraisal process until the monitoring subsystem signals termination". We approximate this two ways: (1) across events, the running appraisal accumulates and is continuously re-read, a coarse form of reappraisal; (2) within-event recursive looping is deliberately excluded (cross_event_coupling = False) and named as a limitation.

#### Processing levels in the appraisal module. 

Following Leventhal & Scherer (1987), each stimulus evaluation check can be resolved at different levels of processing. Scherer's full scheme distinguishes four levels (sensory-motor, schematic, associative, conceptual); for tractability we collapse these to two (schematic and conceptual).

- At the schematic level, a check is resolved automatically and outside of consciousness — fast and effortless. 
- At the conceptual level, the check is resolved through conscious, effortful reasoning — slow and costly. 

Critically, the level is not a fixed property of a check — the same check can be schematic for one event and conceptual for another, because it depends on what the specific event demands. 

In our worked example, the event "walks into the former home" resolves its Relevance check **schematically** (instant recognition that something is happening) but its Implication check conceptually (it invokes the history of having moved out, which requires reasoning); whereas the event "implied failure as a father" resolves all four checks **conceptually**, because it engages identity, blame, control, and moral judgement. This is why the boxes in the diagram are tagged the way they are: the tag reflects how that particular check, for that particular event, has to be processed — not a universal label on the check.

#### How we plan to assign levels. 
Since the level depends on the (event, check) pair, we need a rule for deciding it. For this study we use a transparent, rule-based assignment: a check is marked conceptual when the event text contains cues that engage identity, social history, moral evaluation, or contested control (e.g. accusation, blame, divorce, a child being present), and schematic otherwise.

#### What "schematic" and "conceptual" actually do in the code
A processing level resolves to exactly: **how many steps the check costs on the appraisal clock**. A schematic check occupies 1 step; a conceptual check occupies 3 steps (SchedulerConfig.schematic_cost / conceptual_cost). The level does not change what rating the LLM produces — only when, on the shared τ clock, that rating finishes landing.
#### Motivation for design choice. 
Scherer's theory holds that conceptual processing is "slow and conscious" while schematic processing is "fast and automatic". Our model encodes that single, well-supported claim — conceptual costs more time than schematic — and nothing more. The specific 3:1 ratio is a chosen model parameter, not a measured human appraisal duration; it exists only to make the speed difference pronounced enough to observe.
The justification for reducing "processing level" to "time cost" is because we are studying **when** an emotion converges, so the only property of a check that needs to enter the model is how much it delays convergence. Anything richer (how the level changes the content of the appraisal) is outside of the scope of this study.

**2. The whole situation (events on one shared clock).** Each event's module
collapses to a single bar. Events are staggered on the shared τ clock and their
processing windows overlap. The situation-level emotion is read from all checks
completed so far; H(τ) is tracked until it settles or `t_max` is hit.

![Events on one shared appraisal clock](docs/diagrams/02_shared_clock_cross_event.png)

> **Open modeling decision** (the dashed "background loop" in diagram 2):
> should the shared state loop back and re-bias events still being processed?
> This is **OFF by default** (`SchedulerConfig.cross_event_coupling = False`).
> Turning it on means cyclical re-appraisal — extra complexity and a convergence
> guarantee we can't validate in two weeks. It's already listed as a *limitation*
> in the report, so excluding it keeps the project self-consistent. Leave it for
> future work.

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

## Running codebase

```bash
pip install -e .
pytest -q                          # tests
python scripts/run_experiment.py   # full pipeline on the MockLLM
```

The current setup uses a **deterministic mock LLM**, so the whole pipeline runs
end-to-end. We plan to develop and review logic against the mock, then swap in a
real model with `--backend`.

## Swapping in a real model

`src/cpm_appraisal/llm/__init__.py` defines backends behind one interface:

| backend | use |
|---|---|
| `mock` | development, CI, reviewing logic (default) |
| `local_transformers` | in-process HuggingFace model |

## Comprehensive List of Assumptions

### About emotions and appraisal

- An emotion can be represented as a probability distribution over 13 fixed discrete emotion categories (anger, fear, joy, etc.) — we're not using valence/arousal.
- An appraisal can be captured as a vector of 1–5 ratings across ~23 dimensions, and the distance from that vector to 13 "prototype" emotion profiles tells us which emotion it is.
- "Convergence" means the entropy (uncertainty) of that distribution stops changing — we treat a flattening entropy curve as the emotion having settled.
- The four checks happen in a fixed order (Relevance → Implication → Coping → Normative), and each builds on the previous ones.

### About the processing levels (the schematic/conceptual thing)

- Scherer's four processing levels are collapsed to two (schematic, conceptual). This is our simplification, not Scherer's scheme. (Cite van Reekum & Scherer 1997 for the original levels.)
- The level is assigned per (event, check) — the same check can be fast for one event and slow for another. It is not a fixed property of the check.
- The level affects only timing: schematic costs 1 clock step, conceptual costs 3. The 3:1 ratio is an arbitrary chosen parameter, not a measured human value. It changes when a rating lands, not what the rating is.
- How we decide which level applies is currently a simple keyword rule on the event text — itself a stand-in, not a validated method.

### About time

- "Time" is discretised into fixed steps; an LLM's sequential processing is treated as a proxy for human cognitive time, which it isn't really (this is already in your limitations).
- There's a hard time budget t_max; if processing isn't done by then, we just take whatever the emotion is at that point.
- Events can overlap on one shared clock (a later event starts before an earlier one finishes), but events do not loop back to re-bias each other while in-flight — cross-event coupling is off. (Recursive re-appraisal excluded, listed as a limitation.)
- The three "worlds" (objective / perceived / appraisal) are collapsed to two clocks; no separate parallel threads.

### About the persona and model

- One fixed persona ("Alex," high agency/conscientiousness) stands in for a human appraiser, chosen to counteract known LLM bias — a single persona cannot represent human variance (already a stated limitation).
- The LLM's 1–5 ratings are taken as a meaningful appraisal signal at all (assumes the LLM can do this reliably — the report leans on Ruder et al. 2025 and Broekens et al. 2023 to justify this).

### About the data (still placeholders in code)

- The 13 emotion prototype vectors are currently made-up placeholders — the real ones (from crowd-enVENT Figure 8) aren't transcribed yet, so no current output is scientifically meaningful.
- Three hand-designed scenarios at three complexity levels stand in for the space of social situations; "complexity" is defined by us as "how many checks return contested answers," explicitly not claimed as objective truth.

---

## Code architecture — how the modules fit together

```
┌─────────────────────────────────────────────────────────────────────┐
│  scripts/run_experiment.py  ← entry point, CLI args, writes JSON   │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ calls
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  pipeline/__init__.py  →  run_one()                                 │
│  Wires all stages together; no logic of its own.                    │
└──────┬──────────────────────────────────────────┬───────────────────┘
       │ stage 1–2                                │ stage 3–5
       ▼                                          ▼
┌──────────────────────┐              ┌───────────────────────────────┐
│  generation/         │              │  appraisal/scheduler.py       │
│  ─────────────────   │              │  ───────────────────────────  │
│  narrative_prompt()  │              │  run_appraisal_timeline()     │
│  timeline_prompt()   │              │                               │
│  generate_scenario() │              │  For each (event, SEC):       │
│                      │              │   • levels.py decides         │
│  Calls LLM twice:    │              │     schematic (1 step) or     │
│  1. write narrative  │              │     conceptual (3 steps)      │
│  2. decompose into   │              │   • secs.py calls the LLM     │
│     ordered events   │              │     and parses ratings        │
│                      │              │   • tau advances by cost      │
│  Produces:           │              │   • snapshot saved after      │
│   Scenario           │              │     each completed check      │
│   └─ list[Event]     │              │                               │
└──────────────────────┘              │  Stops when tau ≥ t_max       │
                                      │                               │
                                      │  Produces:                    │
                                      │   list[AppraisalStep]         │
                                      │   (one per completed check)   │
                                      └──────────────┬────────────────┘
                                                     │
                                                     ▼
                                      ┌──────────────────────────────┐
                                      │  emotions/                   │
                                      │  ──────────────────────────  │
                                      │  dimensions.py               │
                                      │   23 named dimensions,       │
                                      │   grouped by SEC             │
                                      │                              │
                                      │  prototypes.py               │
                                      │   13 emotion prototypes      │
                                      │   (placeholders for now)     │
                                      │                              │
                                      │  classify.py                 │
                                      │   vector                     │
                                      │   → Manhattan distance       │
                                      │      to each prototype       │
                                      │   → softmin → probabilities  │
                                      │   → Shannon entropy          │
                                      └──────────────┬───────────────┘
                                                     │
                                                     ▼
                                      ┌──────────────────────────────┐
                                      │  convergence/                │
                                      │  ──────────────────────────  │
                                      │  analyse_trajectory()        │
                                      │                              │
                                      │  Builds entropy trace over   │
                                      │  tau; finds first point      │
                                      │  where |H(τ)−H(τ−1)| <      │
                                      │  threshold for N steps.      │
                                      │                              │
                                      │  Produces:                   │
                                      │   ConvergencePoint           │
                                      │   └─ converged_at_tau        │
                                      │   └─ entropy_trace           │
                                      │   └─ final_distribution      │
                                      └──────────────────────────────┘

Shared across all stages
─────────────────────────

  types.py          The only shared vocabulary. Every module produces
                    or consumes these dataclasses and nothing else.
                    Event, Scenario, AppraisalStep, AppraisalVector,
                    EmotionDistribution, ConvergencePoint.

  llm/__init__.py   One abstract interface (LanguageModel), three
                    backends. Swap by changing one argument at the
                    call site — nothing else changes.

                    MockLLM              → deterministic fake, runs
                                          anywhere, used for dev/CI
                    OpenAICompatLLM      → Groq / Together / vLLM
                                          (needs implementing)
                    LocalTransformersLLM → HuggingFace model on
                                          DelftBlue (needs implementing)
```

### Data types flowing between stages

```
seed + complexity level
        │
        │  (LLM × 2)
        ▼
    Scenario ──────────────────────────────────────────────────────────┐
    └─ narrative: str                                                  │
    └─ events: list[Event]                                             │
              │                                                        │
              │  (LLM × 4 checks × N events)                          │
              ▼                                                        │
    list[AppraisalStep]                                                │
    └─ tau: int                                                        │
    └─ completed_secs: list[SEC]                                       │
    └─ vector: dict[dim_name → 1..5 rating]   (grows each check)      │
              │                                                        │
              │  (distance + softmin + entropy, no LLM)               │
              ▼                                                        │
    ConvergencePoint  ◀─────────────────────────────────────────────── ┘
    └─ converged: bool
    └─ converged_at_tau: int | None
    └─ entropy_trace: list[float]
    └─ final_distribution: dict[emotion → probability]
```