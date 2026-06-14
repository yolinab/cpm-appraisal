"""Run the full experiment across all three complexity levels.

    python scripts/run_experiment.py            # uses MockLLM, runs anywhere
    python scripts/run_experiment.py --backend openai_compat ...

LLM calls are made ONCE per complexity level. If --temperatures is given,
the softmin temperature sweep is applied to the saved trajectories without
repeating any LLM calls. One JSON file is written per temperature.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from cpm_appraisal.emotions.classify import appraisal_to_distribution
from cpm_appraisal.emotions.prototypes import load_prototypes, load_weights
from cpm_appraisal.llm import build_llm
from cpm_appraisal.pipeline import appraise_scenario
from cpm_appraisal.reporting import build_run_record
from cpm_appraisal.scheduler import SchedulerConfig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="mock")
    p.add_argument("--model-id", default=None,
                   help="HuggingFace model id or local path (local_transformers / openai_compat)")
    p.add_argument("--base-url", default=None,
                   help="API base URL for openai_compat backend "
                        "(e.g. https://api.groq.com/openai/v1)")
    p.add_argument("--prototypes", default="data/prototypes/prototypes.json",
                   help="path to prototype CSV/JSON")
    p.add_argument("--weights", default="data/prototypes/weights.json",
                   help="path to weights CSV/JSON")
    # dt = 0.1 s, so the 10 s report budget is 100 steps (was 20 at dt = 0.5 s).
    p.add_argument("--t-max", type=int, default=100,
                   help="appraisal-step budget (100 steps = 10 s at dt = 0.1 s)")
    p.add_argument("--temperatures", type=float, nargs="+",
                   default=[0.1, 0.2, 0.3, 0.4, 0.5],
                   help="softmin temperature(s) to sweep (LLM runs once, analysis repeats per temperature)")
    p.add_argument("--seed", type=int, default=None,
                   help="seed for reproducible event sampling")
    p.add_argument("--full-distribution", action="store_true",
                   help="store the full 13-emotion distribution at every step")
    p.add_argument("--out-dir", default=None,
                   help="output directory (default: data/outputs/<model>/<timestamp>/)")
    args = p.parse_args()

    llm_kwargs = {}
    if args.model_id:
        llm_kwargs["model_id"] = args.model_id
    if args.base_url:
        llm_kwargs["base_url"] = args.base_url
    llm = build_llm(args.backend, **llm_kwargs)

    model_slug = (args.model_id or args.backend).replace("/", "-")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (
        Path("data/outputs") / model_slug / timestamp
    )

    prototypes = load_prototypes(args.prototypes)
    weights = load_weights(args.weights)
    config = SchedulerConfig(t_max=args.t_max)

    # --- Phase 1: run LLM once per complexity level ---
    print("=== Phase 1: generating scenarios and running appraisal (LLM calls) ===")
    runs = []
    for level in (1, 2, 3):
        print(f"\n  Complexity level {level}...")
        scenario, trajectory, _ = appraise_scenario(
            llm, level, prototypes,
            weights=weights, scheduler_config=config,
            seed=args.seed, temperature=0.2,
        ) # default temperature is 0.2
        runs.append((scenario, trajectory))
        ev = scenario.events[0]
        print(f"  sampled event  : {ev.id} — {ev.description}")
        print(f"  trajectory steps: {len(trajectory)}")

    # --- Phase 2: sweep temperatures over saved trajectories (no LLM calls) ---
    print(f"\n=== Phase 2: sweeping {len(args.temperatures)} temperatures ===")
    for temp in args.temperatures:
        print(f"\n--- temperature={temp} ---")

        def to_dist(vector, t=temp):
            return appraisal_to_distribution(
                vector, prototypes, weights=weights, temperature=t
            ).probabilities

        results = []
        for scenario, trajectory in runs:
            record = build_run_record(
                scenario, trajectory, to_dist,
                include_full_distribution=args.full_distribution,
            )
            c = record["convergence"]
            t_rec = record["timing"]
            ev = record["scenario"]["event"]
            print(f"  L{scenario.complexity_level} ({ev['id']}): "
                  f"tau={t_rec['total_tau']}, "
                  f"converged={c['converged']}, "
                  f"entropy={c['final_entropy']:.3f}")
            results.append(record)

        out = out_dir / f"temp_{temp}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"  Wrote {out}")

    print(f"\nAll results in {out_dir}/")


if __name__ == "__main__":
    main()