"""Run the full experiment across all three complexity levels.

    python scripts/run_experiment.py            # uses MockLLM, runs anywhere
    python scripts/run_experiment.py --backend openai_compat ...

Writes an enriched JSON results file to data/outputs/ and prints a summary.
Each record contains the scenario, a TIMING block (per-SEC latency + total
processing time, for RQ2 / "do complex events take longer"), and a CONVERGENCE
block (entropy trace plus Jensen-Shannon based convergence and validity-risk
measures, for RQ1 / RQ3). See cpm_appraisal/reporting.py for the schema.
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
    p.add_argument("--temperature", type=float, default=0.3,
                   help="softmin temperature for distance -> emotion distribution")
    p.add_argument("--seed", type=int, default=None,
                   help="seed for reproducible event sampling")
    p.add_argument("--full-distribution", action="store_true",
                   help="store the full 13-emotion distribution at every step")
    p.add_argument("--out", default=None,
                   help="output path (default: data/outputs/<model>/<timestamp>.json)")
    args = p.parse_args()

    llm_kwargs = {}
    if args.model_id:
        llm_kwargs["model_id"] = args.model_id
    if args.base_url:
        llm_kwargs["base_url"] = args.base_url
    llm = build_llm(args.backend, **llm_kwargs)

    if args.out:
        out = Path(args.out)
    else:
        model_slug = (args.model_id or args.backend).replace("/", "-")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path("data/outputs") / model_slug / f"{timestamp}.json"

    prototypes = load_prototypes(args.prototypes)
    weights = load_weights(args.weights)
    config = SchedulerConfig(t_max=args.t_max)

    # Reuse the project's own distance->distribution function so the probability
    # model in the report record is identical to convergence.py. We unwrap to the
    # raw {emotion: p} dict because reporting's entropy/JS helpers expect that.
    def to_dist(vector):
        return appraisal_to_distribution(
            vector, prototypes, weights=weights, temperature=args.temperature
        ).probabilities

    results = []
    for level in (1, 2, 3):
        scenario, trajectory, _conv = appraise_scenario(
            llm, level, prototypes,
            weights=weights, scheduler_config=config,
            seed=args.seed, temperature=args.temperature,
        )

        record = build_run_record(
            scenario, trajectory, to_dist,
            include_full_distribution=args.full_distribution,
        )

        t = record["timing"]
        c = record["convergence"]
        ev = record["scenario"]["event"]
        print(f"\n=== Complexity level {level} ({scenario.scenario_id}) ===")
        print(f"  sampled event : {ev['id']} — {ev['description']}")
        print(f"  secs          : {c['secs_total']}")
        print(f"  timing        : total_tau={t['total_tau']} "
              f"({t['total_ms']:.0f} ms), budget_exhausted={t['budget_exhausted']}")
        if c["converged_before_completion"]:
            print(f"  converged     : before completion, at tau={c['converged_at_tau']} "
                  f"(frac={c['fraction_processed_before_convergence']:.2f})")
        else:
            print("  converged     : NOT before completion")
        print(f"  validity risk : max JS-from-final={c['max_div_from_final']:.3f}, "
              f"initial={c['initial_div_from_final']:.3f}")
        print(f"  final entropy : {c['final_entropy']:.3f} (normalised)")
        print(f"  final top-2   : {record['final_distribution']}")

        results.append(record)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()