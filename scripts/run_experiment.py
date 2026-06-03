"""Run the full experiment across all three complexity levels.

    python scripts/run_experiment.py            # uses MockLLM, runs anywhere
    python scripts/run_experiment.py --backend openai_compat ...

Writes a JSON results file to data/outputs/ and prints a summary.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from cpm_appraisal.appraisal.scheduler import SchedulerConfig
from cpm_appraisal.emotions.prototypes import load_prototypes
from cpm_appraisal.llm import build_llm
from cpm_appraisal.pipeline import run_one


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="mock")
    p.add_argument("--model-id", default=None, help="HuggingFace model id or local path (local_transformers backend)")
    p.add_argument("--prototypes", default=None, help="path to prototype CSV/JSON")
    p.add_argument("--t-max", type=int, default=20)
    p.add_argument("--out", default="data/outputs/results.json")
    args = p.parse_args()

    llm_kwargs = {}
    if args.model_id:
        llm_kwargs["model_id"] = args.model_id
    llm = build_llm(args.backend, **llm_kwargs)
    prototypes = load_prototypes(args.prototypes)
    config = SchedulerConfig(t_max=args.t_max)

    results = []
    for level in (1, 2, 3):
        scenario, conv = run_one(llm, level, prototypes, config)
        top2 = conv.final_distribution.top(2)
        print(f"\n=== Complexity level {level} ===")
        print(f"  events generated : {len(scenario.events)}")
        print(f"  trajectory length: {len(conv.entropy_trace)} steps")
        print(f"  converged        : {conv.converged} "
              f"(at tau={conv.converged_at_tau})")
        print(f"  final top-2      : {top2}")
        results.append({
            "scenario": asdict(scenario),
            "convergence": {
                "converged": conv.converged,
                "converged_at_tau": conv.converged_at_tau,
                "entropy_trace": conv.entropy_trace,
                "final_top2": top2,
            },
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
