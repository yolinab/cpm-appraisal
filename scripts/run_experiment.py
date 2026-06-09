"""Run the full experiment across all three complexity levels.

    python scripts/run_experiment.py            # uses MockLLM, runs anywhere
    python scripts/run_experiment.py --backend openai_compat ...

Writes a JSON results file to data/outputs/ and prints a summary.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from cpm_appraisal.emotions.prototypes import load_prototypes
from cpm_appraisal.llm import build_llm
from cpm_appraisal.pipeline import appraise_scenario
from cpm_appraisal.scheduler import SchedulerConfig

log = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="mock")
    p.add_argument("--model-id", default=None, help="HuggingFace model id or local path (local_transformers backend)")
    p.add_argument("--prototypes", default=None, help="path to prototype CSV/JSON")
    p.add_argument("--t-max", type=int, default=20)
    p.add_argument("--seed", type=int, default=None, help="seed for reproducible event sampling")
    p.add_argument("--out", default="data/outputs/results.json")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("backend=%s model_id=%s t_max=%d seed=%s", args.backend, args.model_id, args.t_max, args.seed)

    llm_kwargs = {}
    if args.model_id:
        llm_kwargs["model_id"] = args.model_id
    llm = build_llm(args.backend, **llm_kwargs)
    log.info("LLM ready: %s", llm.__class__.__name__)

    prototypes = load_prototypes(args.prototypes)
    config = SchedulerConfig(t_max=args.t_max)

    results = []
    for level in (1, 2, 3):
        log.info("--- complexity level %d ---", level)
        scenario, conv = appraise_scenario(llm, level, prototypes, config, seed=args.seed)
        top2 = conv.final_distribution.top(2)

        log.info(
            "level=%d event=%s steps=%d converged=%s tau=%s top2=%s",
            level,
            scenario.events[0].id,
            len(conv.entropy_trace),
            conv.converged,
            conv.converged_at_tau,
            top2,
        )
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
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
