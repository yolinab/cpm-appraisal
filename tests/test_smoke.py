"""End-to-end smoke tests against the MockLLM. No keys, no network, no GPU."""
from cpm_appraisal.scheduler import SchedulerConfig
from cpm_appraisal.emotions.dimensions import ALL_DIMENSIONS
from cpm_appraisal.emotions.prototypes import EMOTION_LABELS, load_prototypes
from cpm_appraisal.llm import build_llm
from cpm_appraisal.pipeline import run_one


def test_dimension_and_emotion_counts():
    assert len(ALL_DIMENSIONS) == 23
    assert len(EMOTION_LABELS) == 13


def test_pipeline_runs_end_to_end():
    llm = build_llm("mock")
    protos = load_prototypes()
    for level in (1, 2, 3):
        scenario, conv = run_one(llm, level, protos, SchedulerConfig(t_max=20))
        assert len(scenario.events) == 1  # single sampled event
        assert len(conv.entropy_trace) >= 1
        # probabilities form a valid distribution
        total = sum(conv.final_distribution.probabilities.values())
        assert abs(total - 1.0) < 1e-6


def test_entropy_in_unit_range():
    from cpm_appraisal.emotions.classify import appraisal_to_distribution, entropy
    protos = load_prototypes()
    h = entropy(appraisal_to_distribution({}, protos))
    assert 0.99 <= h <= 1.0001  # empty vector -> uniform -> max entropy


def test_langgraph_matches_plain_loop():
    from cpm_appraisal.generation import PERSONA_SYSTEM, generate_scenario
    from cpm_appraisal.scheduler import run_appraisal_timeline
    from cpm_appraisal.pipeline import run_appraisal_timeline_graph
    from cpm_appraisal.secs import run_sec

    llm = build_llm("mock")
    scenario = generate_scenario(llm, 2)

    def run_check(sec, desc, prior):
        return run_sec(llm, sec, desc, prior, PERSONA_SYSTEM)

    config = SchedulerConfig(t_max=20)
    plain = run_appraisal_timeline(scenario.events, run_check, config=config)
    graph = run_appraisal_timeline_graph(scenario.events, run_check, config=config)

    assert len(plain) == len(graph)
    assert [s.tau for s in plain] == [s.tau for s in graph]
    assert [s.vector for s in plain] == [s.vector for s in graph]


def test_langgraph_respects_budget():
    from cpm_appraisal.generation import PERSONA_SYSTEM, generate_scenario
    from cpm_appraisal.pipeline import run_appraisal_timeline_graph
    from cpm_appraisal.secs import run_sec

    llm = build_llm("mock")
    scenario = generate_scenario(llm, 1)

    def run_check(sec, desc, prior):
        return run_sec(llm, sec, desc, prior, PERSONA_SYSTEM)

    tight_config = SchedulerConfig(t_max=4, schematic_cost=1, conceptual_cost=3)
    trajectory = run_appraisal_timeline_graph(scenario.events, run_check, config=tight_config)

    assert all(s.tau <= tight_config.t_max for s in trajectory)
