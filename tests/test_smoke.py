"""End-to-end smoke tests against the MockLLM. No keys, no network, no GPU."""
from cpm_appraisal.appraisal.scheduler import SchedulerConfig
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
        assert 6 <= len(scenario.events) <= 12
        assert len(conv.entropy_trace) >= 1
        # probabilities form a valid distribution
        total = sum(conv.final_distribution.probabilities.values())
        assert abs(total - 1.0) < 1e-6


def test_entropy_in_unit_range():
    from cpm_appraisal.emotions.classify import appraisal_to_distribution, entropy
    protos = load_prototypes()
    h = entropy(appraisal_to_distribution({}, protos))
    assert 0.99 <= h <= 1.0001  # empty vector -> uniform -> max entropy
