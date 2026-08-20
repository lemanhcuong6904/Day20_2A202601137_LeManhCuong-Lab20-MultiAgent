from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark


def test_benchmark_derives_metrics_from_state() -> None:
    def runner(query: str) -> ResearchState:
        state = ResearchState(request=ResearchQuery(query=query))
        state.sources = [SourceDocument(title="A", url="https://example.com", snippet="Snippet")]
        state.research_notes = "notes [1]"
        state.analysis_notes = "analysis [1]"
        state.final_answer = "answer [1]"
        return state

    _state, metrics = run_benchmark("multi-agent", "Explain multi-agent systems", runner)

    assert metrics.run_name == "multi-agent"
    assert metrics.latency_seconds >= 0
    assert metrics.quality_score is not None
    assert metrics.quality_score > 0
    assert metrics.citation_coverage == 1.0
    assert metrics.failure_rate == 0.0
