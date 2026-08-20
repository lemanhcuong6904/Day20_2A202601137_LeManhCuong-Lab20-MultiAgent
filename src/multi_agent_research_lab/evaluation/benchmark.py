"""Benchmark helpers for single-agent vs multi-agent."""

from collections.abc import Callable
import re
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency and derive lightweight quality metrics."""

    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started
    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=_total_cost(state),
        quality_score=_quality_score(state),
        citation_coverage=_citation_coverage(state),
        failure_rate=1.0 if state.errors or not state.final_answer else 0.0,
        notes=_notes(state),
    )
    return state, metrics


def _total_cost(state: ResearchState) -> float:
    total = 0.0
    for result in state.agent_results:
        cost = result.metadata.get("cost_usd")
        if isinstance(cost, int | float):
            total += float(cost)
    return round(total, 6)


def _citation_coverage(state: ResearchState) -> float:
    if not state.final_answer:
        return 0.0
    if not state.sources:
        return 0.0
    cited_ids = {int(match) for match in re.findall(r"\[(\d+)\]", state.final_answer)}
    valid_ids = set(range(1, len(state.sources) + 1))
    return round(len(cited_ids & valid_ids) / len(valid_ids), 3)


def _quality_score(state: ResearchState) -> float:
    if not state.final_answer:
        return 0.0
    score = 4.0
    if state.sources:
        score += 1.5
    if state.research_notes:
        score += 1.0
    if state.analysis_notes:
        score += 1.0
    if _citation_coverage(state) > 0:
        score += 1.5
    if state.errors:
        score -= 2.0
    return max(0.0, min(10.0, round(score, 1)))


def _notes(state: ResearchState) -> str:
    if state.errors:
        return "; ".join(state.errors)
    return f"routes={','.join(state.route_history) or 'baseline'}"
