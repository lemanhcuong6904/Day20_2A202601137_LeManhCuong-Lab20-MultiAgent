"""Benchmark report rendering."""

from statistics import mean

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to markdown."""

    lines = [
        "# Benchmark Report",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        citation = "" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure = "" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} "
            f"| {citation} | {failure} | {item.notes} |"
        )
    if metrics:
        lines.extend(
            [
                "",
                "## Summary",
                "",
                f"- Average latency: {_avg_latency(metrics):.2f}s.",
                f"- Average quality score: {_avg_quality(metrics):.1f}/10.",
                f"- Average citation coverage: {_avg_citation(metrics):.0%}.",
                f"- Failure rate across runs: {_avg_failure(metrics):.0%}.",
                "",
                "## Interpretation",
                "",
                (
                    "Use the multi-agent path when traceability, source grounding, and "
                    "intermediate review matter more than raw latency. Keep the baseline "
                    "for short questions or when cost and response time dominate."
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def _avg_latency(metrics: list[BenchmarkMetrics]) -> float:
    return mean(item.latency_seconds for item in metrics)


def _avg_quality(metrics: list[BenchmarkMetrics]) -> float:
    values = [item.quality_score for item in metrics if item.quality_score is not None]
    return mean(values) if values else 0.0


def _avg_citation(metrics: list[BenchmarkMetrics]) -> float:
    values = [item.citation_coverage for item in metrics if item.citation_coverage is not None]
    return mean(values) if values else 0.0


def _avg_failure(metrics: list[BenchmarkMetrics]) -> float:
    values = [item.failure_rate for item in metrics if item.failure_rate is not None]
    return mean(values) if values else 0.0
