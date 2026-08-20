"""Search client abstraction for ResearcherAgent."""

import json
from urllib import request

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.schemas import SourceDocument


class SearchClient:
    """Provider-agnostic search client with Tavily and local mock modes."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query."""

        max_results = max(1, min(max_results, 20))
        if self.settings.tavily_api_key:
            try:
                return self._search_tavily(query, max_results)
            except Exception as exc:  # pragma: no cover - depends on external provider
                return [
                    source.model_copy(
                        update={
                            "metadata": {
                                **source.metadata,
                                "fallback_reason": f"provider failed: {exc}",
                            }
                        }
                    )
                    for source in _local_sources(query, max_results)
                ]

        return _local_sources(query, max_results)

    def _search_tavily(self, query: str, max_results: int) -> list[SourceDocument]:
        payload = json.dumps(
            {
                "api_key": self.settings.tavily_api_key,
                "query": query,
                "max_results": max_results,
                "include_answer": False,
            }
        ).encode("utf-8")
        req = request.Request(
            "https://api.tavily.com/search",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.settings.timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))

        results = data.get("results", [])
        sources: list[SourceDocument] = []
        for index, item in enumerate(results[:max_results], start=1):
            sources.append(
                SourceDocument(
                    title=str(item.get("title") or f"Search result {index}"),
                    url=item.get("url"),
                    snippet=str(item.get("content") or item.get("snippet") or ""),
                    metadata={"rank": index, "provider": "tavily"},
                )
            )
        return sources


def _local_sources(query: str, max_results: int) -> list[SourceDocument]:
    templates = [
        (
            "Concept overview",
            "Defines the topic, its common use cases, and the problem it is intended to solve.",
        ),
        (
            "Implementation patterns",
            "Describes architectures, handoffs, tool use, evaluation loops, and "
            "production constraints.",
        ),
        (
            "Risks and evaluation",
            "Highlights reliability, source quality, latency, cost, governance, and "
            "measurement concerns.",
        ),
        (
            "Operational guidance",
            "Compares simple baselines with more complex workflows and recommends "
            "benchmark-driven rollout.",
        ),
        (
            "Case-study signals",
            "Summarizes practical lessons from using structured workflows on research-style tasks.",
        ),
    ]
    sources: list[SourceDocument] = []
    for index, (title, snippet) in enumerate(templates[:max_results], start=1):
        sources.append(
            SourceDocument(
                title=f"{title}: {query}",
                url=f"local://source/{index}",
                snippet=snippet,
                metadata={"rank": index, "provider": "local-mock"},
            )
        )
    return sources
