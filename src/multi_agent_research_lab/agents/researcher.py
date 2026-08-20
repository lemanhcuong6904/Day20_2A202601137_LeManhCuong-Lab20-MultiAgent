"""Researcher agent skeleton."""

from time import perf_counter

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError, ValidationError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient, LLMResponse
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(
        self,
        search_client: SearchClient | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.search_client = search_client or SearchClient()
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""

        started = perf_counter()
        state.add_trace_event("agent.start", {"agent": self.name})
        sources = self.search_client.search(state.request.query, state.request.max_sources)
        sources = _dedupe_sources(sources)[: state.request.max_sources]
        if not sources:
            raise ValidationError("Researcher requires at least one source")

        prompt = _build_research_prompt(state, sources)
        response = self.llm_client.complete(
            system_prompt=(
                "You are a research agent. Produce concise notes grounded only in the "
                "provided sources and preserve citation IDs like [1]."
            ),
            user_prompt=prompt,
        )
        notes = response.content.strip()
        if not notes:
            raise AgentExecutionError("Researcher produced empty notes")

        state.sources = sources
        state.research_notes = notes
        duration = perf_counter() - started
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=notes,
                metadata=_metadata(response, duration, {"source_count": len(sources)}),
            )
        )
        state.add_trace_event(
            "agent.end",
            {
                "agent": self.name,
                "duration_seconds": duration,
                "source_count": len(sources),
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            },
        )
        return state


def _dedupe_sources(sources: list[SourceDocument]) -> list[SourceDocument]:
    seen: set[str] = set()
    unique: list[SourceDocument] = []
    for source in sources:
        key = source.url or source.title
        if key in seen:
            continue
        seen.add(key)
        unique.append(source)
    return unique


def _build_research_prompt(state: ResearchState, sources: list[SourceDocument]) -> str:
    source_lines = []
    for index, source in enumerate(sources, start=1):
        url = source.url or "no-url"
        source_lines.append(f"[{index}] {source.title} ({url})\n{source.snippet}")
    joined_sources = "\n\n".join(source_lines)
    return (
        f"Query: {state.request.query}\n"
        f"Audience: {state.request.audience}\n\n"
        "Sources:\n"
        f"{joined_sources}\n\n"
        "Return bullet research notes with citation IDs."
    )


def _metadata(
    response: LLMResponse, duration: float, extra: dict[str, int | float | str]
) -> dict[str, int | float | str | None]:
    metadata: dict[str, int | float | str | None] = {
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "cost_usd": response.cost_usd,
        "duration_seconds": duration,
    }
    metadata.update(extra)
    return metadata
