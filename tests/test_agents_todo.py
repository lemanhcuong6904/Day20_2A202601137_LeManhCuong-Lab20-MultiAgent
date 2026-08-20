import pytest

from multi_agent_research_lab.agents import (
    AnalystAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.core.config import Settings
from multi_agent_research_lab.core.errors import ValidationError
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.llm_client import LLMClient, LLMResponse
from multi_agent_research_lab.services.search_client import SearchClient


class FakeLLMClient(LLMClient):
    def __init__(self) -> None:
        pass

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        if "research agent" in system_prompt.lower():
            return LLMResponse(
                content="Research notes [1]",
                input_tokens=3,
                output_tokens=3,
                cost_usd=0.0,
            )
        if "analyst" in system_prompt.lower():
            return LLMResponse(
                content="Analysis notes [1]",
                input_tokens=3,
                output_tokens=3,
                cost_usd=0.0,
            )
        return LLMResponse(
            content="Final answer with citation [1]",
            input_tokens=3,
            output_tokens=5,
            cost_usd=0.0,
        )


class FakeSearchClient(SearchClient):
    def __init__(self) -> None:
        pass

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        return [
            SourceDocument(
                title="Source A",
                url="https://example.com/a",
                snippet=f"Snippet about {query}",
            )
        ]


def _state() -> ResearchState:
    return ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))


def test_supervisor_routes_by_missing_state_fields() -> None:
    settings = Settings(MAX_ITERATIONS=6)
    supervisor = SupervisorAgent(settings=settings)
    state = _state()

    supervisor.run(state)
    assert state.route_history[-1] == "researcher"

    state.sources = [SourceDocument(title="A", url=None, snippet="s")]
    state.research_notes = "notes"
    supervisor.run(state)
    assert state.route_history[-1] == "analyst"

    state.analysis_notes = "analysis"
    supervisor.run(state)
    assert state.route_history[-1] == "writer"

    state.final_answer = "answer"
    supervisor.run(state)
    assert state.route_history[-1] == "done"


def test_researcher_populates_sources_notes_and_result() -> None:
    state = _state()
    agent = ResearcherAgent(search_client=FakeSearchClient(), llm_client=FakeLLMClient())

    result = agent.run(state)

    assert result.sources
    assert result.research_notes == "Research notes [1]"
    assert result.agent_results[-1].metadata["source_count"] == 1


def test_analyst_and_writer_validate_preconditions() -> None:
    state = _state()

    with pytest.raises(ValidationError):
        AnalystAgent(llm_client=FakeLLMClient()).run(state)

    state.research_notes = "notes"
    with pytest.raises(ValidationError):
        WriterAgent(llm_client=FakeLLMClient()).run(state)


def test_workflow_runs_end_to_end_with_injected_agents() -> None:
    settings = Settings(MAX_ITERATIONS=6)
    llm = FakeLLMClient()
    workflow = MultiAgentWorkflow(
        supervisor=SupervisorAgent(settings=settings),
        researcher=ResearcherAgent(search_client=FakeSearchClient(), llm_client=llm),
        analyst=AnalystAgent(llm_client=llm),
        writer=WriterAgent(llm_client=llm),
        settings=settings,
    )

    result = workflow.run(_state())

    assert result.route_history == ["researcher", "analyst", "writer", "done"]
    assert result.final_answer == "Final answer with citation [1]"
    assert not result.errors
