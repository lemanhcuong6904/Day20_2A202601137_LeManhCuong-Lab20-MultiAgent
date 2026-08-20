"""Pure runners shared by CLI and evaluation code."""

from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.services.llm_client import LLMClient


def run_baseline_query(
    query: str, max_sources: int = 5, audience: str = "technical learners"
) -> ResearchState:
    """Run a single-call baseline and return a normal ResearchState."""

    state = ResearchState(
        request=ResearchQuery(query=query, max_sources=max_sources, audience=audience)
    )
    response = LLMClient().complete(
        system_prompt=(
            "You are a single-agent baseline. Answer directly and mention limitations "
            "when sources are not available."
        ),
        user_prompt=f"Query: {query}\nAudience: {audience}",
    )
    state.final_answer = response.content
    state.agent_results.append(
        AgentResult(
            agent=AgentName.WRITER,
            content=response.content,
            metadata={
                "mode": "baseline",
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            },
        )
    )
    state.add_trace_event("baseline.complete", {"model_cost_usd": response.cost_usd})
    return state


def run_multi_agent_query(
    query: str, max_sources: int = 5, audience: str = "technical learners"
) -> ResearchState:
    """Run the multi-agent workflow and return the final ResearchState."""

    state = ResearchState(
        request=ResearchQuery(query=query, max_sources=max_sources, audience=audience)
    )
    return MultiAgentWorkflow().run(state)
