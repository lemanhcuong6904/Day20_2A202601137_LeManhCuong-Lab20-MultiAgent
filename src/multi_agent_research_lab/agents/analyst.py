"""Analyst agent skeleton."""

from time import perf_counter

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError, ValidationError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""

        if not state.research_notes:
            raise ValidationError("Analyst requires research_notes")

        started = perf_counter()
        state.add_trace_event("agent.start", {"agent": self.name})
        response = self.llm_client.complete(
            system_prompt=(
                "You are an analyst. Extract key claims, evidence strength, weak points, "
                "and recommendation from the research notes."
            ),
            user_prompt=(
                f"Query: {state.request.query}\n"
                f"Audience: {state.request.audience}\n\n"
                f"Research notes:\n{state.research_notes}\n\n"
                "Return structured analysis and keep citation IDs."
            ),
        )
        analysis = response.content.strip()
        if not analysis:
            raise AgentExecutionError("Analyst produced empty analysis")

        duration = perf_counter() - started
        state.analysis_notes = analysis
        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=analysis,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                    "duration_seconds": duration,
                },
            )
        )
        state.add_trace_event(
            "agent.end",
            {
                "agent": self.name,
                "duration_seconds": duration,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            },
        )
        return state
