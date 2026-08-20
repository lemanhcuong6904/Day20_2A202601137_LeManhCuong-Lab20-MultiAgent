"""Writer agent skeleton."""

from time import perf_counter

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import AgentExecutionError, ValidationError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""

        if not state.research_notes:
            raise ValidationError("Writer requires research_notes")
        if not state.analysis_notes:
            raise ValidationError("Writer requires analysis_notes")

        started = perf_counter()
        state.add_trace_event("agent.start", {"agent": self.name})
        source_refs = "\n".join(
            f"[{index}] {source.title} - {source.url or 'no-url'}"
            for index, source in enumerate(state.sources, start=1)
        )
        response = self.llm_client.complete(
            system_prompt=(
                "You are a writer. Synthesize a concise, evidence-bound answer for the "
                "requested audience. Use only provided notes and source refs."
            ),
            user_prompt=(
                f"Query: {state.request.query}\n"
                f"Audience: {state.request.audience}\n\n"
                f"Research notes:\n{state.research_notes}\n\n"
                f"Analysis notes:\n{state.analysis_notes}\n\n"
                f"Source refs:\n{source_refs}\n\n"
                "Return the final answer with citation IDs like [1]."
            ),
        )
        answer = response.content.strip()
        if not answer:
            raise AgentExecutionError("Writer produced empty final answer")

        duration = perf_counter() - started
        state.final_answer = answer
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=answer,
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
