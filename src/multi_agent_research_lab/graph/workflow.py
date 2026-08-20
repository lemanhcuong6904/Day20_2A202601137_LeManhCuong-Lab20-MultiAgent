"""LangGraph workflow skeleton."""

from multi_agent_research_lab.agents import (
    AnalystAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import LabError
from multi_agent_research_lab.core.state import ResearchState


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def __init__(
        self,
        supervisor: SupervisorAgent | None = None,
        researcher: ResearcherAgent | None = None,
        analyst: AnalystAgent | None = None,
        writer: WriterAgent | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.supervisor = supervisor or SupervisorAgent(settings=self.settings)
        self.agents: dict[str, BaseAgent] = {
            "researcher": researcher or ResearcherAgent(),
            "analyst": analyst or AnalystAgent(),
            "writer": writer or WriterAgent(),
        }

    def build(self) -> dict[str, object]:
        """Return the executable graph description.

        The project keeps `langgraph` optional for conda/offline use. This method exposes
        the same node/route shape that a LangGraph implementation would compile.
        """

        return {
            "supervisor": self.supervisor,
            "nodes": self.agents,
            "terminal": "done",
        }

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the workflow and return final state."""

        self.build()
        while state.iteration < self.settings.max_iterations:
            state = self.supervisor.run(state)
            route = state.route_history[-1]
            if route == "done":
                return state

            agent = self.agents.get(route)
            if agent is None:
                state.errors.append(f"Unknown route: {route}")
                state.add_trace_event("error", {"route": route, "message": "unknown route"})
                return state

            try:
                state = agent.run(state)
            except LabError as exc:
                state.errors.append(str(exc))
                state.add_trace_event(
                    "error",
                    {"agent": route, "error_type": type(exc).__name__, "message": str(exc)},
                )
                return state

        state.errors.append("Workflow stopped after reaching max_iterations")
        state.add_trace_event(
            "error",
            {
                "message": "max_iterations reached",
                "max_iterations": self.settings.max_iterations,
            },
        )
        if state.route_history[-1:] != ["done"]:
            state.record_route("done")
        return state
