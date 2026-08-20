"""Optional critic agent skeleton for bonus work."""

import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append lightweight citation findings."""

        if not state.final_answer:
            finding = "Critic skipped: final_answer is missing."
        else:
            cited_ids = {int(match) for match in re.findall(r"\[(\d+)\]", state.final_answer)}
            valid_ids = set(range(1, len(state.sources) + 1))
            invalid_ids = sorted(cited_ids - valid_ids)
            if invalid_ids:
                finding = f"Invalid citation IDs found: {invalid_ids}."
                state.errors.append(finding)
            elif not cited_ids and state.sources:
                finding = "Final answer has sources available but no citation IDs."
                state.errors.append(finding)
            else:
                finding = "Citation check passed."

        state.agent_results.append(
            AgentResult(agent=AgentName.CRITIC, content=finding, metadata={"mode": "heuristic"})
        )
        state.add_trace_event("agent.end", {"agent": self.name, "finding": finding})
        return state
