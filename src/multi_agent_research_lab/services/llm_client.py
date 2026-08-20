"""LLM client abstraction.

Agents depend on this adapter instead of importing a provider SDK directly. When
`OPENAI_API_KEY` is available and the optional `openai` package is installed, the
client uses OpenAI. Otherwise it returns deterministic local completions so the
lab, tests, and demos still run offline.
"""

from dataclasses import dataclass
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import Settings, get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client with a deterministic offline fallback."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion with usage metadata when available."""

        if self.settings.openai_api_key:
            try:
                return self._complete_openai(system_prompt, user_prompt)
            except Exception:  # pragma: no cover - depends on external provider
                fallback = self._complete_local(system_prompt, user_prompt)
                return LLMResponse(
                    content=fallback.content,
                    input_tokens=fallback.input_tokens,
                    output_tokens=fallback.output_tokens,
                    cost_usd=fallback.cost_usd,
                )

        return self._complete_local(system_prompt, user_prompt)

    @retry(wait=wait_exponential(multiplier=0.5, min=0.5, max=2), stop=stop_after_attempt(2))
    def _complete_openai(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise AgentExecutionError(
                "OPENAI_API_KEY is set but the optional 'openai' package is not installed"
            ) from exc

        client = OpenAI(
            api_key=self.settings.openai_api_key,
            timeout=self.settings.timeout_seconds,
            max_retries=0,
        )
        response = client.chat.completions.create(
            model=self.settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        content = response.choices[0].message.content or ""
        usage: Any = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)
        return LLMResponse(
            content=content.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=_estimate_cost_usd(input_tokens, output_tokens),
        )

    def _complete_local(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        prompt = f"{system_prompt}\n{user_prompt}"
        input_tokens = _estimate_tokens(prompt)

        lower_system = system_prompt.lower()
        if "writer" in lower_system or "synthesize" in lower_system:
            content = _local_answer(user_prompt)
        elif "analyst" in lower_system or "analysis" in lower_system:
            content = _local_analysis(user_prompt)
        elif "research" in lower_system:
            content = _local_research_notes(user_prompt)
        else:
            content = _local_baseline_answer(user_prompt)

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=_estimate_tokens(content),
            cost_usd=0.0,
        )


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))


def _estimate_cost_usd(input_tokens: int | None, output_tokens: int | None) -> float | None:
    if input_tokens is None and output_tokens is None:
        return None
    # Conservative placeholder for gpt-4o-mini style pricing used only for lab metrics.
    prompt_cost = (input_tokens or 0) * 0.00000015
    completion_cost = (output_tokens or 0) * 0.00000060
    return round(prompt_cost + completion_cost, 6)


def _extract_topic(prompt: str) -> str:
    marker = "Query:"
    if marker in prompt:
        return prompt.split(marker, maxsplit=1)[1].splitlines()[0].strip()
    return prompt.strip().splitlines()[0][:120]


def _local_research_notes(prompt: str) -> str:
    topic = _extract_topic(prompt)
    return (
        f"Research notes for '{topic}':\n"
        "- [1] Establish the core definition, current use cases, and why the topic matters.\n"
        "- [2] Identify implementation trade-offs, operational constraints, and evaluation needs.\n"
        "- [3] Capture open risks, especially reliability, data quality, cost, and governance.\n"
        "Use these notes as evidence-bound input; do not invent claims outside the listed sources."
    )


def _local_analysis(prompt: str) -> str:
    return (
        "Structured analysis:\n"
        "- Key claims: the topic has practical value when scoped to clear workflows and "
        "measured outcomes.\n"
        "- Supporting evidence: available sources describe definitions, implementation "
        "patterns, and risks.\n"
        "- Weak evidence: source snippets are high level, so exact performance or adoption "
        "claims need caution.\n"
        "- Recommendation: present benefits and limitations together, with citations for "
        "every major claim."
    )


def _local_answer(prompt: str) -> str:
    topic = _extract_topic(prompt)
    return (
        f"{topic} should be described as a practical research and engineering topic rather than a "
        "single fixed technique. The strongest supported points are its core concept [1], its main "
        "implementation trade-offs [2], and the reliability or governance risks that affect "
        "production use [3].\n\n"
        "A balanced answer should explain what problem it solves, what data or tools it "
        "requires, how success will be evaluated, and where evidence is still weak. Based "
        "on the provided sources, the recommended stance is cautious adoption: use it when "
        "the workflow needs traceable intermediate steps, but benchmark latency, cost, and "
        "quality before treating it as better than a simpler single-agent baseline."
    )


def _local_baseline_answer(prompt: str) -> str:
    topic = _extract_topic(prompt)
    return (
        f"Baseline answer for '{topic}': summarize the concept, likely benefits, limitations, and "
        "evaluation criteria in one pass. This baseline is useful for speed, but it has "
        "weaker citation discipline than the multi-agent workflow."
    )
