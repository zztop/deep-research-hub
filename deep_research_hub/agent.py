from typing import Any

from google.adk.agents import LlmAgent
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps import App
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from api.schemas import ResearchReport

from .prompts import COORDINATOR_INSTRUCTION
from .sub_agents.research_aggregator import research_aggregator_agent
from .sub_agents.researcher import researcher_agent
from .sub_agents.research_verifier import research_verifier_agent

MIN_RESEARCHERS = 2
MAX_RESEARCHERS = 5  # researcher results passed to one aggregator call
MAX_ATTEMPTS = 2  # research -> aggregate -> verify rounds per run
MAX_RESEARCHER_CALLS = MAX_RESEARCHERS * MAX_ATTEMPTS  # across all attempts
_CALLS_KEY = "temp:researcher_calls"  # "temp:" state lives for one invocation
_ATTEMPT_KEYS = {
    "research_aggregator": "temp:aggregator_calls",
    "research_verifier_agent": "temp:verifier_calls",
}


def enforce_researcher_limits(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
) -> dict[str, Any] | None:
    """Hard-enforce the researcher/attempt limits, whatever the model decides.

    Returning a dict short-circuits the tool call and is handed to the model
    as the tool result, so it can correct itself. The callback is synchronous,
    so the counter is safe against parallel tool calls.
    """
    if tool.name == "researcher":
        calls = tool_context.state.get(_CALLS_KEY, 0)
        if calls >= MAX_RESEARCHER_CALLS:
            return {
                "error": f"Limit reached: at most {MAX_RESEARCHER_CALLS} researcher "
                "calls are allowed. Do not call researcher again; call "
                "research_aggregator with the results you already have."
            }
        tool_context.state[_CALLS_KEY] = calls + 1
    elif tool.name == "research_aggregator":
        n = len(args.get("results") or [])
        if n < MIN_RESEARCHERS:
            return {
                "error": f"Only {n} researcher result(s) were provided; at least "
                f"{MIN_RESEARCHERS} are required. Call researcher for more "
                "sub-questions, then call research_aggregator again."
            }
        if n > MAX_RESEARCHERS:
            return {
                "error": f"{n} results were provided; pass at most "
                f"{MAX_RESEARCHERS} (the strongest ones)."
            }
    if tool.name in _ATTEMPT_KEYS:
        key = _ATTEMPT_KEYS[tool.name]
        attempts = tool_context.state.get(key, 0)
        if attempts >= MAX_ATTEMPTS:
            return {
                "error": f"Limit reached: {tool.name} may be called at most "
                f"{MAX_ATTEMPTS} times. Stop and return the best report you have."
            }
        tool_context.state[key] = attempts + 1
    return None


# researcher_agent and research_aggregator_agent have mode="single_turn", so
# attaching them via sub_agents makes the framework auto-expose each as a
# callable tool on this agent (see
# google.adk.agents.llm_agent.LlmAgent.model_post_init) rather than a chat
# hand-off.
#
# output_schema forces the final reply into ResearchReport's shape, which the
# API relies on. The aggregator does the synthesis; the coordinator just
# relays its report. This ADK version supports output_schema + tools together
# (tools are exposed during the thought loop, structure is only enforced on
# the final output).
root_agent = LlmAgent(
    name="coordinator",
    model="gemini-2.5-pro",
    description="""Decomposes a research topic, delegates sub-questions to the researcher agent, and has the aggregator agent synthesize a sourced report, which is validated by the verifier agent""",
    instruction=COORDINATOR_INSTRUCTION,
    sub_agents=[researcher_agent, research_aggregator_agent, research_verifier_agent],
    output_schema=ResearchReport,
    generate_content_config=types.GenerateContentConfig(
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            ),
        ],
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(initial_delay=1, attempts=3)
        ),
    ),
)


# Context caching is configured on the App and applies to every LLM agent in
# it. Each agent (coordinator, researcher, aggregator, verifier) has its own system
# instruction and tool set, so each gets its own cache and a transfer between
# agents no longer re-sends the whole prompt uncached. Gemini only caches once
# the prefix reaches its model minimum (2048 tokens on 2.5), and from a
# session's second request onward.
app = App(
    name="deep_research_hub",
    root_agent=root_agent,
    context_cache_config=ContextCacheConfig(),
)
