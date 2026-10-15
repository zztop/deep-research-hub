from typing import List, Optional
from google.genai.types import GenerateContentConfig, HttpOptions, HttpRetryOptions

from google.adk.agents import LlmAgent
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.agents.callback_context import CallbackContext
from pydantic import types

from api.schemas import ResearcherResult

from ..prompts import RESEARCHER_DESCRIPTION, RESEARCHER_INSTRUCTION

# def store_research_results(callback_context: CallbackContext)-> Optional[types.Content]:
#     agent_name = callback_context.agent_name

#     invocation_id = callback_context.invocation_id

#     current_state = callback_context.state.to_dict()

#     all_research:List[ResearcherResult] = current_state.get('research',[])
#     all_research.append(callback_context.ou)


# mode="single_turn" exposes this agent to its parent as a callable tool
# (args: {"request": str}) that runs inline and returns a result, instead of
# the default "chat" mode which hands control away via transfer_to_agent.
researcher_agent = LlmAgent(
    name="researcher",
    model="gemini-2.5-flash",
    description=RESEARCHER_DESCRIPTION,
    instruction=RESEARCHER_INSTRUCTION,
    tools=[GoogleSearchTool(bypass_multi_tools_limit=True)],
    mode="single_turn",
    output_schema=ResearcherResult,
    generate_content_config=GenerateContentConfig(
        http_options=HttpOptions(
            retry_options=HttpRetryOptions(initial_delay=1, attempts=3)
        ),
    ),
    # output_key="researched_item",
    # after_agent_callback=store_research_results
)
