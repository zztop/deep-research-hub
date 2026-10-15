from google.adk.agents import LlmAgent
from google.genai import types

from api.schemas import AggregatorInput, ResearchReport

from ..prompts import AGGREGATOR_DESCRIPTION, AGGREGATOR_INSTRUCTION

# mode="single_turn" exposes this agent to its parent as a callable tool
# whose args follow AggregatorInput: the topic plus the list of
# ResearcherResult objects collected from the researcher calls.
research_aggregator_agent = LlmAgent(
    name="research_aggregator",
    model="gemini-2.5-pro",
    description=AGGREGATOR_DESCRIPTION,
    instruction=AGGREGATOR_INSTRUCTION,
    mode="single_turn",
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    input_schema=AggregatorInput,
    output_schema=ResearchReport,
    generate_content_config=types.GenerateContentConfig(
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(initial_delay=1, attempts=3)
        ),
    ),
)
