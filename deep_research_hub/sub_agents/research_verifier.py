from google.adk.agents import LlmAgent
from pydantic import BaseModel, Field

from google.genai import types

from api.schemas import ResearchReport


class ResearchScore(BaseModel):
    summary: str = Field(description="A short synthesized summary of the findings.")
    original_research: ResearchReport = Field(
        description="Structure and aggregated research report"
    )
    score: int = Field(description="Assigned score to the research")


research_verifier_agent = LlmAgent(
    name="research_verifier_agent",
    model="gemini-2.5-pro",
    description="Verifies whether the summarized result of researched topic is appropriate",
    instruction="""You a research verifier that verifies that summarized result of the researched topic is valid and appropriate.
    You recieve the aggregated researched containing `summary`,`key_findings` and `sources`.
    Rules to validate and verify a research
    1. Validate that all the `sources` cited are correct
    2. Validate that the `key_findings` are factual and based on evidence
    3. Validate that the topic has been researched in depth and summarized correctly
    4. For complex topic determine that the topic has been researched with various perspective based on:
        - historical facts that are disputed, or where fact and fiction/myth must
             be told apart;
        - opinions or perspectives that differ by region, country or culture;
        - religious views or interpretations, including differences between
             traditions or denominations;
        - contested, politically or ethically charged, or multi-faceted questions
             that need deep research, careful thought and evaluation.
        - issues dealing with different philosophical discources 
    
    Produce the final structured output with:
    - `summary`: a short paragraph synthesizing the overall findings.
    - `original_research`: Aggregated researched results
    - `score`: assigned score to the research between 1-5, 5 being well researched and one being ill researched""",
    mode="single_turn",
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    input_schema=ResearchReport,
    output_schema=ResearchScore,
    generate_content_config=types.GenerateContentConfig(
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(initial_delay=1, attempts=3)
        ),
    ),
)
