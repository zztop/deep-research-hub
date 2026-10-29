from google.adk.agents import LlmAgent
from pydantic import BaseModel, Field

from google.genai import types

from api.schemas import ResearchReport


class VerifierInput(BaseModel):
    topic: str = Field(description="The original research topic.")
    report: ResearchReport = Field(description="The aggregator's report, unchanged.")


class ResearchScore(BaseModel):
    summary: str = Field(description="A short summary of the verification outcome.")
    score: int = Field(ge=1, le=5, description="Research quality score, 1-5.")
    feedback: str = Field(
        description="What is missing, unsupported or one-sided, and what a rerun "
        "should research instead. Empty string if score is 3 or higher."
    )


research_verifier_agent = LlmAgent(
    name="research_verifier_agent",
    model="gemini-2.5-flash",
    description="Verifies whether the summarized result of researched topic is appropriate",
    instruction="""You are a research verifier. You receive the original `topic`
    and a `report` containing `summary`, `key_findings` and `sources`, and judge
    whether the report answers the topic validly and appropriately.
    Rules to validate and verify a research
    1. Check that the `sources` are plausible (a real-looking title and an
       http(s) URL) and that they support the findings they back. You cannot
       open URLs, so do not claim to have verified that a page exists.
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
    - `summary`: one short paragraph on how well the topic was researched.
    - `score`: 1-5; 5 is thorough, balanced and well sourced, 1 is poorly researched.
    - `feedback`: if the score is below 3, the concrete gaps and what to
      research next; otherwise an empty string.""",
    mode="single_turn",
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    input_schema=VerifierInput,
    output_schema=ResearchScore,
    generate_content_config=types.GenerateContentConfig(
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(initial_delay=1, attempts=3)
        ),
    ),
)
