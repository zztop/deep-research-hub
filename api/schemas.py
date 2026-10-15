from pydantic import BaseModel, ConfigDict, Field


class ResearchRequest(BaseModel):
    topic: str = Field(
        min_length=1,
        max_length=500,
        description="The research topic or question to investigate.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"topic": "Impact of remote work on urban commercial real estate"}
            ]
        }
    )


class Source(BaseModel):
    title: str
    # Plain str, not HttpUrl: this model is used as an ADK output_schema, and
    # ADK's SetModelResponseTool does `model_dump()` + raw `json.dumps()` on
    # the result, which crashes on HttpUrl (not JSON-serializable) at
    # google/adk/flows/llm_flows/prompt/_schema.py:get_structured_model_response.
    url: str


class ResearcherResult(BaseModel):
    """Fixed output of one researcher call; a list of these feeds the aggregator."""

    sub_question: str = Field(description="The sub-question that was researched.")
    answer: str = Field(description="Concise, factual answer in 3-6 sentences.")
    sources: list[Source] = Field(description="Sources used for the answer.")
    inconclusive: bool = Field(
        description="True if the search results were inconclusive for this sub-question."
    )


class AggregatorInput(BaseModel):
    """Input to the research aggregator: the topic plus every researcher result."""

    topic: str = Field(description="The original research topic.")
    results: list[ResearcherResult] = Field(
        description="One entry per researcher call, unmodified."
    )


class ResearchReport(BaseModel):
    summary: str = Field(description="A short synthesized summary of the findings.")
    key_findings: list[str] = Field(description="Bulleted list of key findings.")
    sources: list[Source] = Field(
        description="Unique sources cited across all findings."
    )

    # No json_schema_extra "examples" here: this model is an ADK output_schema,
    # and ADK converts it to a genai Schema, which rejects an `examples` key
    # (pydantic "extra_forbidden"). The OpenAPI example lives in
    # RESEARCH_REPORT_EXAMPLE, attached to the route in api/main.py.


RESEARCH_REPORT_EXAMPLE = {
    "summary": (
        "Remote work has reduced downtown office demand while increasing "
        "demand for suburban flexible-workspace properties."
    ),
    "key_findings": [
        "Downtown office vacancy rates rose in major metros between 2020 and 2025.",
        "Suburban flex-office and co-working demand grew over the same period.",
    ],
    "sources": [
        {
            "title": "Urban Office Vacancy Trends Report",
            "url": "https://example.com/office-vacancy-trends",
        }
    ],
}
