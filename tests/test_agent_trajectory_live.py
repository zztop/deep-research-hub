"""Live trajectory eval: runs the real agent and checks the path it took.

Calls Gemini, so it is opt-in:  uv run pytest -m live
Needs GOOGLE_API_KEY (loaded from .env if present).
"""

import os

import pytest
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from api.schemas import ResearchReport
from deep_research_hub.agent import root_agent

from .trajectory import AGGREGATOR, RESEARCHER, VERIFIER, Trajectory, check_trajectory

load_dotenv()

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set"),
]

# (topic, min researchers, max researchers). SIMPLE topics may use 2-3 angles,
# COMPLEX ones up to 5, so the bounds encode the coordinator's complexity rule.
CASES = [
    pytest.param("What is the boiling point of water at sea level?", 2, 3, id="simple"),
    pytest.param(
        "Is the Exodus of the Israelites from Egypt historical fact or legend?",
        3,
        5,
        id="complex-disputed-history",
    ),
]


async def run_agent(topic: str) -> Trajectory:
    service = InMemorySessionService()
    runner = Runner(app_name="eval", agent=root_agent, session_service=service)
    session = await service.create_session(app_name="eval", user_id="eval")
    t = Trajectory()
    message = types.Content(role="user", parts=[types.Part.from_text(text=topic)])

    async for event in runner.run_async(
        user_id="eval", session_id=session.id, new_message=message
    ):
        if event.author == "coordinator":
            t.coordinator_calls += [c.name for c in event.get_function_calls()]
            for r in event.get_function_responses():
                if r.name in (RESEARCHER, AGGREGATOR, VERIFIER) and "error" in (r.response or {}):
                    t.rejections.append(str(r.response["error"]))
        elif event.author == RESEARCHER and (
            event.grounding_metadata
            or any(c.name == "google_search" for c in event.get_function_calls())
        ):
            t.grounded_researchers += 1
        if event.author == "coordinator" and event.is_final_response() and event.content:
            t.final_text = "".join(p.text for p in event.content.parts or [] if p.text)
    return t


@pytest.mark.parametrize("topic,lo,hi", CASES)
async def test_correct_path(topic, lo, hi):
    t = await run_agent(topic)
    problems = check_trajectory(t, min_researchers=lo, max_researchers=hi)
    report = ResearchReport.model_validate_json(t.final_text or "{}")
    if not report.sources:
        problems.append("final report has no sources")
    assert not problems, f"trajectory={t.coordinator_calls}\n" + "\n".join(problems)
