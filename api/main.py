import logging

from fastapi import FastAPI, HTTPException
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from deep_research_hub.agent import root_agent

from .schemas import RESEARCH_REPORT_EXAMPLE, ResearchReport, ResearchRequest

logger = logging.getLogger("deep_research_hub.api")

APP_NAME = "deep_research_hub"
CALLER_USER_ID = "api-caller"

app = FastAPI(title="Deep Research Hub")

session_service = InMemorySessionService()
runner = Runner(app_name=APP_NAME, agent=root_agent, session_service=session_service)


@app.post(
    "/research",
    response_model=ResearchReport,
    responses={
        200: {"content": {"application/json": {"example": RESEARCH_REPORT_EXAMPLE}}}
    },
)
async def research(request: ResearchRequest) -> ResearchReport:
    session = await session_service.create_session(
        app_name=APP_NAME, user_id=CALLER_USER_ID
    )
    message = types.Content(
        role="user", parts=[types.Part.from_text(text=request.topic)]
    )

    final_text: str | None = None
    agent_error: str | None = None
    logger.info("research started session=%s topic=%r", session.id, request.topic)
    try:
        async for event in runner.run_async(
            user_id=CALLER_USER_ID, session_id=session.id, new_message=message
        ):
            logger.info(
                "event author=%s fcalls=%s fresps=%s final=%s",
                event.author,
                [c.name for c in event.get_function_calls()],
                [r.name for r in event.get_function_responses()],
                event.is_final_response(),
            )
            if event.error_code:
                agent_error = f"{event.error_code}: {event.error_message or ''}"
                logger.error(
                    "agent error session=%s author=%s error_code=%s error_message=%r",
                    session.id,
                    event.author,
                    event.error_code,
                    event.error_message,
                )
            if event.is_final_response() and event.content and event.content.parts:
                final_text = "".join(
                    part.text for part in event.content.parts if part.text
                )
    finally:
        await session_service.delete_session(
            app_name=APP_NAME, user_id=CALLER_USER_ID, session_id=session.id
        )

    if not final_text:
        detail = "Agent produced no response"
        if agent_error:
            detail = f"{detail} ({agent_error})"
        logger.error("research failed session=%s: %s", session.id, detail)
        raise HTTPException(status_code=502, detail=detail)

    try:
        result = ResearchReport.model_validate_json(final_text)
        logger.info("research done session=%s sources=%d", session.id, len(result.sources))
        return result
    except ValueError as exc:
        logger.error(
            "invalid report session=%s: %s; raw=%.500r", session.id, exc, final_text
        )
        raise HTTPException(
            status_code=502,
            detail="Agent response did not match the expected report shape",
        ) from exc
