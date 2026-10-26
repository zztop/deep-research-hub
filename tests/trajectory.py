"""Trajectory checks for the coordinator -> researcher* -> aggregator path."""

from dataclasses import dataclass, field

from deep_research_hub.agent import MAX_RESEARCHERS, MIN_RESEARCHERS

RESEARCHER = "researcher"
AGGREGATOR = "research_aggregator"


@dataclass
class Trajectory:
    """What one run did, in order, as seen on the ADK event stream."""

    # Coordinator tool calls, in order (names only).
    coordinator_calls: list[str] = field(default_factory=list)
    # Coordinator tool results that were guardrail rejections ({"error": ...}).
    rejections: list[str] = field(default_factory=list)
    # Number of researcher runs that used Google Search grounding.
    grounded_researchers: int = 0
    final_text: str | None = None


def check_trajectory(
    t: Trajectory,
    min_researchers: int = MIN_RESEARCHERS,
    max_researchers: int = MAX_RESEARCHERS,
) -> list[str]:
    """Return a list of violations; empty means the correct path was followed."""
    problems: list[str] = []
    calls = t.coordinator_calls

    unexpected = sorted({c for c in calls if c not in (RESEARCHER, AGGREGATOR)})
    if unexpected:
        problems.append(f"unexpected tool calls: {unexpected}")

    n_research = calls.count(RESEARCHER)
    if not min_researchers <= n_research <= max_researchers:
        problems.append(
            f"researcher called {n_research}x, expected "
            f"{min_researchers}-{max_researchers}"
        )

    if AGGREGATOR not in calls:
        problems.append("research_aggregator was never called")
    elif RESEARCHER in calls:
        last_researcher = len(calls) - 1 - calls[::-1].index(RESEARCHER)
        if calls.index(AGGREGATOR) < last_researcher:
            problems.append("aggregator was called before all researchers finished")

    if calls and calls[-1] != AGGREGATOR:
        problems.append(f"last tool call was {calls[-1]!r}, expected {AGGREGATOR!r}")

    if t.rejections:
        problems.append(
            f"model tripped {len(t.rejections)} guardrail(s) (self-corrected): "
            f"{t.rejections}"
        )

    if t.grounded_researchers < n_research:
        problems.append(
            f"only {t.grounded_researchers}/{n_research} researcher runs used "
            "Google Search"
        )

    if not t.final_text:
        problems.append("no final response")
    return problems
