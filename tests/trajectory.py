"""Trajectory checks: coordinator -> (researcher* -> aggregator -> verifier){1,2}."""

from dataclasses import dataclass, field

from deep_research_hub.agent import MAX_ATTEMPTS, MAX_RESEARCHERS, MIN_RESEARCHERS

RESEARCHER = "researcher"
AGGREGATOR = "research_aggregator"
VERIFIER = "research_verifier_agent"


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

    unexpected = sorted(
        {c for c in calls if c not in (RESEARCHER, AGGREGATOR, VERIFIER)}
    )
    if unexpected:
        problems.append(f"unexpected tool calls: {unexpected}")

    # Split into attempts, each ending at a verifier call.
    attempts: list[list[str]] = [[]]
    for c in calls:
        attempts[-1].append(c)
        if c == VERIFIER:
            attempts.append([])
    trailing = attempts.pop()
    if trailing:
        problems.append(f"calls after the last verifier call: {trailing}")

    if not 1 <= len(attempts) <= MAX_ATTEMPTS:
        problems.append(
            f"{len(attempts)} verified attempt(s), expected 1-{MAX_ATTEMPTS}"
        )
    for i, attempt in enumerate(attempts, 1):
        n = attempt.count(RESEARCHER)
        # Only the first attempt is bounded by the topic's complexity; a rerun
        # may use any valid number of sub-questions.
        lo, hi = (min_researchers, max_researchers) if i == 1 else (
            MIN_RESEARCHERS,
            MAX_RESEARCHERS,
        )
        if not lo <= n <= hi:
            problems.append(f"attempt {i}: researcher called {n}x, expected {lo}-{hi}")
        expected_tail = [AGGREGATOR, VERIFIER]
        if attempt[-2:] != expected_tail or attempt.count(AGGREGATOR) != 1:
            problems.append(
                f"attempt {i}: expected researchers, then one aggregator, then "
                f"the verifier; got {attempt}"
            )

    n_research = calls.count(RESEARCHER)

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
