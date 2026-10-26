"""Offline tests: the checker and the guardrail callback, no model calls."""

from types import SimpleNamespace

from deep_research_hub.agent import enforce_researcher_limits

from .trajectory import AGGREGATOR, RESEARCHER, Trajectory, check_trajectory


def good(n: int = 3) -> Trajectory:
    return Trajectory(
        coordinator_calls=[RESEARCHER] * n + [AGGREGATOR],
        grounded_researchers=n,
        final_text="{}",
    )


def test_good_path_passes():
    assert check_trajectory(good()) == []


def test_too_few_researchers():
    assert any("expected 2-5" in p for p in check_trajectory(good(1)))


def test_too_many_researchers():
    assert any("expected 2-5" in p for p in check_trajectory(good(6)))


def test_missing_aggregator():
    t = good()
    t.coordinator_calls.pop()
    problems = check_trajectory(t)
    assert any("never called" in p for p in problems)


def test_aggregator_before_researchers_finish():
    t = good()
    t.coordinator_calls = [RESEARCHER, AGGREGATOR, RESEARCHER, AGGREGATOR]
    assert any("before all researchers" in p for p in check_trajectory(t))


def test_ungrounded_researcher_flagged():
    t = good()
    t.grounded_researchers = 1
    assert any("Google Search" in p for p in check_trajectory(t))


def test_unexpected_tool_flagged():
    t = good()
    t.coordinator_calls.insert(0, "transfer_to_agent")
    assert any("unexpected" in p for p in check_trajectory(t))


def _ctx():
    return SimpleNamespace(state={})


def test_guardrail_blocks_sixth_researcher():
    ctx = _ctx()
    tool = SimpleNamespace(name=RESEARCHER)
    for _ in range(5):
        assert enforce_researcher_limits(tool, {}, ctx) is None
    assert "Limit reached" in enforce_researcher_limits(tool, {}, ctx)["error"]


def test_guardrail_rejects_aggregator_with_one_result():
    tool = SimpleNamespace(name=AGGREGATOR)
    out = enforce_researcher_limits(tool, {"results": [{}]}, _ctx())
    assert "at least 2" in out["error"]
