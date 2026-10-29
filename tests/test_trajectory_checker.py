"""Offline tests: the checker and the guardrail callback, no model calls."""

from types import SimpleNamespace

from deep_research_hub.agent import enforce_researcher_limits

from .trajectory import AGGREGATOR, RESEARCHER, VERIFIER, Trajectory, check_trajectory


def good(n: int = 3) -> Trajectory:
    return Trajectory(
        coordinator_calls=[RESEARCHER] * n + [AGGREGATOR, VERIFIER],
        grounded_researchers=n,
        final_text="{}",
    )


def test_good_path_passes():
    assert check_trajectory(good()) == []


def test_rerun_after_low_score_passes():
    t = good(3)
    t.coordinator_calls += [RESEARCHER] * 2 + [AGGREGATOR, VERIFIER]
    t.grounded_researchers = 5
    assert check_trajectory(t) == []


def test_third_attempt_flagged():
    t = good(2)
    t.coordinator_calls *= 3
    t.grounded_researchers = 6
    assert any("expected 1-2" in p for p in check_trajectory(t))


def test_missing_verifier():
    t = good()
    t.coordinator_calls.pop()
    assert any("expected researchers" in p or "after the last" in p
               for p in check_trajectory(t))


def test_too_few_researchers():
    assert any("expected 2-5" in p for p in check_trajectory(good(1)))


def test_too_many_researchers():
    assert any("expected 2-5" in p for p in check_trajectory(good(6)))


def test_missing_aggregator():
    t = good()
    t.coordinator_calls = [RESEARCHER] * 3 + [VERIFIER]
    problems = check_trajectory(t)
    assert any("expected researchers" in p for p in problems)


def test_aggregator_before_researchers_finish():
    t = good()
    t.coordinator_calls = [RESEARCHER, AGGREGATOR, RESEARCHER, VERIFIER]
    assert any("expected researchers" in p for p in check_trajectory(t))


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


def test_guardrail_blocks_eleventh_researcher():
    ctx = _ctx()
    tool = SimpleNamespace(name=RESEARCHER)
    for _ in range(10):
        assert enforce_researcher_limits(tool, {}, ctx) is None
    assert "Limit reached" in enforce_researcher_limits(tool, {}, ctx)["error"]


def test_guardrail_rejects_aggregator_with_one_result():
    tool = SimpleNamespace(name=AGGREGATOR)
    out = enforce_researcher_limits(tool, {"results": [{}]}, _ctx())
    assert "at least 2" in out["error"]


def test_guardrail_allows_two_attempts_then_blocks():
    ctx = _ctx()
    results = {"results": [{}, {}]}
    for name, args in ((AGGREGATOR, results), (VERIFIER, {})):
        tool = SimpleNamespace(name=name)
        assert enforce_researcher_limits(tool, args, ctx) is None
        assert enforce_researcher_limits(tool, args, ctx) is None
        assert "at most 2" in enforce_researcher_limits(tool, args, ctx)["error"]


def test_rejected_aggregator_call_does_not_use_up_an_attempt():
    ctx = _ctx()
    tool = SimpleNamespace(name=AGGREGATOR)
    assert enforce_researcher_limits(tool, {"results": [{}]}, ctx) is not None
    assert enforce_researcher_limits(tool, {"results": [{}, {}]}, ctx) is None
    assert enforce_researcher_limits(tool, {"results": [{}, {}]}, ctx) is None
