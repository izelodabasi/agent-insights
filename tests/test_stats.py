import math
from datetime import UTC, datetime

import pytest

from agent_insights.models import Event, Kind, Session
from agent_insights.stats import _cost_factor, commit_subject, performance, score_trend

TS = datetime(2026, 9, 1, tzinfo=UTC)


def _session(*turn_tags: set[str], commit: bool = False) -> Session:
    events = [Event(kind=Kind.PROMPT, timestamp=TS, text="x", tags=t) for t in turn_tags]
    if commit:
        events += [
            Event(
                kind=Kind.TOOL_USE,
                timestamp=TS,
                tool="Bash",
                tool_id="c",
                tool_input={"command": "git commit -m 'x'"},
            ),
            Event(kind=Kind.TOOL_RESULT, timestamp=TS, tool_id="c"),
        ]
    return Session(agent="claude-code", session_id="s", project="p", cwd="", events=events)


def test_score_decays_with_friction_per_turn():
    s = _session({"frustrated"}, {"interrupted"}, set(), set())
    expected = round(100 * math.exp(-3 * (1.0 + 0.4) / 4))
    assert performance(s) == {"score": expected, "turns": 4, "bad_turns": 2}


def test_failed_tool_calls_add_friction_without_a_cap():
    s = _session(set())
    s.events += [
        Event(kind=Kind.TOOL_RESULT, timestamp=TS, tool_id=str(i), is_error=True) for i in range(12)
    ]
    assert performance(s)["score"] == round(100 * math.exp(-3 * 1.2))


def test_score_trend_uses_a_rolling_window_and_thins_long_sessions():
    assert score_trend([0, 0, 1.0], window=2) == [100, 100, round(100 * math.exp(-1.5))]
    assert len(score_trend([0.0] * 300, points=60)) == 60


@pytest.mark.parametrize(
    ("total", "calls", "typical", "factor"),
    [(2.0, 1, 1.0, 0.9), (0.5, 1, 1.0, 1.1), (1.0, 1, 1.0, 1.0), (100.0, 1, 1.0, 0.9)],
)
def test_cost_factor(total: float, calls: int, typical: float, factor: float):
    assert _cost_factor(total, calls, typical) == pytest.approx(factor)


def test_commit_multiplies_score_and_caps_at_100():
    with_commit = performance(_session({"interrupted"}, set(), commit=True))["score"]
    assert with_commit == round(100 * math.exp(-3 * 0.2) * 1.1)
    assert performance(_session(set(), commit=True))["score"] == 100


@pytest.mark.parametrize(
    ("command", "subject"),
    [
        ('git commit -m "fix parser"', "fix parser"),
        ("git add a.py && git commit -m 'feat: x'", "feat: x"),
        (
            "git commit -m \"$(cat <<'EOF'\nfeat: split parser\n\nbody\nEOF\n)\"",
            "feat: split parser",
        ),
        ("git commit --amend --no-edit", "commit"),
    ],
)
def test_commit_subject(command: str, subject: str):
    assert commit_subject(command) == subject
