import math
from datetime import UTC, date, datetime

import pytest

from agent_insights.models import Event, Kind, Session, Usage
from agent_insights.stats import (
    _cost_factor,
    build,
    commit_subject,
    performance,
    score_trend,
    summarize,
)

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


def test_score_decays_with_challenges_per_turn():
    s = _session({"disliked"}, {"interrupted"}, set(), set())
    expected = round(100 * math.exp(-3 * (1.0 + 0.4) / 4))
    assert performance(s) == {"score": expected, "turns": 4, "bad_turns": 2}


def test_failed_tool_calls_add_challenges_without_a_cap():
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


def test_all_agent_rows_keep_their_source_attribution():
    codex = _session(set())
    codex.agent = "codex"
    codex.session_id = "codex-session"
    codex.project = "shared"
    codex.usage["c"] = Usage(
        model="shared-model", timestamp=TS, output_tokens=10, reasoning_tokens=4
    )
    codex.reasoning_efforts.add("high")
    codex.events.append(Event(kind=Kind.TOOL_USE, timestamp=TS, tool="Read", tool_id="c"))

    copilot = _session(set())
    copilot.agent = "copilot"
    copilot.session_id = "copilot-session"
    copilot.project = "shared"
    copilot.usage["p"] = Usage(model="shared-model", timestamp=TS, output_tokens=20)
    copilot.events.append(Event(kind=Kind.TOOL_USE, timestamp=TS, tool="Read", tool_id="p"))

    result = summarize([codex, copilot], {})
    assert result["projects"][0]["agents"] == {"codex": 1, "copilot": 1}
    assert result["models"][0]["agents"] == {"codex": 1, "copilot": 1}
    assert result["models"][0]["reasoning"] == 4
    assert result["tools"][0]["agents"] == {"codex": 1, "copilot": 1}
    assert {message["agent"] for message in result["messages"]} == {"codex", "copilot"}
    assert result["totals"]["tokens_reasoning"] == 4
    assert result["totals"]["reasoning_efforts"] == {"high": 1}
    codex_row = next(row for row in result["sessions"] if row["agent"] == "codex")
    assert codex_row["reasoning_efforts"] == ["high"]
    assert {message["reasoning_tokens"] for message in result["messages"]} == {0, 4}


def test_build_includes_complete_rolling_period_views():
    recent = _session(set())
    recent.session_id = "recent"
    recent.events[0].timestamp = datetime(2026, 9, 25, tzinfo=UTC)
    old = _session(set())
    old.session_id = "old"
    old.events[0].timestamp = datetime(2026, 8, 1, tzinfo=UTC)

    result = build([recent, old], {}, ["claude-code"], until=date(2026, 9, 25))

    assert result["views"]["all"]["totals"]["sessions"] == 2
    assert result["period_views"]["7"]["all"]["totals"]["sessions"] == 1
    assert result["period_views"]["30"]["claude-code"]["totals"]["sessions"] == 1
    assert result["period_views"]["90"]["all"]["totals"]["sessions"] == 2


def test_session_timeline_marks_tools_failures_and_retries():
    session = _session({"correction"})
    command = {"command": "uv run pytest"}
    session.events += [
        Event(kind=Kind.ASSISTANT, timestamp=TS, text="Running the tests."),
        Event(kind=Kind.TOOL_USE, timestamp=TS, tool="Bash", tool_id="first", tool_input=command),
        Event(kind=Kind.TOOL_RESULT, timestamp=TS, tool_id="first", text="failed", is_error=True),
        Event(kind=Kind.TOOL_USE, timestamp=TS, tool="Bash", tool_id="retry", tool_input=command),
        Event(kind=Kind.TOOL_RESULT, timestamp=TS, tool_id="retry", text="passed"),
    ]

    timeline = build([session], {}, ["claude-code"], since=date(2026, 9, 1))[
        "session_events"
    ]["claude-code"]["s"]

    assert [event["kind"] for event in timeline] == [
        "prompt",
        "assistant",
        "tool_use",
        "tool_result",
        "tool_use",
        "tool_result",
    ]
    assert timeline[0]["flags"] == ["correction"]
    assert timeline[2]["flags"] == ["test"]
    assert timeline[3]["flags"] == ["failed"]
    assert timeline[4]["flags"] == ["retry", "test"]
    assert timeline[3]["tool"] == "Bash"
