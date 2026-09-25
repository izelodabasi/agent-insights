from datetime import UTC, date, datetime

import pytest

from agent_insights.date_filter import filter_sessions, resolve_range
from agent_insights.models import Event, Kind, Session, Usage


def _at(day: int, hour: int = 12) -> datetime:
    return datetime(2026, 9, day, hour, tzinfo=UTC)


def _session() -> Session:
    return Session(
        agent="codex",
        session_id="session-1",
        project="demo",
        cwd="/demo",
        events=[
            Event(kind=Kind.PROMPT, timestamp=_at(1), text="first"),
            Event(kind=Kind.ASSISTANT, timestamp=_at(1, 13), text="done"),
            Event(kind=Kind.PROMPT, timestamp=_at(2), text="second"),
        ],
        usage={
            "first": Usage(model="gpt-5", timestamp=_at(1), output_tokens=10),
            "second": Usage(model="gpt-5", timestamp=_at(2), output_tokens=20),
        },
    )


def test_filters_events_and_usage_with_inclusive_boundaries():
    [filtered] = filter_sessions(
        [_session()], since=date(2026, 9, 2), until=date(2026, 9, 2)
    )

    assert [event.text for event in filtered.events] == ["second"]
    assert list(filtered.usage) == ["second"]
    assert filtered.start == _at(2)


def test_undated_usage_uses_the_original_session_start():
    session = _session()
    session.usage["undated"] = Usage(model="gpt-5", output_tokens=30)

    [filtered] = filter_sessions([session], since=date(2026, 9, 2))

    assert list(filtered.usage) == ["second"]


def test_sessions_without_events_in_range_are_removed():
    assert filter_sessions([_session()], since=date(2026, 9, 3)) == []


def test_no_range_returns_sessions_unchanged():
    sessions = [_session()]
    assert filter_sessions(sessions) is sessions


def test_rejects_reversed_range():
    with pytest.raises(ValueError, match="--since must be on or before --until"):
        filter_sessions([_session()], since=date(2026, 9, 2), until=date(2026, 9, 1))


def test_resolves_rolling_range_ending_today():
    assert resolve_range(None, None, 7, today=date(2026, 9, 25)) == (
        date(2026, 9, 19),
        date(2026, 9, 25),
    )


def test_resolves_rolling_range_ending_on_until_date():
    assert resolve_range(None, date(2026, 9, 10), 30) == (
        date(2026, 8, 12),
        date(2026, 9, 10),
    )
