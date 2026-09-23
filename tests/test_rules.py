from datetime import UTC, datetime

from agent_insights.models import Event, Kind, Session
from agent_insights.tagging import rules

TS = datetime(2026, 9, 1, tzinfo=UTC)


def _use(tool: str, tool_id: str, **inp) -> Event:
    return Event(kind=Kind.TOOL_USE, timestamp=TS, tool=tool, tool_id=tool_id, tool_input=inp)


def _result(tool_id: str, error: bool = False) -> Event:
    return Event(kind=Kind.TOOL_RESULT, timestamp=TS, tool_id=tool_id, is_error=error)


def _session(*window: Event) -> Session:
    prompt = Event(kind=Kind.PROMPT, timestamp=TS, text="go")
    return Session(
        agent="claude-code", session_id="s", project="p", cwd="", events=[prompt, *window]
    )


def test_testing_and_debugging_tags():
    s = _session(
        _use("Bash", "1", command="uv run pytest -x"),
        _result("1", error=True),
        _use("Edit", "2", file_path="a.py"),
        _result("2"),
    )
    rules.tag_session(s)
    assert {"testing", "debugging"} <= s.events[0].tags


def test_edit_without_prior_failure_is_not_debugging():
    s = _session(_use("Edit", "1", file_path="a.py"), _result("1"))
    rules.tag_session(s)
    assert "debugging" not in s.events[0].tags
