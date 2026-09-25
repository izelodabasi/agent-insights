from datetime import UTC, datetime

from agent_insights.models import Event, Kind, Session
from agent_insights.timeline import build_event_index

TS = datetime(2026, 9, 1, tzinfo=UTC)


def test_marks_tools_failures_and_retries():
    command = {"command": "uv run pytest"}
    session = Session(
        agent="codex",
        session_id="session-1",
        project="demo",
        cwd="/demo",
        events=[
            Event(kind=Kind.PROMPT, timestamp=TS, text="run tests", tags={"correction"}),
            Event(kind=Kind.ASSISTANT, timestamp=TS, text="Running the tests."),
            Event(
                kind=Kind.TOOL_USE,
                timestamp=TS,
                tool="Bash",
                tool_id="first",
                tool_input=command,
            ),
            Event(
                kind=Kind.TOOL_RESULT,
                timestamp=TS,
                tool_id="first",
                text="failed",
                is_error=True,
            ),
            Event(
                kind=Kind.TOOL_USE,
                timestamp=TS,
                tool="Bash",
                tool_id="retry",
                tool_input=command,
            ),
            Event(kind=Kind.TOOL_RESULT, timestamp=TS, tool_id="retry", text="passed"),
        ],
    )

    timeline = build_event_index([session])["codex"]["session-1"]

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
