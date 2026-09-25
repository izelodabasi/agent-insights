import json
from collections import defaultdict

from agent_insights.models import Kind, Session
from agent_insights.tagging.rules import EDIT_TOOLS, TEST_COMMAND, successful_commits

type TimelineEvent = dict[str, object]
type TimelineIndex = dict[str, dict[str, list[TimelineEvent]]]


def build_event_index(sessions: list[Session]) -> TimelineIndex:
    """Build one timeline per agent and source session ID."""
    timelines: dict[str, dict[str, list[TimelineEvent]]] = defaultdict(dict)
    for session in sessions:
        timelines[session.agent][session.session_id] = _timeline(session)
    return dict(timelines)


def _timeline(session: Session) -> list[TimelineEvent]:
    calls: dict[str, tuple[str, str]] = {}
    failed: set[tuple[str, str]] = set()
    commits = {id(event) for event in successful_commits(session)}
    timeline = []

    for event in session.events:
        flags = set(event.tags)
        tool = event.tool
        detail = event.text

        if event.kind == Kind.TOOL_USE:
            detail = json.dumps(event.tool_input, ensure_ascii=False, indent=2, default=str)
            signature = (event.tool, detail)
            calls[event.tool_id] = signature
            if signature in failed:
                flags.add("retry")
            if event.tool in EDIT_TOOLS:
                flags.add("edit")
            if _is_test(event.tool, event.tool_input):
                flags.add("test")
            if id(event) in commits:
                flags.add("commit")
        elif event.kind == Kind.TOOL_RESULT:
            signature = calls.get(event.tool_id)
            tool = signature[0] if signature else ""
            if event.is_error:
                flags.add("failed")
                if signature:
                    failed.add(signature)

        clipped, truncated = _clip(detail)
        timeline.append(
            {
                "kind": event.kind.value,
                "ts": event.timestamp.astimezone().isoformat(timespec="seconds"),
                "tool": tool,
                "detail": clipped,
                "truncated": truncated,
                "sidechain": event.sidechain,
                "flags": sorted(flags),
            }
        )

    return timeline


def _is_test(tool: str, tool_input: dict) -> bool:
    return tool == "Bash" and bool(TEST_COMMAND.search(str(tool_input.get("command", ""))))


def _clip(value: str, limit: int = 4000) -> tuple[str, bool]:
    return value[:limit], len(value) > limit
