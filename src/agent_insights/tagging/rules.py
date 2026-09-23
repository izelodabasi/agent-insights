import re

from agent_insights.models import Event, Kind, Session

EDIT_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}
TEST_COMMAND = re.compile(
    r"\b(pytest|jest|vitest|mocha|tox|nox|go test|cargo test|(npm|yarn|pnpm|bun)( run)? test)\b"
)


def tag_session(session: Session) -> None:
    """Tags read straight off the log. A turn-level tag goes on the prompt that opened it."""
    for e in session.events:
        match e.kind:
            case Kind.REJECTION:
                e.tags.add("rejected")
            case Kind.INTERRUPT:
                e.tags.add("interrupted")
    commits = {id(e) for e in successful_commits(session)}
    for opener, window in turns(session):
        if any(id(e) in commits for e in window):
            opener.tags.add("committed")
        if _is_low_performance(window):
            opener.tags.add("low_performance")
        if any(_is_test(e) for e in window):
            opener.tags.add("testing")
        if _is_debugging(window):
            opener.tags.add("debugging")


def turns(session: Session) -> list[tuple[Event, list[Event]]]:
    """Each prompt with everything that happened before the next one."""
    turns: list[tuple[Event, list[Event]]] = []
    for e in session.events:
        if e.kind == Kind.PROMPT:
            turns.append((e, []))
        elif turns:
            turns[-1][1].append(e)

    return turns


def successful_commits(session: Session) -> list[Event]:
    ok = {e.tool_id for e in session.of(Kind.TOOL_RESULT) if not e.is_error}
    return [e for e in session.of(Kind.TOOL_USE) if "git commit" in _command(e) and e.tool_id in ok]


def _is_low_performance(window: list[Event]) -> bool:
    """3+ failed tool calls, or one file edited 3+ times, before the next prompt."""
    errors = sum(1 for e in window if e.kind == Kind.TOOL_RESULT and e.is_error)
    edits: dict[str, int] = {}
    for e in window:
        if e.kind == Kind.TOOL_USE and e.tool in EDIT_TOOLS:
            path = str(e.tool_input.get("file_path", ""))
            edits[path] = edits.get(path, 0) + 1
    return errors >= 3 or max(edits.values(), default=0) >= 3


def _is_test(e: Event) -> bool:
    return bool(TEST_COMMAND.search(_command(e)))


def _is_debugging(window: list[Event]) -> bool:
    failed = False
    for e in window:
        if e.kind == Kind.TOOL_RESULT and e.is_error:
            failed = True
        elif failed and e.kind == Kind.TOOL_USE and e.tool in EDIT_TOOLS:
            return True

    return False


def _command(e: Event) -> str:
    if e.kind != Kind.TOOL_USE or e.tool != "Bash":
        return ""

    return str(e.tool_input.get("command", ""))
