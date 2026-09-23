import json
import re
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from agent_insights.models import Event, Kind, Session, Usage
from agent_insights.repos import repo_name

ROOT = Path.home() / ".claude" / "projects"
REJECTION = "The user doesn't want to proceed with this tool use"
INTERRUPT = "[Request interrupted by user"
REASON = re.compile(r"the user said:\n(.*?)(?:\n\nNote:|$)", re.S)
COMMAND_NAME = re.compile(r"<command-name>/?([^<]+)</command-name>")


def installed(root: Path = ROOT) -> bool:
    return root.is_dir()


def load_sessions(root: Path = ROOT) -> list[Session]:
    """A continued session is a new file that repeats the earlier session's records under
    the same uuids, so files are read oldest first and a record or API message already
    seen stays with the session it first appeared in."""
    sessions: dict[str, Session] = {}
    seen: set[str] = set()
    usage_owner: dict[str, str] = {}
    for path in sorted(root.glob("*/*.jsonl"), key=lambda p: p.stat().st_mtime):
        _read(path, sessions, seen, usage_owner, sidechain=False)
    for path in sorted(root.glob("*/*/subagents/*.jsonl"), key=lambda p: p.stat().st_mtime):
        _read(path, sessions, seen, usage_owner, sidechain=True, session_id=path.parent.parent.name)
    for session in sessions.values():
        session.events.sort(key=lambda e: e.timestamp)

    return [s for s in sessions.values() if s.events]


def _records(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _read(
    path: Path,
    sessions: dict[str, Session],
    seen: set[str],
    usage_owner: dict[str, str],
    sidechain: bool,
    session_id: str | None = None,
) -> None:
    titles: dict[str, dict[str, str]] = {}
    for rec in _records(path):
        match rec.get("type"):
            case "custom-title" if rec.get("customTitle"):
                titles.setdefault(rec.get("sessionId", ""), {})["title"] = rec["customTitle"]
            case "ai-title" if rec.get("aiTitle"):
                titles.setdefault(rec.get("sessionId", ""), {})["ai_title"] = rec["aiTitle"]
        if rec.get("type") not in ("user", "assistant") or "timestamp" not in rec:
            continue

        if uuid := rec.get("uuid"):
            if uuid in seen:
                continue

            seen.add(uuid)
        sid = session_id or rec.get("sessionId") or path.stem
        session = sessions.get(sid)
        if session is None:
            cwd = rec.get("cwd", "")
            session = Session(
                agent="claude-code",
                session_id=sid,
                project=repo_name(cwd) or path.parent.name,
                cwd=cwd,
            )
            sessions[sid] = session
        if branch := rec.get("gitBranch"):
            session.branches.add(branch)

        ts = datetime.fromisoformat(rec["timestamp"])
        side = sidechain or bool(rec.get("isSidechain"))
        if rec["type"] == "user":
            session.events.extend(_user_events(rec, ts, side))
        else:
            session.events.extend(_assistant_events(rec, ts, side))
            _add_usage(session, rec, usage_owner)
    for sid, found in titles.items():
        if session := sessions.get(sid):
            session.title = found.get("title", session.title)
            session.ai_title = found.get("ai_title", session.ai_title)


def _user_events(rec: dict, ts: datetime, sidechain: bool) -> Iterator[Event]:
    if rec.get("isMeta"):
        return

    content = rec.get("message", {}).get("content")
    if isinstance(content, str):
        yield from _user_text(content, ts, sidechain)
        return

    texts = []
    for block in content or []:
        match block.get("type"):
            case "text":
                texts.append(block.get("text", ""))
            case "tool_result":
                yield _tool_result(block, ts, sidechain)
    if texts:
        yield from _user_text("\n".join(texts), ts, sidechain)


def _user_text(text: str, ts: datetime, sidechain: bool) -> Iterator[Event]:
    stripped = text.lstrip()
    if stripped.startswith(INTERRUPT):
        yield Event(kind=Kind.INTERRUPT, timestamp=ts, sidechain=sidechain)
    elif stripped.startswith("<task-notification"):
        yield Event(kind=Kind.NOTIFICATION, timestamp=ts, sidechain=sidechain)
    elif m := COMMAND_NAME.search(text):
        yield Event(kind=Kind.COMMAND, timestamp=ts, text=m.group(1).strip(), sidechain=sidechain)
    elif stripped.startswith(("<local-command", "<command-", "<system-reminder")):
        return
    elif stripped and not sidechain:
        yield Event(kind=Kind.PROMPT, timestamp=ts, text=text)


def _tool_result(block: dict, ts: datetime, sidechain: bool) -> Event:
    body = block.get("content")
    if isinstance(body, list):
        body = "\n".join(b.get("text", "") for b in body if isinstance(b, dict))
    body = body or ""
    tool_id = block.get("tool_use_id", "")
    if body.startswith(REJECTION):
        m = REASON.search(body)
        return Event(
            kind=Kind.REJECTION,
            timestamp=ts,
            text=m.group(1).strip() if m else "",
            tool_id=tool_id,
            sidechain=sidechain,
        )

    return Event(
        kind=Kind.TOOL_RESULT,
        timestamp=ts,
        text=body[:500],
        tool_id=tool_id,
        is_error=bool(block.get("is_error")),
        sidechain=sidechain,
    )


def _assistant_events(rec: dict, ts: datetime, sidechain: bool) -> Iterator[Event]:
    for block in rec.get("message", {}).get("content") or []:
        match block.get("type"):
            case "text" if block.get("text", "").strip():
                yield Event(
                    kind=Kind.ASSISTANT, timestamp=ts, text=block["text"], sidechain=sidechain
                )
            case "tool_use":
                yield Event(
                    kind=Kind.TOOL_USE,
                    timestamp=ts,
                    tool=block.get("name", ""),
                    tool_id=block.get("id", ""),
                    tool_input=block.get("input") or {},
                    sidechain=sidechain,
                )


def _add_usage(session: Session, rec: dict, usage_owner: dict[str, str]) -> None:
    msg = rec.get("message", {})
    usage = msg.get("usage")
    model = msg.get("model", "")
    if not usage or model == "<synthetic>":
        return

    key = msg.get("id") or rec.get("requestId") or rec.get("uuid")
    if usage_owner.setdefault(key, session.session_id) != session.session_id:
        return

    session.usage[key] = Usage(
        model=model,
        timestamp=datetime.fromisoformat(rec["timestamp"]),
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        cache_write_tokens=usage.get("cache_creation_input_tokens", 0),
        cache_write_1h_tokens=(usage.get("cache_creation") or {}).get(
            "ephemeral_1h_input_tokens", 0
        ),
        cache_read_tokens=usage.get("cache_read_input_tokens", 0),
    )
