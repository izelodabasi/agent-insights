import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

from agent_insights.models import Event, Kind, Session, Usage
from agent_insights.repos import repo_name

ROOT = Path.home() / ".codex"
TOOL_NAMES = {
    "exec": "Bash",
    "exec_command": "Bash",
    "read_file": "Read",
    "write_file": "Edit",
    "apply_diff": "Edit",
    "apply_patch": "Edit",
    "spawn_agent": "Agent",
    "close_agent": "Agent",
    "wait_agent": "Agent",
    "read_dir": "Glob",
}
PATCH_PATH = re.compile(r"^\*\*\* (?:Add|Delete|Update) File: (.+)$", re.M)


def installed(root: Path | None = None) -> bool:
    root = _root(root)
    return (root / "sessions").is_dir() or (root / "archived_sessions").is_dir()


def load_sessions(root: Path | None = None) -> list[Session]:
    """Load Codex rollout files and fold subagent work into its parent session.

    Codex subagent rollouts replay the parent's history around their creation time. Events in
    the first five seconds of a child rollout are consequently ignored, matching Codex's fork
    boundary convention and preventing the repeated history from inflating usage.
    """
    root = _root(root)
    seen_usage: set[tuple] = set()
    parsed = [_read(path, seen_usage) for path in _paths(root)]
    parsed = [item for item in parsed if item is not None]
    by_id = {item[0].session_id: item[0] for item in parsed if not item[1]}
    orphans: list[Session] = []
    for child, parent_id in (item for item in parsed if item[1]):
        if parent := by_id.get(parent_id):
            parent.events.extend(child.events)
            parent.usage.update(
                {f"{child.session_id}:{key}": value for key, value in child.usage.items()}
            )
            parent.branches |= child.branches
        else:
            orphans.append(child)

    sessions = [*by_id.values(), *orphans]
    for session in sessions:
        session.events.sort(key=lambda event: event.timestamp)
    return [session for session in sessions if session.events]


def _paths(root: Path) -> list[Path]:
    dated = sorted(root.glob("sessions/[0-9][0-9][0-9][0-9]/[0-9][0-9]/[0-9][0-9]/rollout-*.jsonl"))
    archived = sorted((root / "archived_sessions").glob("rollout-*.jsonl"))
    seen: set[str] = set()
    paths = []
    for path in (*dated, *archived):
        if path.name not in seen:
            seen.add(path.name)
            paths.append(path)
    return paths


def _read(path: Path, seen_usage: set[tuple]) -> tuple[Session, str] | None:
    if not _valid_rollout(path):
        return None

    session: Session | None = None
    parent_id = ""
    fork_cutoff: datetime | None = None
    model = ""
    previous_info = ""
    previous_total: dict[str, int] = {}
    previous_cumulative: int | None = None

    for line_number, rec in enumerate(_records(path), start=1):
        payload = rec.get("payload")
        if not isinstance(payload, dict):
            continue
        timestamp = _timestamp(rec.get("timestamp"))
        record_type = rec.get("type")
        payload_type = payload.get("type")

        if record_type == "session_meta":
            session_id = _string(payload.get("id") or payload.get("session_id")) or path.stem
            cwd = _string(payload.get("cwd"))
            if session is None:
                session = Session(
                    agent="codex",
                    session_id=session_id,
                    project=repo_name(cwd) or path.parent.name,
                    cwd=cwd,
                )
            git = payload.get("git")
            if isinstance(git, dict) and (branch := _string(git.get("branch"))):
                session.branches.add(branch)
            if not parent_id:
                parent_id = _parent_id(payload)
                if parent_id and timestamp:
                    fork_cutoff = timestamp + timedelta(seconds=5)
            model = _string(payload.get("model")) or model
            continue

        if session is None:
            session = Session(
                agent="codex",
                session_id=path.stem,
                project=path.parent.name,
                cwd="",
            )

        if record_type == "turn_context":
            model = _string(payload.get("model")) or model
            continue

        sidechain = bool(parent_id)
        replay = bool(sidechain and timestamp and fork_cutoff and timestamp < fork_cutoff)
        if timestamp is None:
            continue

        if record_type == "response_item" and payload_type == "message":
            role = payload.get("role")
            text = _message_text(payload.get("content"), role)
            if not text:
                continue
            if role == "user" and not sidechain:
                session.events.append(Event(kind=Kind.PROMPT, timestamp=timestamp, text=text))
            elif role == "assistant" and not replay:
                session.events.append(
                    Event(kind=Kind.ASSISTANT, timestamp=timestamp, text=text, sidechain=sidechain)
                )
            continue

        if replay:
            continue

        if record_type == "response_item" and payload_type in {
            "function_call",
            "custom_tool_call",
        }:
            session.events.append(_tool_use(payload, timestamp, sidechain))
        elif record_type == "response_item" and payload_type in {
            "function_call_output",
            "custom_tool_call_output",
        }:
            session.events.append(_tool_result(payload, timestamp, sidechain))
        elif record_type == "event_msg" and payload_type == "turn_aborted" and not sidechain:
            session.events.append(Event(kind=Kind.INTERRUPT, timestamp=timestamp))
        elif record_type == "event_msg" and payload_type == "token_count":
            info = payload.get("info")
            if not isinstance(info, dict):
                continue
            identity = json.dumps(info, sort_keys=True, separators=(",", ":"))
            if identity == previous_info:
                continue
            previous_info = identity
            usage, previous_total, cumulative = _usage(info, previous_total, model, timestamp)
            if cumulative is not None and cumulative == previous_cumulative:
                continue
            previous_cumulative = cumulative
            if key := _usage_key(info, parent_id or session.session_id):
                if key in seen_usage:
                    continue
                seen_usage.add(key)
            if usage:
                session.usage[f"token-{line_number}"] = usage

    return (session, parent_id) if session is not None else None


def _records(path: Path):
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield value


def _valid_rollout(path: Path) -> bool:
    """Codex rollouts always begin with a session_meta object."""
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            first = json.loads(handle.readline())
    except (OSError, json.JSONDecodeError):
        return False
    payload = first.get("payload") if isinstance(first, dict) else None
    return first.get("type") == "session_meta" and isinstance(payload, dict)


def _parent_id(payload: dict) -> str:
    if parent := _string(payload.get("parent_thread_id") or payload.get("forked_from_id")):
        return parent
    source = payload.get("source")
    if not isinstance(source, dict):
        return ""
    subagent = source.get("subagent")
    if not isinstance(subagent, dict):
        return ""
    spawn = subagent.get("thread_spawn")
    return _string(spawn.get("parent_thread_id")) if isinstance(spawn, dict) else ""


def _message_text(content, role) -> str:
    wanted = "input_text" if role == "user" else "output_text"
    if not isinstance(content, list):
        return ""
    return "\n".join(
        text
        for block in content
        if isinstance(block, dict)
        and block.get("type") in {wanted, "text"}
        and (text := _string(block.get("text"))).strip()
    )


def _tool_use(payload: dict, timestamp: datetime, sidechain: bool) -> Event:
    raw_name = _string(payload.get("name"))
    tool = TOOL_NAMES.get(raw_name, raw_name)
    raw_arguments = payload.get("arguments")
    arguments = _json_object(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
    tool_input = dict(arguments) if isinstance(arguments, dict) else {}
    if tool == "Bash" and "command" not in tool_input and isinstance(tool_input.get("cmd"), str):
        tool_input["command"] = tool_input["cmd"]
    custom_input = _string(payload.get("input"))
    if tool == "Bash" and custom_input and "command" not in tool_input:
        tool_input["command"] = custom_input
    if tool == "Edit" and custom_input and "file_path" not in tool_input:
        paths = PATCH_PATH.findall(custom_input)
        if len(paths) == 1:
            tool_input["file_path"] = paths[0]
    return Event(
        kind=Kind.TOOL_USE,
        timestamp=timestamp,
        tool=tool,
        tool_id=_string(payload.get("call_id") or payload.get("id")),
        tool_input=tool_input,
        sidechain=sidechain,
    )


def _tool_result(payload: dict, timestamp: datetime, sidechain: bool) -> Event:
    text, is_error = _output(payload.get("output"))
    return Event(
        kind=Kind.TOOL_RESULT,
        timestamp=timestamp,
        text=text[:500],
        tool_id=_string(payload.get("call_id") or payload.get("id")),
        is_error=is_error,
        sidechain=sidechain,
    )


def _output(output) -> tuple[str, bool]:
    texts = []
    if isinstance(output, str):
        texts.append(output)
    elif isinstance(output, list):
        texts.extend(
            text
            for block in output
            if isinstance(block, dict) and (text := _string(block.get("text")))
        )
    is_error = False
    for text in texts:
        data = _json_object(text)
        if not data:
            continue
        exit_code = data.get("exit_code")
        is_error |= data.get("isError") is True or data.get("is_error") is True
        is_error |= data.get("ok") is False
        is_error |= isinstance(exit_code, int) and exit_code != 0
    return "\n".join(texts), is_error


def _usage(
    info: dict, previous: dict[str, int], session_model: str, timestamp: datetime
) -> tuple[Usage | None, dict[str, int], int | None]:
    total = info.get("total_token_usage")
    last = info.get("last_token_usage")
    current = _token_counts(total)
    cumulative = current.get("total_tokens") if isinstance(total, dict) else None
    if isinstance(last, dict):
        counts = _token_counts(last)
    elif current:
        counts = {key: max(0, value - previous.get(key, 0)) for key, value in current.items()}
    else:
        counts = {}
    if not counts or not any(counts.values()):
        return None, current or previous, cumulative

    input_tokens = counts.get("input_tokens", 0)
    cached = min(input_tokens, counts.get("cached_input_tokens", 0))
    uncached = max(0, input_tokens - cached)
    cache_write = min(uncached, counts.get("cache_write_input_tokens", 0))
    model = _string(info.get("model") or info.get("model_name")) or session_model or "gpt-5"
    return (
        Usage(
            model=model,
            timestamp=timestamp,
            input_tokens=uncached - cache_write,
            output_tokens=counts.get("output_tokens", 0),
            cache_write_tokens=cache_write,
            cache_read_tokens=cached,
            cache_write_requires_explicit_price=True,
        ),
        current or previous,
        cumulative,
    )


def _usage_key(info: dict, namespace: str) -> tuple | None:
    """Identity copied cumulative snapshots across parent and forked rollout files."""
    total = info.get("total_token_usage")
    if not isinstance(total, dict) or not isinstance(total.get("total_tokens"), int):
        return None
    return (
        namespace,
        total["total_tokens"],
        total.get("input_tokens", 0),
        total.get("cached_input_tokens", 0),
        total.get("output_tokens", 0),
        total.get("reasoning_output_tokens", 0),
    )


def _token_counts(value) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    keys = {
        "input_tokens",
        "cached_input_tokens",
        "cache_write_input_tokens",
        "output_tokens",
        "reasoning_output_tokens",
        "total_tokens",
    }
    return {key: max(0, raw) for key in keys if isinstance((raw := value.get(key)), int)}


def _json_object(value: str) -> dict | None:
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _timestamp(value) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _string(value) -> str:
    return value if isinstance(value, str) else ""


def _root(root: Path | None) -> Path:
    return root or Path(os.environ.get("CODEX_HOME", ROOT))
