import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

from agent_insights.models import Event, Kind, Session, Usage
from agent_insights.repos import repo_name

ROOT = Path.home() / ".copilot"
FORBIDDEN_JOURNAL_KEYS = {"__proto__", "prototype", "constructor"}
TOOL_NAMES = {
    "bash": "Bash",
    "create_file": "Write",
    "edit_file": "Edit",
    "list_dir": "Glob",
    "list_directory": "Glob",
    "read_file": "Read",
    "replace_string_in_file": "Edit",
    "run_in_terminal": "Bash",
    "runcommand": "Bash",
    "search": "Grep",
    "search_files": "Grep",
}


def installed(root: Path | None = None) -> bool:
    if root is not None:
        return (root / "session-state").is_dir() or (root / "session-store.db").is_file()
    return (
        installed(ROOT)
        or any(next(path.glob("*/chatSessions/*.jsonl"), None) for path in _workspace_roots())
        or any(_has_jsonl(path / "emptyWindowChatSessions") for path in _global_roots())
    )


def load_sessions(
    root: Path | None = None,
    workspace_roots: list[Path] | None = None,
    global_roots: list[Path] | None = None,
) -> list[Session]:
    copilot_root = root or ROOT
    workspaces = _workspace_roots() if workspace_roots is None else workspace_roots
    globals_ = _global_roots() if global_roots is None else global_roots
    sessions: list[Session] = []

    for path in sorted((copilot_root / "session-state").glob("*/events.jsonl")):
        if session := _read_event_journal(path, _cli_cwd(path), transcript=False):
            sessions.append(session)

    for storage in workspaces:
        for chat_dir in sorted(storage.glob("*/chatSessions")):
            cwd = _workspace_cwd(chat_dir.parent / "workspace.json")
            sessions.extend(
                session
                for path in sorted(chat_dir.glob("*.jsonl"))
                if (session := _read_chat_session(path, cwd)) is not None
            )
        for transcript_dir in sorted(storage.glob("*/GitHub.copilot-chat/transcripts")):
            if _has_jsonl(transcript_dir.parent.parent / "chatSessions"):
                continue
            cwd = _workspace_cwd(transcript_dir.parent.parent / "workspace.json")
            sessions.extend(
                session
                for path in sorted(transcript_dir.glob("*.jsonl"))
                if (session := _read_event_journal(path, cwd, transcript=True)) is not None
            )

    for storage in globals_:
        sessions.extend(
            session
            for path in sorted((storage / "emptyWindowChatSessions").glob("*.jsonl"))
            if (session := _read_chat_session(path, "")) is not None
        )

    return sessions


def _read_chat_session(path: Path, cwd: str) -> Session | None:
    root = _replay_journal(path)
    if not isinstance(root, dict):
        return None
    session_id = _string(root.get("sessionId")) or path.stem
    project = repo_name(cwd) or (Path(cwd).name if cwd else "copilot-chat")
    session = Session(
        agent="copilot",
        session_id=session_id,
        project=project,
        cwd=cwd,
        title=_chat_title(root),
    )
    created = _timestamp(root.get("creationDate"))

    for index, request in enumerate(root.get("requests") or []):
        if not isinstance(request, dict):
            continue
        timestamp = _timestamp(request.get("timestamp")) or created
        if timestamp is None:
            continue
        prompt = _nested_string(request, "message", "text")
        if prompt:
            session.events.append(Event(kind=Kind.PROMPT, timestamp=timestamp, text=prompt))

        result = request.get("result")
        metadata = result.get("metadata", {}) if isinstance(result, dict) else {}
        if not isinstance(metadata, dict):
            metadata = {}
        for tool_id, name, arguments in _chat_tools(metadata):
            session.events.append(
                Event(
                    kind=Kind.TOOL_USE,
                    timestamp=timestamp,
                    tool=_tool_name(name),
                    tool_id=tool_id,
                    tool_input=arguments,
                )
            )
            session.events.append(
                Event(kind=Kind.TOOL_RESULT, timestamp=timestamp, tool_id=tool_id)
            )

        response = _response_text(request.get("response"))
        if response:
            session.events.append(Event(kind=Kind.ASSISTANT, timestamp=timestamp, text=response))

        input_tokens = _positive_int(metadata.get("promptTokens"))
        output_tokens = _positive_int(metadata.get("outputTokens")) or _positive_int(
            request.get("completionTokens")
        )
        reasoning_tokens = _thinking_tokens(metadata)
        if input_tokens or output_tokens or reasoning_tokens:
            model = _string(metadata.get("resolvedModel"))
            if not model:
                model = _string(request.get("modelId")).removeprefix("copilot/") or "unknown"
            key = _string(request.get("requestId")) or f"request-{index}"
            session.usage[key] = Usage(
                model=model,
                timestamp=timestamp,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                reasoning_tokens=reasoning_tokens,
            )

    session.events.sort(key=lambda event: event.timestamp)
    return session if session.events else None


def _read_event_journal(path: Path, cwd: str, transcript: bool) -> Session | None:
    records = list(_records(path))
    session_id = path.stem if transcript else path.parent.name
    title = ""
    for record in records:
        if record.get("type") == "session.start":
            session_id = _nested_string(record, "data", "sessionId") or session_id
            title = _nested_string(record, "data", "title") or _nested_string(
                record, "data", "customTitle"
            )
            break
    session = Session(
        agent="copilot",
        session_id=session_id,
        project=repo_name(cwd) or (Path(cwd).name if cwd else "copilot"),
        cwd=cwd,
        title=title,
    )
    model = ""
    previous_rollup: dict[str, dict[str, int]] = {}
    shutdown_count: dict[str, int] = {}

    for record in records:
        event_type = record.get("type")
        data = record.get("data")
        if not isinstance(data, dict):
            data = {}
        timestamp = _timestamp(record.get("timestamp"))
        if event_type == "session.start":
            model = _string(data.get("selectedModel")) or model
        elif event_type == "session.model_change":
            model = _string(data.get("newModel")) or model
        elif event_type == "user.message" and timestamp:
            prompt = _string(data.get("content"))
            if prompt:
                session.events.append(Event(kind=Kind.PROMPT, timestamp=timestamp, text=prompt))
        elif event_type == "assistant.message" and timestamp:
            model = _string(data.get("model")) or model
            content = _string(data.get("content"))
            if content:
                session.events.append(Event(kind=Kind.ASSISTANT, timestamp=timestamp, text=content))
            for tool_id, name, arguments in _event_tools(data.get("toolRequests")):
                session.events.append(
                    Event(
                        kind=Kind.TOOL_USE,
                        timestamp=timestamp,
                        tool=_tool_name(name),
                        tool_id=tool_id,
                        tool_input=arguments,
                    )
                )
            output = _positive_int(data.get("outputTokens"))
            if output:
                key = _string(data.get("messageId")) or f"message-{len(session.usage)}"
                session.usage[key] = Usage(
                    model=model or "unknown", timestamp=timestamp, output_tokens=output
                )
        elif event_type == "tool.execution_complete" and timestamp:
            tool_id = _string(data.get("toolCallId") or data.get("id"))
            session.events.append(
                Event(
                    kind=Kind.TOOL_RESULT,
                    timestamp=timestamp,
                    tool_id=tool_id,
                    is_error=data.get("success") is False or bool(data.get("error")),
                )
            )
        elif event_type == "session.shutdown" and not transcript:
            metrics = data.get("modelMetrics")
            if not isinstance(metrics, dict):
                continue
            shutdown_ts = timestamp or _timestamp(data.get("sessionStartTime"))
            for model_name, raw in metrics.items():
                usage = raw.get("usage") if isinstance(raw, dict) else None
                if not isinstance(usage, dict) or shutdown_ts is None:
                    continue
                current = {
                    name: _positive_int(usage.get(name))
                    for name in (
                        "inputTokens",
                        "cacheReadTokens",
                        "cacheWriteTokens",
                        "reasoningTokens",
                    )
                }
                previous = previous_rollup.get(model_name, {})
                if current["inputTokens"] < previous.get("inputTokens", 0):
                    previous = {}
                delta = {
                    name: max(0, value - previous.get(name, 0)) for name, value in current.items()
                }
                previous_rollup[model_name] = current
                cache_read = delta["cacheReadTokens"]
                cache_write = delta["cacheWriteTokens"]
                reasoning = delta["reasoningTokens"]
                uncached = max(0, delta["inputTokens"] - cache_read - cache_write)
                if not (uncached or cache_read or cache_write or reasoning):
                    continue
                count = shutdown_count.get(model_name, 0) + 1
                shutdown_count[model_name] = count
                session.usage[f"shutdown:{model_name}:{count}"] = Usage(
                    model=model_name,
                    timestamp=shutdown_ts,
                    input_tokens=uncached,
                    reasoning_tokens=reasoning,
                    cache_read_tokens=cache_read,
                    cache_write_tokens=cache_write,
                    cache_write_requires_explicit_price=True,
                )

    session.events.sort(key=lambda event: event.timestamp)
    return session if session.events else None


def _replay_journal(path: Path):
    root = {}
    for entry in _records(path):
        kind = entry.get("kind")
        if kind == 0:
            root = entry.get("v") if isinstance(entry.get("v"), (dict, list)) else {}
            continue
        raw_path = entry.get("k", ["requests"] if kind == 2 else None)
        if not isinstance(raw_path, list) or any(
            not isinstance(part, (str, int))
            or isinstance(part, bool)
            or part in FORBIDDEN_JOURNAL_KEYS
            for part in raw_path
        ):
            continue
        if kind == 1:
            root = _journal_set(root, raw_path, entry.get("v"))
        elif kind == 2 and isinstance(entry.get("v"), list):
            root = _journal_append(root, raw_path, entry["v"])
    return root


def _journal_set(root, path: list, value):
    if not path:
        return value
    if not isinstance(root, (dict, list)):
        root = {}
    parent = _journal_parent(root, path)
    if parent is not None:
        _container_set(parent, path[-1], value)
    return root


def _journal_append(root, path: list, values: list):
    if not path:
        if isinstance(root, list):
            root.extend(values)
        return root
    if not isinstance(root, (dict, list)):
        root = {}
    parent = _journal_parent(root, path)
    if parent is None:
        return root
    target = _container_get(parent, path[-1])
    if not isinstance(target, list):
        target = []
        _container_set(parent, path[-1], target)
    target.extend(values)
    return root


def _journal_parent(root, path: list):
    current = root
    for index, part in enumerate(path[:-1]):
        child = _container_get(current, part)
        if not isinstance(child, (dict, list)):
            child = [] if isinstance(path[index + 1], int) else {}
            if not _container_set(current, part, child):
                return None
        current = child
    return current


def _container_get(container, key):
    if isinstance(container, dict):
        return container.get(key)
    if isinstance(container, list) and isinstance(key, int) and 0 <= key < len(container):
        return container[key]
    return None


def _container_set(container, key, value) -> bool:
    if isinstance(container, dict):
        container[key] = value
        return True
    if isinstance(container, list) and isinstance(key, int) and key >= 0:
        container.extend([None] * (key - len(container) + 1))
        container[key] = value
        return True
    return False


def _chat_tools(metadata: dict) -> list[tuple[str, str, dict]]:
    found = []
    for round_ in metadata.get("toolCallRounds") or []:
        if not isinstance(round_, dict):
            continue
        for raw in round_.get("toolCalls") or []:
            if not isinstance(raw, dict):
                continue
            found.append(
                (
                    _string(raw.get("id")),
                    _string(raw.get("name") or raw.get("toolName") or raw.get("tool")),
                    _arguments(raw.get("arguments") or raw.get("input")),
                )
            )
    return found


def _thinking_tokens(metadata: dict) -> int:
    total = 0
    for round_ in metadata.get("toolCallRounds") or []:
        if not isinstance(round_, dict) or not isinstance(round_.get("thinking"), dict):
            continue
        total += _positive_int(round_["thinking"].get("tokens"))
    return total


def _chat_title(root: dict) -> str:
    if title := _string(root.get("customTitle") or root.get("title")):
        return title
    for request in root.get("requests") or []:
        if not isinstance(request, dict) or not isinstance(request.get("response"), list):
            continue
        for part in request["response"]:
            if isinstance(part, dict) and (title := _string(part.get("generatedTitle"))):
                return title
    return ""


def _event_tools(value) -> list[tuple[str, str, dict]]:
    if not isinstance(value, list):
        return []
    return [
        (
            _string(raw.get("toolCallId") or raw.get("id")),
            _string(raw.get("name") or raw.get("toolName")),
            _arguments(raw.get("arguments") or raw.get("input")),
        )
        for raw in value
        if isinstance(raw, dict)
    ]


def _arguments(value) -> dict:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return dict(value) if isinstance(value, dict) else {}


def _response_text(value) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    texts = []
    for part in value:
        if not isinstance(part, dict) or part.get("kind") in {
            "thinking",
            "toolInvocationSerialized",
            "mcpServersStarting",
        }:
            continue
        text = part.get("value")
        if isinstance(text, str) and text.strip():
            texts.append(text)
    return "\n".join(texts)


def _tool_name(name: str) -> str:
    base = re.sub(r"^copilot_", "", name, flags=re.I)
    return TOOL_NAMES.get(base.lower(), name or "?")


def _records(path: Path):
    try:
        handle = path.open(encoding="utf-8", errors="replace")
    except OSError:
        return
    with handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield value


def _cli_cwd(path: Path) -> str:
    try:
        text = (path.parent / "workspace.yaml").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    match = re.search(r"(?m)^cwd:\s*['\"]?(.+?)['\"]?\s*(?:#.*)?$", text)
    return match.group(1).strip() if match else ""


def _workspace_cwd(path: Path) -> str:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    uri = data.get("folder") if isinstance(data, dict) else None
    if not isinstance(uri, str):
        return ""
    parsed = urlparse(uri)
    return unquote(parsed.path) if parsed.scheme == "file" else ""


def _workspace_roots() -> list[Path]:
    home = Path.home()
    if sys.platform == "darwin":
        base = home / "Library/Application Support"
        return [
            base / name / "User/workspaceStorage"
            for name in ("Code", "Code - Insiders", "VSCodium")
        ]
    if os.name == "nt":
        base = home / "AppData/Roaming"
        return [
            base / name / "User/workspaceStorage"
            for name in ("Code", "Code - Insiders", "VSCodium")
        ]
    base = home / ".config"
    return [
        base / name / "User/workspaceStorage" for name in ("Code", "Code - Insiders", "VSCodium")
    ]


def _global_roots() -> list[Path]:
    return [path.parent / "globalStorage" for path in _workspace_roots()]


def _has_jsonl(path: Path) -> bool:
    return path.is_dir() and next(path.glob("*.jsonl"), None) is not None


def _timestamp(value) -> datetime | None:
    if isinstance(value, (int, float)) and value > 0:
        seconds = value / 1000 if value > 10_000_000_000 else value
        return datetime.fromtimestamp(seconds).astimezone()
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _positive_int(value) -> int:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return int(value)
    return 0


def _string(value) -> str:
    return value if isinstance(value, str) else ""


def _nested_string(value: dict, *keys: str) -> str:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return ""
        current = current.get(key)
    return _string(current)
