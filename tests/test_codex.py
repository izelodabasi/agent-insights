import json
import sqlite3
from pathlib import Path

import pytest

from agent_insights.models import Kind
from agent_insights.sources import codex
from agent_insights.tagging import rules


def _record(ts: str, record_type: str, payload: dict) -> dict:
    return {"timestamp": ts, "type": record_type, "payload": payload}


def _write(root: Path, name: str, records: list[dict], archived: bool = False) -> Path:
    folder = root / "archived_sessions" if archived else root / "sessions/2026/09/23"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"rollout-{name}.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records) + "\nnot json\n")
    return path


def test_load_codex_session(tmp_path: Path):
    usage = {
        "input_tokens": 100,
        "cached_input_tokens": 60,
        "cache_write_input_tokens": 10,
        "output_tokens": 20,
        "reasoning_output_tokens": 5,
        "total_tokens": 120,
    }
    records = [
        _record(
            "2026-09-23T10:00:00Z",
            "session_meta",
            {"id": "s1", "cwd": "/repos/demo", "git": {"branch": "main"}, "source": "cli"},
        ),
        _record(
            "2026-09-23T10:00:01Z",
            "turn_context",
            {"model": "gpt-5.6-sol", "effort": "high"},
        ),
        _record(
            "2026-09-23T10:00:02Z",
            "response_item",
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "run the tests"}],
            },
        ),
        _record(
            "2026-09-23T10:00:03Z",
            "response_item",
            {
                "type": "function_call",
                "name": "exec_command",
                "call_id": "c1",
                "arguments": json.dumps({"cmd": "uv run pytest"}),
            },
        ),
        _record(
            "2026-09-23T10:00:04Z",
            "response_item",
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": json.dumps({"exit_code": 0, "output": "passed"}),
            },
        ),
        _record(
            "2026-09-23T10:00:05Z",
            "event_msg",
            {
                "type": "token_count",
                "info": {"last_token_usage": usage, "total_token_usage": usage},
            },
        ),
        _record(
            "2026-09-23T10:00:06Z",
            "event_msg",
            {
                "type": "token_count",
                "info": {"last_token_usage": usage, "total_token_usage": usage},
            },
        ),
    ]
    _write(tmp_path, "s1", records)

    [session] = codex.load_sessions(tmp_path)
    assert session.agent == "codex"
    assert session.project == "demo"
    assert session.branches == {"main"}
    assert [event.text for event in session.of(Kind.PROMPT)] == ["run the tests"]
    [tool] = session.of(Kind.TOOL_USE)
    assert (tool.tool, tool.tool_input["cmd"]) == ("Bash", "uv run pytest")
    [found] = session.usage.values()
    assert (found.input_tokens, found.cache_read_tokens, found.cache_write_tokens) == (30, 60, 10)
    assert found.output_tokens == 20
    assert found.reasoning_tokens == 5
    assert session.reasoning_efforts == {"high"}
    assert found.cache_write_requires_explicit_price
    rules.tag_session(session)
    assert "testing" in session.tags


def test_cumulative_usage_is_converted_to_deltas(tmp_path: Path):
    first = {
        "input_tokens": 100,
        "cached_input_tokens": 20,
        "output_tokens": 10,
        "total_tokens": 110,
    }
    second = {
        "input_tokens": 250,
        "cached_input_tokens": 70,
        "output_tokens": 30,
        "total_tokens": 280,
    }
    records = [
        _record("2026-09-23T10:00:00Z", "session_meta", {"id": "s1", "cwd": "/repos/demo"}),
        _record("2026-09-23T10:00:01Z", "turn_context", {"model": "gpt-5"}),
        _record(
            "2026-09-23T10:00:02Z",
            "response_item",
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "go"}]},
        ),
        _record(
            "2026-09-23T10:00:03Z",
            "event_msg",
            {"type": "token_count", "info": {"total_token_usage": first}},
        ),
        _record(
            "2026-09-23T10:00:04Z",
            "event_msg",
            {"type": "token_count", "info": {"total_token_usage": second}},
        ),
    ]
    _write(tmp_path, "s1", records)

    [session] = codex.load_sessions(tmp_path)
    assert [
        (u.input_tokens, u.cache_read_tokens, u.output_tokens) for u in session.usage.values()
    ] == [
        (80, 20, 10),
        (100, 50, 20),
    ]


def test_subagent_replay_is_skipped_and_new_work_is_merged(tmp_path: Path):
    root = [
        _record("2026-09-23T10:00:00Z", "session_meta", {"id": "parent", "cwd": "/repos/demo"}),
        _record(
            "2026-09-23T10:00:01Z",
            "response_item",
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "delegate"}],
            },
        ),
    ]
    child = [
        _record(
            "2026-09-23T10:01:00Z",
            "session_meta",
            {
                "id": "child",
                "cwd": "/repos/demo",
                "parent_thread_id": "parent",
                "source": {"subagent": {}},
            },
        ),
        _record(
            "2026-09-23T10:01:01Z",
            "response_item",
            {
                "type": "custom_tool_call",
                "name": "apply_patch",
                "call_id": "old",
                "input": "*** Update File: old.py",
            },
        ),
        _record(
            "2026-09-23T10:01:06Z",
            "response_item",
            {
                "type": "custom_tool_call",
                "name": "apply_patch",
                "call_id": "new",
                "input": "*** Update File: new.py",
            },
        ),
    ]
    _write(tmp_path, "parent", root)
    _write(tmp_path, "child", child)

    [session] = codex.load_sessions(tmp_path)
    assert session.session_id == "parent"
    assert [event.tool_id for event in session.of(Kind.TOOL_USE)] == ["new"]
    [edit] = session.of(Kind.TOOL_USE)
    assert (edit.tool, edit.tool_input["file_path"], edit.sidechain) == ("Edit", "new.py", True)


def test_fork_usage_replayed_after_cutoff_is_deduplicated(tmp_path: Path):
    def usage(last: int, total: int) -> dict:
        return {
            "last_token_usage": {"input_tokens": last, "total_tokens": last},
            "total_token_usage": {"input_tokens": total, "total_tokens": total},
        }

    parent = [
        _record("2026-09-23T10:00:00Z", "session_meta", {"id": "parent", "cwd": "/demo"}),
        _record(
            "2026-09-23T10:00:01Z",
            "response_item",
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "go"}]},
        ),
        _record(
            "2026-09-23T10:00:02Z", "event_msg", {"type": "token_count", "info": usage(700, 700)}
        ),
        _record(
            "2026-09-23T10:00:03Z", "event_msg", {"type": "token_count", "info": usage(400, 1100)}
        ),
    ]
    child = [
        _record(
            "2026-09-23T10:00:00Z",
            "session_meta",
            {"id": "child", "cwd": "/demo", "forked_from_id": "parent"},
        ),
        _record(
            "2026-09-23T10:00:10Z", "event_msg", {"type": "token_count", "info": usage(700, 700)}
        ),
        _record(
            "2026-09-23T10:00:11Z", "event_msg", {"type": "token_count", "info": usage(400, 1100)}
        ),
        _record(
            "2026-09-23T10:00:12Z", "event_msg", {"type": "token_count", "info": usage(400, 1500)}
        ),
    ]
    _write(tmp_path, "1-parent", parent)
    _write(tmp_path, "2-child", child)

    [session] = codex.load_sessions(tmp_path)
    assert (
        sum(
            u.input_tokens + u.cache_read_tokens + u.cache_write_tokens + u.output_tokens
            for u in session.usage.values()
        )
        == 1500
    )


def test_rejects_file_without_session_meta_first(tmp_path: Path):
    records = [
        _record(
            "2026-09-23T10:00:00Z",
            "response_item",
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "go"}]},
        )
    ]
    _write(tmp_path, "foreign", records)
    assert codex.load_sessions(tmp_path) == []


def test_loads_app_thread_name_with_generated_title_fallback(tmp_path: Path):
    records = [
        _record("2026-09-23T10:00:00Z", "session_meta", {"id": "named", "cwd": "/demo"}),
        _record(
            "2026-09-23T10:00:01Z",
            "response_item",
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "go"}]},
        ),
    ]
    _write(tmp_path, "named", records)
    database = sqlite3.connect(tmp_path / "state_5.sqlite")
    database.execute("CREATE TABLE threads (id TEXT, title TEXT, name TEXT)")
    database.execute(
        "INSERT INTO threads VALUES (?, ?, ?)",
        ("named", "Generated title", "Renamed chat"),
    )
    database.commit()
    database.close()

    [session] = codex.load_sessions(tmp_path)
    assert session.title == "Renamed chat"

    database = sqlite3.connect(tmp_path / "state_5.sqlite")
    database.execute("UPDATE threads SET name = NULL WHERE id = 'named'")
    database.commit()
    database.close()
    [session] = codex.load_sessions(tmp_path)
    assert session.title == "Generated title"


def test_archived_copy_is_not_loaded_twice(tmp_path: Path):
    records = [
        _record("2026-09-23T10:00:00Z", "session_meta", {"id": "s1", "cwd": "/repos/demo"}),
        _record(
            "2026-09-23T10:00:01Z",
            "response_item",
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "go"}]},
        ),
    ]
    _write(tmp_path, "same", records)
    _write(tmp_path, "same", records, archived=True)
    assert len(codex.load_sessions(tmp_path)) == 1


def test_installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    assert not codex.installed(tmp_path)
    (tmp_path / "sessions").mkdir()
    assert codex.installed(tmp_path)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    assert codex.installed()


@pytest.mark.parametrize("exit_code", [1, 2])
def test_failed_command_result(exit_code: int, tmp_path: Path):
    records = [
        _record("2026-09-23T10:00:00Z", "session_meta", {"id": "s1", "cwd": "/repos/demo"}),
        _record(
            "2026-09-23T10:00:01Z",
            "response_item",
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": json.dumps({"exit_code": exit_code}),
            },
        ),
    ]
    _write(tmp_path, "s1", records)
    [session] = codex.load_sessions(tmp_path)
    assert session.of(Kind.TOOL_RESULT)[0].is_error
