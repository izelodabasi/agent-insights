import json
import os
from pathlib import Path

import pytest

from agent_insights.models import Kind
from agent_insights.sources import claude_code
from agent_insights.tagging import rules

BASE = {"sessionId": "s1", "cwd": "/repos/demo", "gitBranch": "main"}


def _user(ts: str, content, **extra) -> dict:
    return {
        **BASE,
        "type": "user",
        "timestamp": ts,
        "message": {"role": "user", "content": content},
        **extra,
    }


def _assistant(ts: str, msg_id: str, content: list, usage: dict | None = None) -> dict:
    return {
        **BASE,
        "type": "assistant",
        "timestamp": ts,
        "message": {
            "id": msg_id,
            "model": "claude-opus-5",
            "role": "assistant",
            "content": content,
            "usage": usage or {"input_tokens": 10, "output_tokens": 5},
        },
    }


@pytest.fixture
def root(tmp_path: Path) -> Path:
    records = [
        _user("2026-09-01T10:00:00Z", "fix the parser"),
        _assistant("2026-09-01T10:00:05Z", "m1", [{"type": "text", "text": "On it"}]),
        _assistant(
            "2026-09-01T10:00:06Z",
            "m1",
            [{"type": "tool_use", "id": "t1", "name": "Edit", "input": {"file_path": "a.py"}}],
        ),
        _user(
            "2026-09-01T10:00:07Z",
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": "The user doesn't want to proceed with this tool use. "
                    "The tool use was rejected. To tell you how to proceed, "
                    "the user said:\nno comments\n\n"
                    "Note: pay attention",
                }
            ],
        ),
        _user("2026-09-01T10:01:00Z", "[Request interrupted by user]"),
        _user("2026-09-01T10:02:00Z", "<command-name>/clear</command-name>"),
        _user("2026-09-01T10:02:01Z", "caveat", isMeta=True),
        _user("2026-09-01T10:03:00Z", "okay go"),
        _assistant(
            "2026-09-01T10:03:05Z",
            "m2",
            [
                {
                    "type": "tool_use",
                    "id": "t2",
                    "name": "Bash",
                    "input": {"command": "git commit -m x"},
                }
            ],
        ),
        _user(
            "2026-09-01T10:03:06Z",
            [{"type": "tool_result", "tool_use_id": "t2", "content": "1 file changed"}],
        ),
        {"type": "mode", "sessionId": "s1"},
        {"type": "ai-title", "sessionId": "s1", "aiTitle": "Fix parser"},
        {"type": "custom-title", "sessionId": "s1", "customTitle": "parser fix"},
    ]
    project = tmp_path / "-repos-demo"
    project.mkdir()
    (project / "s1.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\nnot json\n")
    sub = project / "s1" / "subagents"
    sub.mkdir(parents=True)
    (sub / "agent-a.jsonl").write_text(
        json.dumps(
            {
                **_user("2026-09-01T10:00:30Z", "subagent brief"),
                "isSidechain": True,
            }
        )
        + "\n"
        + json.dumps(_assistant("2026-09-01T10:00:40Z", "m3", []))
    )
    return tmp_path


def test_load_sessions(root: Path):
    [s] = claude_code.load_sessions(root)
    assert s.project == "demo"
    assert s.branches == {"main"}
    assert (s.title, s.ai_title) == ("parser fix", "Fix parser")
    assert [e.text for e in s.of(Kind.PROMPT)] == ["fix the parser", "okay go"]
    assert [e.text for e in s.of(Kind.REJECTION)] == ["no comments"]
    assert len(s.of(Kind.INTERRUPT)) == 1
    assert [e.text for e in s.of(Kind.COMMAND)] == ["clear"]
    assert set(s.usage) == {"m1", "m2", "m3"}


def test_continued_session_keeps_copied_records_once(tmp_path: Path):
    project = tmp_path / "-repos-demo"
    project.mkdir()
    reply = _assistant("2026-09-01T10:00:05Z", "m1", [{"type": "text", "text": "ok"}])
    first = [
        {**_user("2026-09-01T10:00:00Z", "start the tool"), "uuid": "u1"},
        {**reply, "uuid": "a1"},
    ]
    copied = [{**r, "sessionId": "s2"} for r in first]
    later = [{**_user("2026-09-02T09:00:00Z", "continue"), "sessionId": "s2", "uuid": "u2"}]
    (project / "s1.jsonl").write_text("\n".join(json.dumps(r) for r in first))
    (project / "s2.jsonl").write_text("\n".join(json.dumps(r) for r in copied + later))
    os.utime(project / "s1.jsonl", (1, 1))

    sessions = {s.session_id: s for s in claude_code.load_sessions(tmp_path)}
    assert [e.text for e in sessions["s1"].of(Kind.PROMPT)] == ["start the tool"]
    assert [e.text for e in sessions["s2"].of(Kind.PROMPT)] == ["continue"]
    assert set(sessions["s1"].usage) == {"m1"}
    assert sessions["s2"].usage == {}


@pytest.mark.parametrize("field", ["reasoning_tokens", "thinking_tokens"])
def test_exact_thinking_tokens_are_loaded_when_present(tmp_path: Path, field: str):
    project = tmp_path / "-repos-demo"
    project.mkdir()
    records = [
        _user("2026-09-01T10:00:00Z", "think carefully"),
        _assistant(
            "2026-09-01T10:00:05Z",
            "m1",
            [{"type": "text", "text": "Done"}],
            {"input_tokens": 10, "output_tokens": 8, field: 3},
        ),
    ]
    (project / "s1.jsonl").write_text("\n".join(json.dumps(record) for record in records))

    [session] = claude_code.load_sessions(tmp_path)
    assert session.usage["m1"].reasoning_tokens == 3


def test_rule_tags(root: Path):
    [s] = claude_code.load_sessions(root)
    rules.tag_session(s)
    assert s.tags == {"rejected", "interrupted", "committed"}
    [go] = [e for e in s.of(Kind.PROMPT) if e.text == "okay go"]
    assert "committed" in go.tags
    assert [e.tool_id for e in rules.successful_commits(s)] == ["t2"]
