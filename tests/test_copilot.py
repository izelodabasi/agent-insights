import json
from pathlib import Path

from agent_insights.models import Kind
from agent_insights.sources import copilot


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(record) for record in records) + "\nnot json\n")


def test_loads_vscode_chat_session_journal(tmp_path: Path):
    workspace = tmp_path / "workspaceStorage"
    folder = workspace / "hash"
    (folder / "workspace.json").parent.mkdir(parents=True)
    (folder / "workspace.json").write_text(json.dumps({"folder": "file:///repos/my-app"}))
    request = {
        "requestId": "request-1",
        "timestamp": 1_780_157_113_020,
        "modelId": "copilot/auto",
        "message": {"text": "run the tests"},
        "response": [
            {"kind": "thinking", "value": "private reasoning"},
            {"value": "Everything passes."},
        ],
        "completionTokens": 99,
        "result": {
            "metadata": {
                "promptTokens": 32543,
                "outputTokens": 60,
                "resolvedModel": "claude-sonnet-4-6",
                "toolCallRounds": [
                    {
                        "thinking": {"tokens": 17},
                        "toolCalls": [
                            {
                                "id": "call-1",
                                "name": "run_in_terminal",
                                "arguments": json.dumps({"command": "uv run pytest"}),
                            }
                        ],
                    }
                ],
            }
        },
    }
    _write_jsonl(
        folder / "chatSessions/session.jsonl",
        [
            {
                "kind": 0,
                "v": {
                    "version": 3,
                    "creationDate": 1_780_157_113_000,
                    "sessionId": "session-1",
                    "customTitle": "Test the parser",
                    "requests": [],
                },
            },
            {"kind": 2, "k": ["requests"], "v": [request]},
            {"kind": 1, "k": ["requests", 0, "result", "metadata", "outputTokens"], "v": 88},
        ],
    )

    [session] = copilot.load_sessions(tmp_path / "copilot", [workspace], [])
    assert (session.agent, session.session_id, session.project) == (
        "copilot",
        "session-1",
        "my-app",
    )
    assert session.title == "Test the parser"
    assert [event.text for event in session.of(Kind.PROMPT)] == ["run the tests"]
    assert [event.text for event in session.of(Kind.ASSISTANT)] == ["Everything passes."]
    [tool] = session.of(Kind.TOOL_USE)
    assert (tool.tool, tool.tool_input["command"]) == ("Bash", "uv run pytest")
    [usage] = session.usage.values()
    assert (usage.model, usage.input_tokens, usage.output_tokens) == (
        "claude-sonnet-4-6",
        32543,
        88,
    )
    assert usage.reasoning_tokens == 17


def test_loads_empty_window_chat(tmp_path: Path):
    global_storage = tmp_path / "globalStorage"
    _write_jsonl(
        global_storage / "emptyWindowChatSessions/session.jsonl",
        [
            {
                "kind": 0,
                "v": {
                    "sessionId": "empty-1",
                    "creationDate": 1_780_157_113_000,
                    "requests": [
                        {
                            "requestId": "r1",
                            "timestamp": 1_780_157_113_020,
                            "modelId": "copilot/gpt-4.1",
                            "message": {"text": "hello"},
                            "response": [{"generatedTitle": "Quick question"}],
                            "completionTokens": 12,
                        }
                    ],
                },
            }
        ],
    )

    [session] = copilot.load_sessions(tmp_path / "copilot", [], [global_storage])
    assert session.project == "copilot-chat"
    assert session.title == "Quick question"
    assert session.usage["r1"].model == "gpt-4.1"


def test_loads_cli_events_and_shutdown_usage(tmp_path: Path):
    session_dir = tmp_path / "session-state/s1"
    session_dir.mkdir(parents=True)
    (session_dir / "workspace.yaml").write_text("cwd: /repos/demo\n")
    _write_jsonl(
        session_dir / "events.jsonl",
        [
            {
                "type": "session.model_change",
                "timestamp": "2026-09-23T10:00:00Z",
                "data": {"newModel": "gpt-4.1"},
            },
            {
                "type": "user.message",
                "timestamp": "2026-09-23T10:00:01Z",
                "data": {"content": "inspect this"},
            },
            {
                "type": "assistant.message",
                "timestamp": "2026-09-23T10:00:02Z",
                "data": {
                    "messageId": "m1",
                    "content": "Done.",
                    "outputTokens": 20,
                    "toolRequests": [
                        {
                            "toolCallId": "call-1",
                            "name": "read_file",
                            "arguments": {"path": "app.py"},
                        }
                    ],
                },
            },
            {
                "type": "session.shutdown",
                "timestamp": "2026-09-23T10:01:00Z",
                "data": {
                    "modelMetrics": {
                        "gpt-4.1": {
                            "usage": {
                                "inputTokens": 100,
                                "outputTokens": 20,
                                "cacheReadTokens": 60,
                                "cacheWriteTokens": 10,
                                "reasoningTokens": 5,
                            }
                        }
                    }
                },
            },
        ],
    )

    [session] = copilot.load_sessions(tmp_path, [], [])
    assert session.project == "demo"
    assert [event.text for event in session.of(Kind.PROMPT)] == ["inspect this"]
    assert session.of(Kind.TOOL_USE)[0].tool == "Read"
    assert [
        (usage.input_tokens, usage.output_tokens, usage.cache_read_tokens, usage.cache_write_tokens)
        for usage in session.usage.values()
    ] == [(0, 20, 0, 0), (30, 0, 60, 10)]
    assert [usage.reasoning_tokens for usage in session.usage.values()] == [0, 5]


def test_installed_detects_vscode_workspace(monkeypatch, tmp_path: Path):
    workspace = tmp_path / "workspaceStorage"
    _write_jsonl(workspace / "hash/chatSessions/session.jsonl", [{"kind": 0, "v": {}}])
    monkeypatch.setattr(copilot, "ROOT", tmp_path / "missing")
    monkeypatch.setattr(copilot, "_workspace_roots", lambda: [workspace])
    monkeypatch.setattr(copilot, "_global_roots", lambda: [])
    assert copilot.installed()
