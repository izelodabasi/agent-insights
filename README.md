# agent-insights

`agent-insights` is a personal dashboard for understanding coding-agent usage, including token
consumption, effective sessions, and workflow challenges.

It reads the chat logs already stored on your machine and turns them into one self-contained HTML report. You can explore activity and cost by project or model, see which tools were used, compare session performance, and review moments where you corrected, interrupted, approved, or disliked an answer.

Nothing is uploaded. Your prompts, code, and project names stay on your computer.

## Supported agents

- Claude Code from `~/.claude/projects`
- Codex from `$CODEX_HOME/sessions`, or `~/.codex/sessions` by default
- GitHub Copilot from its CLI journals and VS Code chat history

Codex subagent work is folded into its parent session, and replayed history is filtered out so it does not inflate token or tool counts. Copilot reads both workspace chats and chats that were opened without a workspace.

## Getting started

You need Python 3.12 or newer and [`uv`](https://docs.astral.sh/uv/).

```sh
uv sync --extra nli
uv run agent-insights scan
```

The report is written to `./report.html`. Open it in any browser.

The first full scan downloads two local language models and may take a little while. Their results are cached, so later scans only classify new messages. If you want a quick report without the model download, use the rule-based scan:

```sh
uv run agent-insights scan --no-nli
```

To write the report somewhere else:

```sh
uv run agent-insights scan --out path/to/report.html
```

Filter activity by an inclusive local-date range:

```sh
uv run agent-insights scan --since 2026-09-01 --until 2026-09-30
```

`--since` and `--until` can be used independently. The report header displays the active range.

Use a rolling range:

```sh
uv run agent-insights scan --days 30
```

The available presets are 7, 30, and 90 days. `--days` can be combined with `--until` to end
the rolling window on a historical date.

## What the report shows

- Sessions, prompts, active time, token usage, and estimated API cost
- Breakdowns by project, model, day, hour, and tool
- Successful commits and edited-file counts
- Failed or rejected tool calls
- A performance score for each session and project
- Your messages, searchable and grouped by interaction tags

The generated report includes excerpts from your prompts. It stays local, but you should still treat the HTML file as personal data if you share or archive it.

## How messages are tagged

Some tags come directly from things that happened in the session:

- `rejected`: you denied a tool call
- `interrupted`: you stopped a running request
- `committed`: the turn produced a successful Git commit
- `low_performance`: the agent had at least three failed tool calls or edited the same file at least three times before your next message
- `testing`: the turn ran a test command
- `debugging`: the agent edited a file after a tool call failed

Other tags look at the wording of your message: `disliked`, `correction`, `style`, `approval`, `question`, and `new_task`. Classification runs locally with `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` and `SamLowe/roberta-base-go_emotions`; it does not send your messages to an API. Scores are cached under `~/.cache/agent-insights/`.

## About the score

The score is meant as a useful signal, not a verdict on you or the agent. It starts from the challenges in each turn: corrections, rejected calls, interruptions, repeated failures, and other tags carry different weights. Successful commits help the score, while unusually expensive API calls make a small adjustment in either direction.

The exact calculation lives in `src/agent_insights/stats.py`. Session scores are capped at 100, and project scores are weighted by the number of turns in each session.

## Cost estimates

Prices come from LiteLLM's public model-price table and are cached for 24 hours. The parser handles the different token formats used by Claude Code, Codex, and Copilot, and avoids counting repeated or cumulative usage records twice. Thinking effort and reasoning tokens are shown when the agent recorded exact values; they are never estimated.

These figures use published API prices. They are useful for comparison, but they may not match what you pay through a subscription plan.
