# agent-insights

Reads local coding-agent chat logs and writes one HTML report: activity, cost and tokens per
repo and model, tool usage, a performance score per session and project, and your messages
tagged by what happened (rejected, interrupted, frustrated, committed, low_performance, …).

Supported agents: Claude Code (`~/.claude/projects`).

## Usage

    uv sync --extra nli
    uv run agent-insights scan            # writes ./report.html
    uv run agent-insights scan --no-nli   # rules only, no model download

## Tags

From the log itself (`tagging/rules.py`):

- `rejected`: a tool call you denied (the reason you typed is kept as a message)
- `interrupted`: you stopped a running request
- `committed`: your message that led to a `git commit` that succeeded
- `low_performance`: the agent hit 3+ failed tool calls, or edited one file 3+ times, before your next message
- `testing`: the turn ran a test command
- `debugging`: a tool call failed and a file was edited after it in the same turn

From your wording, with local models only, never keywords:

- `tagging/nli.py`, `MoritzLaurer/deberta-v3-large-zeroshot-v2.0`: `frustrated`, `correction`,
  `style`, `approval`, `question`, `new_task`
- `tagging/emotion.py`, `SamLowe/roberta-base-go_emotions`: `frustrated` when annoyance, anger,
  disapproval or disappointment reaches 0.3

Model scores are cached in `~/.cache/agent-insights/`, so reruns only classify new messages.

## Score

Each turn (one of your messages plus everything the agent did until the next one) gets a
penalty: the sum of `TAG_WEIGHTS` in `stats.py` over its tags, plus 0.1 per failed tool
call. A session scores `100 × e^(−3 × penalty per turn)`, ×1.1 when it made at least one
commit, ×0.9–1.1 for cost per API call against the median session, capped at 100. A project
score is its sessions' scores weighted by turns.

## Cost

LiteLLM's price table, cached for 24h, with 1-hour cache writes at their own rate. Tokens are
counted once per API message, including when a continued session repeats earlier records.
