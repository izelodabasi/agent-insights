import math
import re
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from statistics import median

from agent_insights.cost import Prices, cost
from agent_insights.models import Event, Kind, Session
from agent_insights.tagging.rules import EDIT_TOOLS, successful_commits, turns

TAG_WEIGHTS = {
    "frustrated": 1.0,
    "correction": 0.8,
    "rejected": 0.6,
    "low_performance": 0.6,
    "interrupted": 0.4,
    "debugging": 0.2,
    "failed_tool_call": 0.1,
    "approval": -0.2,
}


def build(sessions: list[Session], prices: Prices, agents: list[str]) -> dict:
    views = {"all": summarize(sessions, prices)}
    for agent in agents:
        views[agent] = summarize([s for s in sessions if s.agent == agent], prices)

    return {
        "generated": datetime.now().isoformat(timespec="minutes"),
        "agents": agents,
        "views": views,
    }


def summarize(sessions: list[Session], prices: Prices) -> dict:
    spent = {s.session_id: _session_cost(s, prices) for s in sessions}
    factors = _cost_factors(sessions, spent)
    perf = {s.session_id: performance(s, factors[s.session_id]) for s in sessions}
    by_project: dict[str, list[Session]] = defaultdict(list)
    for s in sessions:
        by_project[s.project].append(s)

    return {
        "totals": _totals(sessions, spent),
        "projects": sorted(
            (_project(name, group, spent, perf) for name, group in by_project.items()),
            key=lambda p: p["prompts"],
            reverse=True,
        ),
        "models": _models(sessions, prices),
        "daily": _daily(sessions, prices),
        "hours": _hours(sessions),
        "tools": _tools(sessions),
        "sessions": sorted(
            (_session_row(s, spent[s.session_id], perf[s.session_id]) for s in sessions),
            key=lambda r: r["start"],
            reverse=True,
        ),
        "messages": _messages(sessions, prices),
    }


def active_seconds(session: Session) -> float:
    """Wall time between consecutive events, with gaps over 30 minutes counted as idle."""
    stamps = [e.timestamp for e in session.events]
    gaps = (b - a for a, b in zip(stamps, stamps[1:], strict=False))
    return sum(g.total_seconds() for g in gaps if g <= timedelta(minutes=30))


def performance(s: Session, cost_factor: float = 1.0) -> dict:
    """100 x e^(-3 x friction per turn), times 1.1 when the session made at least one commit
    and times cost_factor, capped at 100.

    A turn's friction is the sum of TAG_WEIGHTS over every tag on its prompt and on the
    rejections and interrupts inside it, plus one failed_tool_call per failed tool call.
    Averaging exponentially keeps a majority of clean turns from hiding the bad ones.
    """
    friction = _turn_friction(s)
    if not friction:
        return {"score": None, "turns": 0, "bad_turns": 0}

    score = 100 * math.exp(-3 * sum(friction) / len(friction)) * cost_factor
    if successful_commits(s):
        score *= 1.1
    return {
        "score": min(100, round(score)),
        "turns": len(friction),
        "bad_turns": sum(1 for f in friction if f > 0),
    }


def score_trend(friction: list[float], window: int = 5, points: int = 60) -> list[int]:
    """Score over the last `window` turns at each turn, thinned to at most `points` values."""
    trend = [
        round(100 * math.exp(-3 * sum(chunk) / len(chunk)))
        for i in range(len(friction))
        if (chunk := friction[max(0, i - window + 1) : i + 1])
    ]
    step = max(1, math.ceil(len(trend) / points))
    return trend[::step]


def commit_subject(command: str) -> str:
    """First line of the message passed to `git commit -m`, including the heredoc form."""
    if m := re.search(r"<<\s*'?EOF'?\s*\n\s*(.+)", command):
        return m.group(1).strip()
    if m := re.search(r"-m\s+(\"([^\"]*)\"|'([^']*)')", command):
        return (m.group(2) or m.group(3) or "").split("\n")[0].strip()

    return "commit"


def _turn_friction(s: Session) -> list[float]:
    friction = []
    for opener, window in turns(s):
        tags = [*opener.tags, *(t for e in window for t in e.tags)]
        tags += ["failed_tool_call" for e in window if e.kind == Kind.TOOL_RESULT and e.is_error]
        friction.append(max(0.0, sum(TAG_WEIGHTS.get(t, 0) for t in tags)))

    return friction


def _cost_factors(sessions: list[Session], spent: dict[str, float]) -> dict[str, float]:
    per_call = [spent[s.session_id] / len(s.usage) for s in sessions if s.usage]
    typical = median(per_call) if per_call else 0.0
    return {
        s.session_id: _cost_factor(spent[s.session_id], len(s.usage), typical) for s in sessions
    }


def _cost_factor(total: float, calls: int, typical_per_call: float) -> float:
    """x0.9 at twice the typical cost per API call, x1.1 at half of it, bounded to that range."""
    if not calls or not total or not typical_per_call:
        return 1.0

    return min(1.1, max(0.9, 1 + 0.1 * math.log2(typical_per_call / (total / calls))))


def _session_cost(s: Session, prices: Prices) -> float:
    return sum(c for u in s.usage.values() if (c := cost(u, prices)) is not None)


def _day(ts: datetime) -> str:
    return ts.astimezone().date().isoformat()


def _prompts(sessions: list[Session]) -> list[str]:
    return [e.text for s in sessions for e in s.of(Kind.PROMPT)]


def _tag_counts(sessions: list[Session]) -> dict[str, int]:
    return dict(Counter(t for s in sessions for e in s.events for t in e.tags))


def _totals(sessions: list[Session], spent: dict[str, float]) -> dict:
    usage = [u for s in sessions for u in s.usage.values()]
    return {
        "sessions": len(sessions),
        "projects": len({s.project for s in sessions}),
        "prompts": len(_prompts(sessions)),
        "active_hours": round(sum(map(active_seconds, sessions)) / 3600, 1),
        "cost": round(sum(spent.values()), 2),
        "tokens_in": sum(u.input_tokens for u in usage),
        "tokens_out": sum(u.output_tokens for u in usage),
        "tokens_cached": sum(u.cache_read_tokens for u in usage),
        "tokens_written": sum(u.cache_write_tokens for u in usage),
        "tags": _tag_counts(sessions),
    }


def _project(
    name: str, sessions: list[Session], spent: dict[str, float], perf: dict[str, dict]
) -> dict:
    events = [e for s in sessions for e in s.events]
    scored = [p for s in sessions if (p := perf[s.session_id])["score"] is not None]
    turns_total = sum(p["turns"] for p in scored)
    return {
        "name": name,
        "sessions": len(sessions),
        "prompts": len(_prompts(sessions)),
        "tool_calls": sum(1 for e in events if e.kind == Kind.TOOL_USE),
        "tool_errors": sum(1 for e in events if e.kind == Kind.TOOL_RESULT and e.is_error),
        "active_hours": round(sum(map(active_seconds, sessions)) / 3600, 1),
        "cost": round(sum(spent[s.session_id] for s in sessions), 2),
        "median_prompt_len": int(median([len(p) for p in _prompts(sessions)] or [0])),
        "first": _day(min(s.start for s in sessions)),
        "last": _day(max(s.end for s in sessions)),
        "tags": _tag_counts(sessions),
        "score": round(sum(p["score"] * p["turns"] for p in scored) / turns_total)
        if turns_total
        else None,
        "turns": turns_total,
        "bad_turns": sum(p["bad_turns"] for p in scored),
    }


def _models(sessions: list[Session], prices: Prices) -> list[dict]:
    models: dict[str, dict] = {}
    for u in (u for s in sessions for u in s.usage.values()):
        m = models.setdefault(
            u.model,
            {
                "model": u.model,
                "input": 0,
                "output": 0,
                "cache_write": 0,
                "cache_read": 0,
                "cost": 0.0,
                "priced": True,
            },
        )
        m["input"] += u.input_tokens
        m["output"] += u.output_tokens
        m["cache_write"] += u.cache_write_tokens
        m["cache_read"] += u.cache_read_tokens
        if (c := cost(u, prices)) is None:
            m["priced"] = False
        else:
            m["cost"] += c
    return sorted(models.values(), key=lambda m: m["cost"], reverse=True)


def _daily(sessions: list[Session], prices: Prices) -> list[dict]:
    daily: dict[str, Counter] = defaultdict(Counter)
    for s in sessions:
        for u in s.usage.values():
            daily[_day(u.timestamp or s.start)]["cost"] += cost(u, prices) or 0.0
        for e in s.of(Kind.PROMPT):
            daily[_day(e.timestamp)]["prompts"] += 1
    return [
        {"date": d, "prompts": c["prompts"], "cost": round(c["cost"], 4)}
        for d, c in sorted(daily.items())
    ]


def _hours(sessions: list[Session]) -> list[int]:
    hours = [0] * 24
    for e in (e for s in sessions for e in s.of(Kind.PROMPT)):
        hours[e.timestamp.astimezone().hour] += 1
    return hours


def _tools(sessions: list[Session]) -> list[dict]:
    tools: dict[str, Counter] = defaultdict(Counter)
    for s in sessions:
        names = {e.tool_id: e.tool for e in s.of(Kind.TOOL_USE)}
        for e in s.events:
            match e.kind:
                case Kind.TOOL_USE:
                    tools[e.tool]["calls"] += 1
                case Kind.TOOL_RESULT if e.is_error:
                    tools[names.get(e.tool_id, "?")]["errors"] += 1
                case Kind.REJECTION:
                    tools[names.get(e.tool_id, "?")]["rejections"] += 1
    return sorted(
        ({"tool": t, **c} for t, c in tools.items()), key=lambda t: t["calls"], reverse=True
    )


def _turn_spend(s: Session, prices: Prices) -> tuple[list[datetime], list[list]]:
    """Start time of each turn, and [output tokens, cost] of the agent's work in it."""
    starts = [opener.timestamp for opener, _ in turns(s)]
    spend = [[0, 0.0] for _ in starts]
    for u in s.usage.values():
        if (i := bisect_right(starts, u.timestamp or s.start) - 1) >= 0:
            spend[i][0] += u.output_tokens
            spend[i][1] += cost(u, prices) or 0.0
    return starts, spend


def _messages(sessions: list[Session], prices: Prices) -> list[dict]:
    """Each user message with the output tokens and cost of the agent's work in its turn."""
    messages = []
    for s in sessions:
        starts, spend = _turn_spend(s, prices)
        for e in s.user_messages():
            i = bisect_right(starts, e.timestamp) - 1
            out_tokens, turn_cost = spend[i] if i >= 0 else (0, 0.0)
            messages.append(
                {
                    "project": s.project,
                    "session": s.session_id,
                    "ts": e.timestamp.astimezone().isoformat(timespec="minutes"),
                    "kind": e.kind.value,
                    "tokens_out": out_tokens,
                    "cost": round(turn_cost, 4),
                    "text": e.text[:1500],
                    "tags": sorted(e.tags),
                }
            )
    return sorted(messages, key=lambda m: m["ts"], reverse=True)


def _session_row(s: Session, spent: float, perf: dict) -> dict:
    return {
        "id": s.session_id,
        "agent": s.agent,
        "project": s.project,
        "name": s.title or s.ai_title,
        "start": s.start.astimezone().isoformat(timespec="minutes"),
        "end": s.end.astimezone().isoformat(timespec="minutes"),
        "active_min": round(active_seconds(s) / 60),
        "prompts": len(s.of(Kind.PROMPT)),
        "cost": round(spent, 2),
        "cost_per_call": round(spent / len(s.usage), 4) if s.usage else 0,
        "commits": [commit_subject(e.tool_input.get("command", "")) for e in successful_commits(s)],
        "files_edited": len(_edited_files(s)),
        "tags": sorted(s.tags),
        "trend": score_trend(_turn_friction(s)),
        **perf,
    }


def _edited_files(s: Session) -> set[str]:
    ok = {e.tool_id for e in s.of(Kind.TOOL_RESULT) if not e.is_error}
    return {path for e in s.events if e.tool_id in ok and (path := _edit_path(e))}


def _edit_path(e: Event) -> str | None:
    if e.kind != Kind.TOOL_USE or e.tool not in EDIT_TOOLS or not e.tool_input.get("file_path"):
        return None

    return str(e.tool_input["file_path"])
