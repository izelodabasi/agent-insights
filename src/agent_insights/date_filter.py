from datetime import date, timedelta

from agent_insights.models import Session


def resolve_range(
    since: date | None,
    until: date | None,
    days: int | None,
    *,
    today: date | None = None,
) -> tuple[date | None, date | None]:
    """Resolve an optional rolling-day preset into inclusive date boundaries."""
    if days is not None:
        end = until or today or date.today()
        return end - timedelta(days=days - 1), end
    if since and until and since > until:
        raise ValueError("--since must be on or before --until")

    return since, until


def filter_sessions(
    sessions: list[Session], since: date | None = None, until: date | None = None
) -> list[Session]:
    """Return copies containing only activity within the inclusive local-date range.

    Usage records without their own timestamp are assigned to the original session start,
    which is the most precise placement available from those source logs.
    """
    if since and until and since > until:
        raise ValueError("--since must be on or before --until")
    if since is None and until is None:
        return sessions

    filtered = []
    for session in sessions:
        events = [
            event
            for event in session.events
            if _includes(event.timestamp.astimezone().date(), since, until)
        ]
        usage = {
            key: record
            for key, record in session.usage.items()
            if _includes((record.timestamp or session.start).astimezone().date(), since, until)
        }
        if events:
            filtered.append(session.model_copy(update={"events": events, "usage": usage}))

    return filtered


def _includes(value: date, since: date | None, until: date | None) -> bool:
    return (since is None or value >= since) and (until is None or value <= until)
