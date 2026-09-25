import argparse
from datetime import date
from pathlib import Path

from agent_insights import sources
from agent_insights.cost import load_prices
from agent_insights.date_filter import filter_sessions, resolve_range
from agent_insights.report import write_report
from agent_insights.stats import build
from agent_insights.tagging import rules


def _date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a date in YYYY-MM-DD format") from error


def main() -> None:
    parser = argparse.ArgumentParser(prog="agent-insights")
    sub = parser.add_subparsers(dest="cmd", required=True)
    scan = sub.add_parser("scan", help="read agent logs and write an HTML report")
    scan.add_argument("--out", type=Path, default=Path("report.html"))
    scan.add_argument(
        "--no-nli", action="store_true", help="skip the local NLI and emotion taggers"
    )
    dates = scan.add_mutually_exclusive_group()
    dates.add_argument(
        "--since", type=_date, metavar="YYYY-MM-DD", help="include activity on or after this date"
    )
    dates.add_argument(
        "--days",
        type=int,
        choices=(7, 30, 90),
        help="include the last 7, 30, or 90 days",
    )
    scan.add_argument(
        "--until", type=_date, metavar="YYYY-MM-DD", help="include activity on or before this date"
    )
    args = parser.parse_args()

    try:
        since, until = resolve_range(args.since, args.until, args.days)
    except ValueError as error:
        parser.error(str(error))

    agents = sources.discover()
    print(f"agents found: {', '.join(agents) or 'none'}")
    sessions = filter_sessions(sources.load_all(), since, until)
    print(f"sessions: {len(sessions)}")
    for s in sessions:
        rules.tag_session(s)
    if not args.no_nli:
        from agent_insights.tagging import emotion, nli

        nli.tag_sessions(sessions)
        emotion.tag_sessions(sessions)
    data = build(sessions, load_prices(), agents, since, until)
    write_report(data, args.out)
    print(f"report: {args.out.resolve()}")


if __name__ == "__main__":
    main()
