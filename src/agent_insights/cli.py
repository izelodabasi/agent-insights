import argparse
from pathlib import Path

from agent_insights import sources
from agent_insights.cost import load_prices
from agent_insights.report import write_report
from agent_insights.stats import build
from agent_insights.tagging import rules


def main() -> None:
    parser = argparse.ArgumentParser(prog="agent-insights")
    sub = parser.add_subparsers(dest="cmd", required=True)
    scan = sub.add_parser("scan", help="read agent logs and write an HTML report")
    scan.add_argument("--out", type=Path, default=Path("report.html"))
    scan.add_argument(
        "--no-nli", action="store_true", help="skip the local NLI and emotion taggers"
    )
    args = parser.parse_args()

    agents = sources.discover()
    print(f"agents found: {', '.join(agents) or 'none'}")
    sessions = sources.load_all()
    print(f"sessions: {len(sessions)}")
    for s in sessions:
        rules.tag_session(s)
    if not args.no_nli:
        from agent_insights.tagging import emotion, nli

        nli.tag_sessions(sessions)
        emotion.tag_sessions(sessions)
    data = build(sessions, load_prices(), agents)
    write_report(data, args.out)
    print(f"report: {args.out.resolve()}")


if __name__ == "__main__":
    main()
