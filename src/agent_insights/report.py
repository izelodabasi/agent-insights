import json
from pathlib import Path

TEMPLATE = Path(__file__).with_name("report.html")


def write_report(data: dict, out: Path) -> None:
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    out.write_text(TEMPLATE.read_text().replace("/*DATA*/null", payload), encoding="utf-8")
