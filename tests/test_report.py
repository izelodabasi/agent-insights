import re

from agent_insights.report import TEMPLATE


def test_template_has_no_remote_dependencies():
    template = TEMPLATE.read_text()

    assert re.search(r"https?://", template) is None
    assert "connect-src 'none'" in template
    assert "default-src 'none'" in template
