import re

from agent_insights.report import TEMPLATE


def test_template_has_no_remote_dependencies():
    template = TEMPLATE.read_text()

    assert re.search(r"https?://", template) is None
    assert "connect-src 'none'" in template
    assert "default-src 'none'" in template


def test_template_has_local_analysis_page():
    template = TEMPLATE.read_text()

    assert 'data-tab="analysis"' in template
    assert template.count('data-tab="') == 4
    assert 'data-tab="usage"' in template
    assert 'data-tab="history"' in template
    assert "Tasks with problems" in template
    assert "Files with problems" in template
