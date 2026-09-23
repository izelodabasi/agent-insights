from agent_insights.models import Session
from agent_insights.sources import claude_code, codex, copilot

SOURCES = {"claude-code": claude_code, "codex": codex, "copilot": copilot}


def discover() -> list[str]:
    return [name for name, mod in SOURCES.items() if mod.installed()]


def load_all() -> list[Session]:
    return [s for name in discover() for s in SOURCES[name].load_sessions()]
