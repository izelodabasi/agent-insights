import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path

from agent_insights.models import Event, Session

CACHE_DIR = Path.home() / ".cache" / "agent-insights"
PASTED = re.compile(r"<pasted_content[^>]*>.*?</pasted_content[^>]*>", re.S)

type Scores = dict[str, float]
type Classifier = Callable[[list[str]], list[Scores]]


def score_messages(
    sessions: list[Session], cache: Path, load: Callable[[], Classifier], batch_size: int
) -> list[tuple[Event, Scores]]:
    """Scores for every user message, keyed in `cache` by text hash so reruns only classify
    new messages. The model is loaded only when something is missing, and the cache is
    saved after every batch so an interrupted run resumes where it stopped."""
    messages = [e for s in sessions for e in s.user_messages()]
    scores: dict[str, Scores] = json.loads(cache.read_text()) if cache.exists() else {}
    todo = {_key(e): _clean(e.text) for e in messages if _key(e) not in scores}
    if todo:
        classify = load()
        keys = list(todo)
        cache.parent.mkdir(parents=True, exist_ok=True)
        for i in range(0, len(keys), batch_size):
            chunk = keys[i : i + batch_size]
            scores.update(zip(chunk, classify([todo[k] for k in chunk]), strict=True))
            cache.write_text(json.dumps(scores))

    return [(e, scores[_key(e)]) for e in messages]


def device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def _key(e: Event) -> str:
    return hashlib.sha1(e.text.encode()).hexdigest()


def _clean(text: str) -> str:
    return PASTED.sub(" [pasted] ", text)[:2000]
