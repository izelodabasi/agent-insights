from agent_insights.models import Session
from agent_insights.tagging.classifier import CACHE_DIR, Classifier, device, score_messages

MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
LABELS = {
    "frustrated": "The user is frustrated or annoyed with the assistant.",
    "correction": "The user is correcting a mistake the assistant made.",
    "style": "The user is stating a coding style or workflow preference.",
    "approval": "The user approves or is satisfied with the work.",
    "question": "The user is asking a question.",
    "new_task": "The user is giving the assistant a new task.",
}


def tag_sessions(sessions: list[Session], threshold: float = 0.8) -> None:
    for e, scores in score_messages(sessions, CACHE_DIR / "nli_tags.json", _load, 16):
        e.tags |= {label for label, score in scores.items() if score >= threshold}


def _load() -> Classifier:
    from transformers import pipeline

    clf = pipeline("zero-shot-classification", model=MODEL, device=device())
    hypotheses = {h: name for name, h in LABELS.items()}

    def classify(texts: list[str]) -> list[dict[str, float]]:
        out = clf(
            texts, candidate_labels=list(hypotheses), hypothesis_template="{}", multi_label=True
        )
        return [
            {hypotheses[h]: round(s, 4) for h, s in zip(r["labels"], r["scores"], strict=True)}
            for r in (out if isinstance(out, list) else [out])
        ]

    return classify
