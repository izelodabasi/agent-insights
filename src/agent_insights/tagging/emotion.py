from agent_insights.models import Session
from agent_insights.tagging.classifier import CACHE_DIR, Classifier, device, score_messages

MODEL = "SamLowe/roberta-base-go_emotions"
DISLIKE_SIGNALS = ("annoyance", "anger", "disapproval", "disappointment")


def tag_sessions(sessions: list[Session], threshold: float = 0.3) -> None:
    """Tag a message disliked when a negative GoEmotions signal reaches the threshold.

    Zero-shot NLI misses short complaints with no mention of the assistant ("bro looks
    shit"); GoEmotions was trained on exactly that register.
    """
    for e, scores in score_messages(sessions, CACHE_DIR / "emotion_scores.json", _load, 32):
        if max(scores.get(label, 0) for label in DISLIKE_SIGNALS) >= threshold:
            e.tags.add("disliked")


def _load() -> Classifier:
    from transformers import pipeline

    clf = pipeline("text-classification", model=MODEL, top_k=None, device=device(), truncation=True)

    def classify(texts: list[str]) -> list[dict[str, float]]:
        return [{r["label"]: round(r["score"], 4) for r in res} for res in clf(texts)]

    return classify
