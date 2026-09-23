import json
import re
import time
import urllib.request
from pathlib import Path

from agent_insights.models import Usage

PRICES_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
)
CACHE = Path.home() / ".cache" / "agent-insights" / "litellm_prices.json"

type Prices = dict[str, dict]


def load_prices(cache: Path = CACHE) -> Prices:
    """LiteLLM price table, refetched when the cached copy is older than a day.

    A failed fetch falls back to a stale cache, then to an empty table (every model unpriced).
    """
    fresh = cache.exists() and time.time() - cache.stat().st_mtime < 86400
    if not fresh:
        try:
            with urllib.request.urlopen(PRICES_URL, timeout=15) as resp:
                data = resp.read()
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(data)
        except OSError:
            pass
    if not cache.exists():
        return {}

    return json.loads(cache.read_text())


def price_for(model: str, prices: Prices) -> dict | None:
    base = re.sub(r"-\d{8}$", "", model)
    for key in (
        model,
        f"anthropic/{model}",
        f"openai/{model}",
        base,
        f"anthropic/{base}",
        f"openai/{base}",
    ):
        if (entry := prices.get(key)) and "input_cost_per_token" in entry:
            return entry

    return None


def cost(usage: Usage, prices: Prices) -> float | None:
    p = price_for(usage.model, prices)
    if p is None:
        return None

    inp = p.get("input_cost_per_token", 0)
    write_5m = usage.cache_write_tokens - usage.cache_write_1h_tokens
    write_rate = p.get("cache_creation_input_token_cost")
    if write_rate is None:
        write_rate = inp if usage.cache_write_requires_explicit_price else inp * 1.25
    return (
        usage.input_tokens * inp
        + usage.output_tokens * p.get("output_cost_per_token", 0)
        + write_5m * write_rate
        + usage.cache_write_1h_tokens * p.get("cache_creation_input_token_cost_above_1hr", inp * 2)
        + usage.cache_read_tokens * p.get("cache_read_input_token_cost", inp)
    )
