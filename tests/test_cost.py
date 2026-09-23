import pytest

from agent_insights.cost import cost, price_for
from agent_insights.models import Usage

PRICES = {
    "claude-opus-5": {
        "input_cost_per_token": 1e-5,
        "output_cost_per_token": 5e-5,
        "cache_creation_input_token_cost": 1.25e-5,
        "cache_read_input_token_cost": 1e-6,
    },
    "anthropic/claude-sonnet-5": {"input_cost_per_token": 3e-6, "output_cost_per_token": 1.5e-5},
}


@pytest.mark.parametrize(
    ("model", "found"),
    [
        ("claude-opus-5", True),
        ("claude-sonnet-5", True),
        ("claude-opus-5-20260101", True),
        ("gpt-9", False),
    ],
)
def test_price_lookup(model: str, found: bool):
    assert (price_for(model, PRICES) is not None) is found


def test_cost():
    u = Usage(
        model="claude-opus-5",
        input_tokens=100,
        output_tokens=10,
        cache_write_tokens=1000,
        cache_read_tokens=10000,
    )
    assert cost(u, PRICES) == pytest.approx(100e-5 + 10 * 5e-5 + 1000 * 1.25e-5 + 10000e-6)


def test_one_hour_cache_writes_use_their_own_rate():
    opus = {**PRICES["claude-opus-5"], "cache_creation_input_token_cost_above_1hr": 2e-5}
    prices = {"claude-opus-5": opus}
    u = Usage(model="claude-opus-5", cache_write_tokens=1000, cache_write_1h_tokens=600)
    assert cost(u, prices) == pytest.approx(400 * 1.25e-5 + 600 * 2e-5)


def test_one_hour_rate_falls_back_to_double_input():
    u = Usage(model="claude-opus-5", cache_write_tokens=100, cache_write_1h_tokens=100)
    assert cost(u, PRICES) == pytest.approx(100 * 2e-5)


def test_unpriced_model():
    assert cost(Usage(model="gpt-9", input_tokens=5), PRICES) is None
