import pytest

from trading_agent.config import Settings
from trading_agent.research.crypto import CryptoResearchService


def test_crypto_regime_scores_risk_on_market() -> None:
    service = CryptoResearchService(Settings())

    regime = service._regime(
        {
            "global": {
                "market_cap_change_24h_pct": 3.0,
                "btc_dominance_pct": 48.0,
            },
            "asset": {"price_change_7d_pct": 8.0},
            "fear_greed": {"value": 55},
            "funding": {"last_funding_rate": 0.0001},
        }
    )

    assert regime["label"] == "risk_on"
    assert regime["score"] > 65


def test_crypto_regime_flags_crowded_risk() -> None:
    service = CryptoResearchService(Settings())

    summary = {
        "fear_greed": {"value": 85},
        "funding": {"last_funding_rate": 0.0009},
        "stablecoins": {"total_circulating_usd": 100_000_000_000},
        "regime": {"label": "neutral"},
    }

    flags = service._risk_flags(summary)

    assert "extreme_greed" in flags
    assert "perp_funding_crowded_long" in flags


def test_crypto_optional_feeds_are_explicitly_not_configured() -> None:
    service = CryptoResearchService(Settings())

    assert service._exchange_flows_stub()["status"] == "not_configured"
    assert service._onchain_stub()["status"] == "not_configured"


@pytest.mark.anyio
async def test_funding_rate_handles_restricted_venue(respx_mock) -> None:
    service = CryptoResearchService(Settings())
    respx_mock.get("https://fapi.binance.com/fapi/v1/premiumIndex").respond(451)

    funding = await service._funding_rate("BTC/USD")

    assert funding["status"] == "unavailable"
    assert funding["reason"] == "venue_restricted_or_unavailable_from_current_location"
