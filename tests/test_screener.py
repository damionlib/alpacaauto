import pytest

from trading_agent.config import Settings
from trading_agent.models import AssetClass, MarketSnapshot
from trading_agent.screener.service import MarketScreener
from trading_agent.screener.universe import symbols_for_universes


class FakeBroker:
    async def get_market_snapshot(self, symbol: str) -> MarketSnapshot:
        if symbol == "GOOD":
            closes = [100 + index for index in range(60)]
            volumes = [1_000_000 for _ in closes]
            return MarketSnapshot(
                symbol=symbol,
                asset_class=AssetClass.EQUITY,
                price=160,
                closes=closes,
                metadata={"volumes": volumes},
            )
        if symbol == "BAD":
            closes = [100 - index for index in range(60)]
            volumes = [100 for _ in closes]
            return MarketSnapshot(
                symbol=symbol,
                asset_class=AssetClass.EQUITY,
                price=40,
                closes=closes,
                metadata={"volumes": volumes},
            )
        closes = [50 + index for index in range(60)]
        return MarketSnapshot(
            symbol=symbol,
            asset_class=AssetClass.CRYPTO,
            price=110,
            closes=closes,
            metadata={"volumes": [1_000 for _ in closes]},
        )


def test_symbols_for_universes_deduplicates_ordered_symbols() -> None:
    symbols = symbols_for_universes(["sp500_core", "nasdaq100"])

    assert symbols[0] == "SPY"
    assert len(symbols) == len(set(symbols))


def test_score_snapshot_accepts_liquid_uptrend() -> None:
    settings = Settings.model_validate(
        {
            "screener": {
                "enabled": True,
                "min_avg_dollar_volume": 1_000_000,
            }
        }
    )
    screener = MarketScreener(settings, FakeBroker())
    closes = [100 + index for index in range(60)]
    snapshot = MarketSnapshot(
        symbol="GOOD",
        asset_class=AssetClass.EQUITY,
        price=160,
        closes=closes,
        metadata={"volumes": [1_000_000 for _ in closes]},
    )

    result = screener.score_snapshot(snapshot)

    assert result is not None
    assert result.symbol == "GOOD"
    assert result.score >= settings.screener.min_trend_score


@pytest.mark.anyio
async def test_top_symbols_respects_max_candidates_and_crypto_limit(monkeypatch) -> None:
    settings = Settings.model_validate(
        {
            "strategy": {"allow_crypto": True},
            "screener": {
                "enabled": True,
                "max_candidates": 2,
                "max_crypto_candidates": 1,
                "universes": ["test"],
                "min_avg_dollar_volume": 1_000_000,
            },
        }
    )
    monkeypatch.setattr(
        "trading_agent.screener.service.symbols_for_universes",
        lambda universes: ["GOOD", "BAD", "BTC/USD"],
    )
    screener = MarketScreener(settings, FakeBroker())

    symbols = await screener.top_symbols()

    assert len(symbols) == 2
    assert sum(1 for symbol in symbols if symbol.asset_class == AssetClass.CRYPTO) == 1
    assert "BAD" not in {symbol.symbol for symbol in symbols}
