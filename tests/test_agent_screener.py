import pytest

from trading_agent.agent import TradingAgent
from trading_agent.config import Settings
from trading_agent.models import AssetClass, MarketSnapshot


class FakeScreener:
    async def top_symbols(self):
        return [
            type(
                "Screened",
                (),
                {
                    "symbol": "GOOD",
                    "asset_class": AssetClass.EQUITY,
                    "score": 88.0,
                    "reasons": ["test"],
                    "snapshot": MarketSnapshot(
                        symbol="GOOD",
                        asset_class=AssetClass.EQUITY,
                        price=100,
                        closes=[100 for _ in range(60)],
                    ),
                },
            )()
        ]


class EmptyScreener:
    async def top_symbols(self):
        return []


class FakeBroker:
    async def get_market_snapshot(self, symbol: str) -> MarketSnapshot:
        return MarketSnapshot(
            symbol=symbol,
            asset_class=AssetClass.EQUITY,
            price=100,
            closes=[100 for _ in range(60)],
        )


@pytest.mark.anyio
async def test_agent_uses_screener_symbols_when_enabled() -> None:
    agent = TradingAgent.__new__(TradingAgent)
    agent.settings = Settings.model_validate({"screener": {"enabled": True}})
    agent.screener = FakeScreener()
    agent.broker = FakeBroker()
    agent.console = type("Console", (), {"print": lambda *args, **kwargs: None})()
    agent._audit_event = lambda *args, **kwargs: None

    screened = await agent._screened_symbols(None)

    assert [symbol for symbol, _ in screened] == ["GOOD"]


@pytest.mark.anyio
async def test_agent_falls_back_to_configured_symbols_when_screener_empty() -> None:
    agent = TradingAgent.__new__(TradingAgent)
    agent.settings = Settings.model_validate(
        {
            "strategy": {"symbols": ["AAPL", "MSFT"]},
            "screener": {"enabled": True},
        }
    )
    agent.screener = EmptyScreener()
    agent.broker = FakeBroker()
    agent.console = type("Console", (), {"print": lambda *args, **kwargs: None})()
    agent._audit_event = lambda *args, **kwargs: None

    screened = await agent._screened_symbols(None)

    assert [symbol for symbol, _ in screened] == ["AAPL", "MSFT"]
