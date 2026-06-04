import pytest

from trading_agent.agent import TradingAgent
from trading_agent.catalyst.service import CatalystEngine
from trading_agent.config import Settings
from trading_agent.day_trading import DayTradingEngine
from trading_agent.models import AssetClass, MarketSnapshot, NewsItem, ResearchSnapshot


class FakeBroker:
    async def get_market_snapshot(self, symbol: str) -> MarketSnapshot:
        return MarketSnapshot(
            symbol=symbol,
            asset_class=AssetClass.EQUITY,
            price=130,
            closes=[90 + index for index in range(60)],
        )


class FakeDayTradeBroker:
    async def get_market_snapshot(self, symbol: str) -> MarketSnapshot:
        return MarketSnapshot(
            symbol=symbol,
            asset_class=AssetClass.EQUITY,
            price=150,
            closes=[90 + index for index in range(60)],
            metadata={"vwap": 125, "relative_volume": 2.0, "spread_pct": 0.05},
        )


class FakeResearch:
    async def research_symbol(self, symbol: str) -> ResearchSnapshot:
        return ResearchSnapshot(
            symbol=symbol,
            news=[
                NewsItem(title=f"{symbol} announces earnings beat and raises guidance"),
                NewsItem(title=f"{symbol} secures new partnership contract"),
                NewsItem(title=f"Analyst upgrade follows record profit at {symbol}"),
            ],
            sec_summary={
                "latest_revenue": {"value": 1_000_000_000},
                "latest_net_income": {"value": 200_000_000},
                "recent_filings": [{"form": "10-Q"}],
            },
        )


class EmptyMomentum:
    def evaluate(self, market, research):
        return []


class SwingMomentum:
    def evaluate(self, market, research):
        from trading_agent.models import OrderSide, TradeCandidate

        return [
            TradeCandidate(
                symbol=market.symbol,
                asset_class=market.asset_class,
                side=OrderSide.BUY,
                strategy="equity_momentum",
                score=80,
                entry_price=market.price,
                stop_price=market.price * 0.95,
            )
        ]


@pytest.mark.anyio
async def test_agent_audits_catalyst_prediction_and_generates_entry() -> None:
    settings = Settings.model_validate(
        {
            "strategy": {"symbols": ["GOOD"], "allow_options": False},
            "screener": {"enabled": False},
            "catalyst": {"min_entry_score": 70},
            "day_trading": {"enabled": False},
        }
    )
    agent = TradingAgent.__new__(TradingAgent)
    agent.settings = settings
    agent.broker = FakeBroker()
    agent.research = FakeResearch()
    agent.momentum = EmptyMomentum()
    agent.catalyst = CatalystEngine(settings)
    agent.console = type("Console", (), {"print": lambda *args, **kwargs: None})()
    events = []
    agent._audit_event = lambda *args, **kwargs: events.append((args, kwargs))

    candidates = await agent._generate_candidates([], cycle_id=123)

    assert [candidate.strategy for candidate in candidates] == ["equity_catalyst"]
    assert any(args[1] == "catalyst_prediction" for args, _ in events)
    assert candidates[0].metadata["catalyst"]["direction"] == "bullish"


@pytest.mark.anyio
async def test_agent_audits_day_trade_signal_and_generates_day_trade_entry() -> None:
    settings = Settings.model_validate(
        {
            "strategy": {"symbols": ["GOOD"], "allow_options": False},
            "screener": {"enabled": False},
            "catalyst": {"min_entry_score": 70, "generate_entry_candidates": False},
            "day_trading": {"enabled": True, "min_intraday_score": 40, "min_combined_score": 70},
        }
    )
    agent = TradingAgent.__new__(TradingAgent)
    agent.settings = settings
    agent.broker = FakeDayTradeBroker()
    agent.research = FakeResearch()
    agent.momentum = EmptyMomentum()
    agent.catalyst = CatalystEngine(settings)
    agent.day_trading = DayTradingEngine(settings)
    agent.console = type("Console", (), {"print": lambda *args, **kwargs: None})()
    agent._day_trade_entries_today = lambda: 0
    events = []
    agent._audit_event = lambda *args, **kwargs: events.append((args, kwargs))

    candidates = await agent._generate_candidates([], cycle_id=123)

    assert any(candidate.strategy == "day_trade_entry" for candidate in candidates)
    assert any(args[1] == "day_trade_signal" for args, _ in events)


@pytest.mark.anyio
async def test_agent_prefers_swing_candidate_over_day_trade_same_symbol() -> None:
    settings = Settings.model_validate(
        {
            "strategy": {"symbols": ["GOOD"], "allow_options": False},
            "screener": {"enabled": False},
            "catalyst": {"generate_entry_candidates": False},
            "day_trading": {"enabled": True, "min_intraday_score": 40, "min_combined_score": 70},
        }
    )
    agent = TradingAgent.__new__(TradingAgent)
    agent.settings = settings
    agent.broker = FakeDayTradeBroker()
    agent.research = FakeResearch()
    agent.momentum = SwingMomentum()
    agent.catalyst = CatalystEngine(settings)
    agent.day_trading = DayTradingEngine(settings)
    agent.console = type("Console", (), {"print": lambda *args, **kwargs: None})()
    agent._day_trade_entries_today = lambda: 0
    agent._day_trade_entry_event_today = lambda symbol: None
    events = []
    agent._audit_event = lambda *args, **kwargs: events.append((args, kwargs))

    candidates = await agent._generate_candidates([], cycle_id=123)

    assert [candidate.strategy for candidate in candidates] == ["equity_momentum"]
    day_signals = [kwargs for args, kwargs in events if args[1] == "day_trade_signal"]
    assert day_signals
    assert day_signals[0]["status"] == "blocked"
    assert "preferring swing" in day_signals[0]["reason"]


def test_agent_blocks_swing_entry_after_same_day_day_trade_entry() -> None:
    settings = Settings.model_validate({"day_trading": {"enabled": True}})
    agent = TradingAgent.__new__(TradingAgent)
    agent.settings = settings
    agent._day_trade_entry_event_today = lambda symbol: {"id": 1} if symbol == "AAPL" else None

    from trading_agent.models import OrderSide, TradeCandidate

    swing = TradeCandidate(
        symbol="AAPL",
        asset_class=AssetClass.EQUITY,
        side=OrderSide.BUY,
        strategy="equity_momentum",
        score=80,
        entry_price=150,
    )

    accepted, blocked = agent._block_swing_after_day_trade_entry([swing])

    assert accepted == []
    assert blocked[0][0] == swing
    assert "position averaging" in blocked[0][1]
