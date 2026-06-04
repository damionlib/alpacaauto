from datetime import UTC, datetime, timedelta

from trading_agent.catalyst.models import CatalystPrediction
from trading_agent.config import Settings
from trading_agent.day_trading import DayTradingEngine
from trading_agent.models import AccountSnapshot, AssetClass, MarketSnapshot, NewsItem, OrderSide, Position, ResearchSnapshot
from trading_agent.risk import RiskEngine


def _prediction(score: float = 88, direction: str = "bullish") -> CatalystPrediction:
    return CatalystPrediction(
        symbol="AAPL",
        asset_class="equity",
        direction=direction,
        prediction_score=score,
        bullish_score=score if direction == "bullish" else 100 - score,
        confidence="high",
        entry_allowed=True,
        market_regime="risk_on",
    )


def _market(price: float = 130) -> MarketSnapshot:
    return MarketSnapshot(
        symbol="AAPL",
        asset_class=AssetClass.EQUITY,
        price=price,
        closes=[90 + index for index in range(60)],
        metadata={"vwap": 125, "relative_volume": 2.0, "spread_pct": 0.05},
    )


def _research() -> ResearchSnapshot:
    return ResearchSnapshot(
        symbol="AAPL",
        news=[
            NewsItem(title="AAPL announces earnings beat and raises guidance"),
            NewsItem(title="Analyst upgrade follows record profit"),
        ],
        sec_summary={"latest_net_income": {"value": 1_000_000}, "recent_filings": [{"form": "10-Q"}]},
    )


def test_day_trading_engine_generates_bullish_entry() -> None:
    settings = Settings.model_validate({"day_trading": {"enabled": True}})
    engine = DayTradingEngine(settings)

    candidate, signal = engine.evaluate_entry(_market(price=150), _research(), _prediction(), [], trades_used_today=0)

    assert candidate is not None
    assert candidate.strategy == "day_trade_entry"
    assert candidate.side == OrderSide.BUY
    assert candidate.metadata["day_trade"] is True
    assert signal["status"] == "generated"


def test_day_trading_engine_blocks_weak_intraday_signal() -> None:
    settings = Settings.model_validate({"day_trading": {"enabled": True}})
    engine = DayTradingEngine(settings)
    market = _market(price=100).model_copy(update={"metadata": {"vwap": 125, "relative_volume": 0.7, "spread_pct": 0.05}})

    candidate, signal = engine.evaluate_entry(market, _research(), _prediction(), [], trades_used_today=0)

    assert candidate is None
    assert signal["status"] == "blocked"
    assert "Intraday score" in signal["reason"]


def test_day_trading_exit_triggers_on_stop_loss() -> None:
    settings = Settings.model_validate({"day_trading": {"enabled": True}})
    engine = DayTradingEngine(settings)
    entry_event = {"id": 10, "created_at": (datetime.now(UTC) - timedelta(minutes=20)).isoformat()}
    position = Position(
        symbol="AAPL",
        asset_class=AssetClass.EQUITY,
        qty=10,
        market_value=1_280,
        avg_entry_price=130,
        current_price=128,
        unrealized_pl=-20,
    )

    candidate, signal = engine.evaluate_exit(
        position,
        _market(price=128),
        _research(),
        _prediction(),
        entry_event=entry_event,
    )

    assert candidate is not None
    assert candidate.strategy == "day_trade_exit"
    assert candidate.metadata["exit"] is True
    assert "stop loss" in signal["reason"]


def test_day_trade_risk_uses_tighter_position_sizing() -> None:
    settings = Settings.model_validate(
        {
            "day_trading": {
                "enabled": True,
                "risk_per_trade_pct": 0.25,
                "max_position_pct": 4.0,
            }
        }
    )
    engine = DayTradingEngine(settings)
    candidate, _signal = engine.evaluate_entry(_market(price=150), _research(), _prediction(), [], trades_used_today=0)

    decision = RiskEngine(settings).evaluate(
        candidate,
        AccountSnapshot(equity=100_000, cash=100_000, buying_power=100_000, last_equity=100_000),
        [],
    )

    assert decision.approved
    assert decision.intent is not None
    assert decision.intent.qty <= 250
