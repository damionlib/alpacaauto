from trading_agent.config import Settings
from trading_agent.models import (
    AccountSnapshot,
    AssetClass,
    OrderSide,
    Position,
    TradeCandidate,
)
from trading_agent.risk import RiskEngine


def test_daily_loss_stop_rejects_trade() -> None:
    settings = Settings()
    engine = RiskEngine(settings)
    decision = engine.evaluate(
        TradeCandidate(
            symbol="SPY",
            asset_class=AssetClass.EQUITY,
            side=OrderSide.BUY,
            strategy="equity_momentum",
            score=90,
            entry_price=100,
            stop_price=95,
        ),
        AccountSnapshot(equity=96_000, cash=20_000, buying_power=20_000, last_equity=100_000),
        [],
    )
    assert not decision.approved
    assert "Daily loss stop" in decision.reason


def test_equity_position_sizing_respects_two_percent_risk() -> None:
    settings = Settings()
    engine = RiskEngine(settings)
    decision = engine.evaluate(
        TradeCandidate(
            symbol="SPY",
            asset_class=AssetClass.EQUITY,
            side=OrderSide.BUY,
            strategy="equity_momentum",
            score=90,
            entry_price=100,
            stop_price=95,
        ),
        AccountSnapshot(equity=100_000, cash=50_000, buying_power=50_000, last_equity=100_000),
        [],
    )
    assert decision.approved
    assert decision.intent is not None
    assert decision.intent.qty == 120


def test_position_cap_rejects_existing_full_position() -> None:
    settings = Settings()
    engine = RiskEngine(settings)
    decision = engine.evaluate(
        TradeCandidate(
            symbol="SPY",
            asset_class=AssetClass.EQUITY,
            side=OrderSide.BUY,
            strategy="equity_momentum",
            score=90,
            entry_price=100,
            stop_price=95,
        ),
        AccountSnapshot(equity=100_000, cash=50_000, buying_power=50_000, last_equity=100_000),
        [Position(symbol="SPY", asset_class=AssetClass.EQUITY, qty=120, market_value=12_000)],
    )
    assert not decision.approved
    assert "Position cap" in decision.reason


def test_order_sizing_respects_buying_power_when_cash_is_higher() -> None:
    settings = Settings()
    engine = RiskEngine(settings)
    decision = engine.evaluate(
        TradeCandidate(
            symbol="SPY",
            asset_class=AssetClass.EQUITY,
            side=OrderSide.BUY,
            strategy="equity_momentum",
            score=90,
            entry_price=100,
            stop_price=95,
        ),
        AccountSnapshot(equity=100_000, cash=100_000, buying_power=6_000, last_equity=100_000),
        [],
    )
    assert decision.approved
    assert decision.intent is not None
    assert decision.intent.qty == 10


def test_long_option_rejects_premium_above_cap() -> None:
    settings = Settings()
    engine = RiskEngine(settings)
    decision = engine.evaluate(
        TradeCandidate(
            symbol="AAPL260116C00200000",
            asset_class=AssetClass.OPTION,
            side=OrderSide.BUY,
            strategy="long_call",
            score=90,
            entry_price=25,
            metadata={"contract": {"strike_price": "200"}},
        ),
        AccountSnapshot(equity=100_000, cash=50_000, buying_power=50_000, last_equity=100_000),
        [],
    )
    assert not decision.approved
    assert "premium cap" in decision.reason


def test_long_option_approval_sets_limit_price() -> None:
    settings = Settings()
    engine = RiskEngine(settings)
    decision = engine.evaluate(
        TradeCandidate(
            symbol="AAPL260116C00200000",
            asset_class=AssetClass.OPTION,
            side=OrderSide.BUY,
            strategy="long_call",
            score=90,
            entry_price=4.25,
            metadata={"contract": {"strike_price": "200"}},
        ),
        AccountSnapshot(equity=100_000, cash=50_000, buying_power=50_000, last_equity=100_000),
        [],
    )
    assert decision.approved
    assert decision.intent is not None
    assert decision.intent.limit_price == 4.25


def test_debit_spread_approval_builds_mleg_order() -> None:
    settings = Settings()
    engine = RiskEngine(settings)
    decision = engine.evaluate(
        TradeCandidate(
            symbol="AAPL_call_debit_spread",
            asset_class=AssetClass.OPTION,
            side=OrderSide.BUY,
            strategy="call_debit_spread",
            score=90,
            entry_price=3.10,
            metadata={
                "legs": [
                    {
                        "symbol": "AAPL260116C00200000",
                        "ratio_qty": "1",
                        "side": "buy",
                        "position_intent": "buy_to_open",
                    },
                    {
                        "symbol": "AAPL260116C00210000",
                        "ratio_qty": "1",
                        "side": "sell",
                        "position_intent": "sell_to_open",
                    },
                ]
            },
        ),
        AccountSnapshot(equity=100_000, cash=50_000, buying_power=50_000, last_equity=100_000),
        [],
    )
    assert decision.approved
    assert decision.intent is not None
    assert decision.intent.order_class == "mleg"
    assert decision.intent.limit_price == 3.10
