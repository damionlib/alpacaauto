from trading_agent.agent import TradingAgent
from trading_agent.config import Settings
from trading_agent.models import AssetClass, OrderIntent, OrderSide, RiskDecision, TradeCandidate


def test_cycle_interval_defaults_by_mode() -> None:
    paper = Settings()
    live = Settings.model_validate({"broker": {"mode": "live"}, "allow_live_trading": True})

    assert paper.agent.cycle_interval(live=paper.is_live) == 300
    assert live.agent.cycle_interval(live=live.is_live) == 1800


def test_daily_entry_cap_blocks_new_entries() -> None:
    agent = TradingAgent.__new__(TradingAgent)
    decision = _decision(exit_trade=False)

    reason = agent._daily_cap_reason(
        decision,
        daily_counts={"total_orders": 2, "entry_orders": 3, "exit_orders": 0},
        submitted_entries=0,
        submitted_total=0,
        max_entry_orders=3,
        max_total_orders=6,
    )

    assert "Daily entry order cap reached" in reason


def test_daily_total_cap_blocks_exits_too() -> None:
    agent = TradingAgent.__new__(TradingAgent)
    decision = _decision(exit_trade=True)

    reason = agent._daily_cap_reason(
        decision,
        daily_counts={"total_orders": 6, "entry_orders": 3, "exit_orders": 3},
        submitted_entries=0,
        submitted_total=0,
        max_entry_orders=3,
        max_total_orders=6,
    )

    assert "Daily total order cap reached" in reason


def _decision(*, exit_trade: bool) -> RiskDecision:
    candidate = TradeCandidate(
        symbol="AAPL",
        asset_class=AssetClass.EQUITY,
        side=OrderSide.SELL if exit_trade else OrderSide.BUY,
        strategy="stop_loss_exit" if exit_trade else "equity_momentum",
        score=100,
        entry_price=100,
        metadata={"exit": True} if exit_trade else {},
    )
    return RiskDecision(
        approved=True,
        reason="approved",
        intent=OrderIntent(
            symbol="AAPL",
            asset_class=AssetClass.EQUITY,
            side=candidate.side,
            qty=1,
            metadata=candidate.metadata,
        ),
        candidate=candidate,
    )
