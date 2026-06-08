import pytest
from pydantic import ValidationError

from trading_agent.agent import TradingAgent
from trading_agent.audit import AuditStore
from trading_agent.config import PositionManagerConfig, Settings
from trading_agent.models import (
    AccountSnapshot,
    AssetClass,
    MarketSnapshot,
    OrderSide,
    Position,
    ResearchSnapshot,
    TradeCandidate,
)
from trading_agent.risk import RiskEngine
from trading_agent.strategies.momentum import MomentumStrategy


# --- #4 correlated-exposure cap ---------------------------------------------
def _account(equity=100_000.0) -> AccountSnapshot:
    return AccountSnapshot(equity=equity, cash=equity / 2, buying_power=equity / 2, last_equity=equity)


def _nvda_buy() -> TradeCandidate:
    return TradeCandidate(
        symbol="NVDA",
        asset_class=AssetClass.EQUITY,
        side=OrderSide.BUY,
        strategy="equity_momentum",
        score=90,
        entry_price=100,
        stop_price=95,
    )


def test_correlated_exposure_cap_rejects_overweight_cluster() -> None:
    engine = RiskEngine(Settings())  # default cap 25% of equity, AI/semis cluster
    decision = engine.evaluate(
        _nvda_buy(),
        _account(100_000),
        [Position(symbol="AVGO", asset_class=AssetClass.EQUITY, qty=100, market_value=26_000)],
    )
    assert not decision.approved
    assert "Correlated-group exposure cap" in decision.reason


def test_correlated_exposure_cap_allows_within_limit() -> None:
    engine = RiskEngine(Settings())
    decision = engine.evaluate(
        _nvda_buy(),
        _account(100_000),
        [Position(symbol="AVGO", asset_class=AssetClass.EQUITY, qty=100, market_value=10_000)],
    )
    assert decision.approved


def test_correlated_exposure_counts_option_underlyings() -> None:
    engine = RiskEngine(Settings())
    # A long NVDA call counts toward the NVDA/AI cluster exposure.
    decision = engine.evaluate(
        _nvda_buy(),
        _account(100_000),
        [Position(symbol="AVGO260618C00400000", asset_class=AssetClass.OPTION, qty=10, market_value=26_000)],
    )
    assert not decision.approved


# --- #2 drawdown circuit breaker --------------------------------------------
def test_drawdown_halt_triggers_below_trailing_peak(tmp_path) -> None:
    agent = TradingAgent.__new__(TradingAgent)
    agent.settings = Settings()  # max_drawdown_halt_pct default 8
    agent.audit = AuditStore(tmp_path / "a.sqlite3")
    agent.audit.start_cycle(_account(100_000), [])  # peak 100k

    halt, info = agent._drawdown_halt(_account(91_000))  # -9% from peak
    assert halt is True
    assert info["drawdown_pct"] > 8


def test_drawdown_halt_quiet_within_threshold(tmp_path) -> None:
    agent = TradingAgent.__new__(TradingAgent)
    agent.settings = Settings()
    agent.audit = AuditStore(tmp_path / "a.sqlite3")
    agent.audit.start_cycle(_account(100_000), [])

    halt, _ = agent._drawdown_halt(_account(96_000))  # -4% only
    assert halt is False


# --- #3 broad-market regime gate --------------------------------------------
class _RegimeBroker:
    def __init__(self, price: float) -> None:
        self._price = price

    async def get_market_snapshot(self, symbol: str) -> MarketSnapshot:
        return MarketSnapshot(symbol=symbol, asset_class=AssetClass.EQUITY, price=self._price, closes=[100.0] * 60)


@pytest.mark.anyio
async def test_regime_gate_flags_downtrend() -> None:
    agent = TradingAgent.__new__(TradingAgent)
    agent.settings = Settings()
    agent.broker = _RegimeBroker(price=90.0)  # below SMA50 (=100)
    ok, info = await agent._market_regime_ok()
    assert ok is False
    assert info["uptrend"] is False


@pytest.mark.anyio
async def test_regime_gate_allows_uptrend() -> None:
    agent = TradingAgent.__new__(TradingAgent)
    agent.settings = Settings()
    agent.broker = _RegimeBroker(price=110.0)  # above SMA50
    ok, info = await agent._market_regime_ok()
    assert ok is True


# --- #6 pullback-in-uptrend entry -------------------------------------------
def test_momentum_blocks_equity_entry_in_downtrend() -> None:
    strat = MomentumStrategy(Settings())
    closes = [200.0 - i for i in range(60)]  # falling -> sma20 < sma50
    market = MarketSnapshot(symbol="AAPL", asset_class=AssetClass.EQUITY, price=closes[-1], closes=closes)
    assert strat.evaluate(market, ResearchSnapshot(symbol="AAPL")) == []


def test_momentum_prefers_pullback_over_extended() -> None:
    strat = MomentumStrategy(Settings())
    closes = [100.0 + i for i in range(60)]  # steady uptrend; sma20 ~149.5
    research = ResearchSnapshot(symbol="AAPL")
    pullback = MarketSnapshot(symbol="AAPL", asset_class=AssetClass.EQUITY, price=150.0, closes=closes)
    extended = MarketSnapshot(symbol="AAPL", asset_class=AssetClass.EQUITY, price=172.0, closes=closes)
    assert len(strat.evaluate(pullback, research)) == 1
    assert strat.evaluate(extended, research) == []


# --- profit-lock config validator -------------------------------------------
def test_profit_lock_rejects_lock_ge_profit() -> None:
    with pytest.raises(ValidationError):
        PositionManagerConfig(profit_lock_steps=[{"profit_pct": 5.0, "lock_pct": 8.0}])


def test_profit_lock_steps_get_sorted() -> None:
    cfg = PositionManagerConfig(
        profit_lock_steps=[{"profit_pct": 10.0, "lock_pct": 8.0}, {"profit_pct": 5.0, "lock_pct": 2.0}]
    )
    assert [s.profit_pct for s in cfg.profit_lock_steps] == [5.0, 10.0]
