from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

from trading_agent.agent import TradingAgent
from trading_agent.audit import AuditStore
from trading_agent.brokers.alpaca import AlpacaBroker
from trading_agent.config import Settings
from trading_agent.models import (
    AccountSnapshot,
    AssetClass,
    MarketSnapshot,
    OrderSide,
    Position,
    TradeCandidate,
)
from trading_agent.position_manager import PositionManager
from trading_agent.risk import RiskEngine

ET = ZoneInfo("America/New_York")


class _SilentConsole:
    def print(self, *args, **kwargs) -> None:
        return None


# --- Fix 1: intraday feature computation ------------------------------------
def _minute_bar(i: int, close: float, vol: float) -> dict:
    when = datetime.combine(datetime.now(ET).date(), time(9, 30), tzinfo=ET) + timedelta(minutes=i)
    return {
        "t": when.astimezone(ZoneInfo("UTC")).isoformat().replace("+00:00", "Z"),
        "o": close,
        "h": close + 0.5,
        "l": close - 0.5,
        "c": close,
        "v": vol,
        "vw": close,
    }


def test_intraday_features_from_today_minute_bars() -> None:
    broker = AlpacaBroker.__new__(AlpacaBroker)
    bars = [_minute_bar(i, 100.0 + i * 0.1, 1000 + i * 10) for i in range(15)]
    bars.append(_minute_bar(15, 103.0, 1500))  # break above the opening range

    features = broker._intraday_features_from_bars(bars, daily_volumes=[1_000_000.0] * 20)

    assert "vwap" in features and 100.0 < features["vwap"] < 103.5
    assert features["minute_trend_pct"] > 0
    assert features["opening_range_break"] is True
    assert features["relative_volume"] > 0


def test_intraday_features_empty_when_no_today_session() -> None:
    broker = AlpacaBroker.__new__(AlpacaBroker)
    # A bar timestamped well before today's session contributes nothing.
    old = {"t": "2020-01-02T14:30:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 1, "vw": 1}
    assert broker._intraday_features_from_bars([old], daily_volumes=[1.0]) == {}


@pytest.mark.anyio
async def test_enrich_intraday_merges_features_for_equity() -> None:
    agent = TradingAgent.__new__(TradingAgent)
    agent.settings = Settings.model_validate({"day_trading": {"enabled": True}})
    agent.console = _SilentConsole()

    class _Broker:
        async def get_intraday_features(self, symbol, daily_volumes):
            return {"vwap": 101.0, "bid": 100.9, "ask": 101.1, "relative_volume": 1.4}

    agent.broker = _Broker()
    market = MarketSnapshot(symbol="AAPL", asset_class=AssetClass.EQUITY, price=101, closes=[], metadata={"volumes": [1, 2]})
    await agent._enrich_intraday(market)
    assert market.metadata["vwap"] == 101.0
    assert market.metadata["relative_volume"] == 1.4


@pytest.mark.anyio
async def test_enrich_intraday_noop_when_disabled_or_crypto() -> None:
    agent = TradingAgent.__new__(TradingAgent)
    agent.console = _SilentConsole()

    class _Broker:
        async def get_intraday_features(self, symbol, daily_volumes):
            raise AssertionError("should not be called")

    agent.broker = _Broker()
    # day trading disabled -> no call
    agent.settings = Settings.model_validate({"day_trading": {"enabled": False}})
    crypto = MarketSnapshot(symbol="AAPL", asset_class=AssetClass.EQUITY, price=1, closes=[], metadata={})
    await agent._enrich_intraday(crypto)
    # enabled but crypto asset -> no call
    agent.settings = Settings.model_validate({"day_trading": {"enabled": True}})
    btc = MarketSnapshot(symbol="BTC/USD", asset_class=AssetClass.CRYPTO, price=1, closes=[], metadata={})
    await agent._enrich_intraday(btc)


# --- Fix 3: regime gate exempts day trades ----------------------------------
def _candidate(symbol, asset_class, strategy, day_trade=False) -> TradeCandidate:
    return TradeCandidate(
        symbol=symbol,
        asset_class=asset_class,
        side=OrderSide.BUY,
        strategy=strategy,
        score=85,
        entry_price=100,
        metadata={"day_trade": True} if day_trade else {},
    )


def test_regime_filter_keeps_crypto_and_day_trades_drops_swing() -> None:
    agent = TradingAgent.__new__(TradingAgent)
    agent.settings = Settings()
    cands = [
        _candidate("AAPL", AssetClass.EQUITY, "equity_momentum"),
        _candidate("BTC/USD", AssetClass.CRYPTO, "crypto_momentum"),
        _candidate("MSFT", AssetClass.EQUITY, "day_trade_entry", day_trade=True),
    ]
    kept = {c.symbol for c in agent._apply_regime_filter(cands, equity_entries_allowed=False)}
    assert kept == {"BTC/USD", "MSFT"}


def test_regime_filter_can_apply_to_day_trades_when_configured() -> None:
    agent = TradingAgent.__new__(TradingAgent)
    agent.settings = Settings.model_validate({"regime": {"apply_to_day_trades": True}})
    cands = [
        _candidate("BTC/USD", AssetClass.CRYPTO, "crypto_momentum"),
        _candidate("MSFT", AssetClass.EQUITY, "day_trade_entry", day_trade=True),
    ]
    kept = {c.symbol for c in agent._apply_regime_filter(cands, equity_entries_allowed=False)}
    assert kept == {"BTC/USD"}


def test_regime_filter_passthrough_when_allowed() -> None:
    agent = TradingAgent.__new__(TradingAgent)
    agent.settings = Settings()
    cands = [_candidate("AAPL", AssetClass.EQUITY, "equity_momentum")]
    assert agent._apply_regime_filter(cands, equity_entries_allowed=True) == cands


# --- Fix 4: position manager skips day-trade positions ----------------------
def test_position_manager_skips_day_trade_symbols() -> None:
    manager = PositionManager(Settings())
    positions = [
        Position(
            symbol="AAPL",
            asset_class=AssetClass.EQUITY,
            qty=10,
            market_value=9_300,
            avg_entry_price=100,
            current_price=93,  # -7% -> would trigger a swing stop-loss exit
            unrealized_pl=-70,
        )
    ]
    assert len(manager.evaluate(positions)) == 1
    assert manager.evaluate(positions, skip_symbols={"AAPL"}) == []


# --- daily-loss halt is day-trade-aware -------------------------------------
def _dt_candidate(pl_pct: float) -> TradeCandidate:
    return TradeCandidate(
        symbol="AAPL",
        asset_class=AssetClass.EQUITY,
        side=OrderSide.BUY,
        strategy="day_trade_entry",
        score=90,
        entry_price=100,
        stop_price=99,
        metadata={"day_trade": True, "day_trade_daily_pl_pct": pl_pct},
    )


def test_day_trade_uses_its_own_daily_pl_not_account() -> None:
    engine = RiskEngine(Settings.model_validate({"day_trading": {"enabled": True}}))
    # Account is down ~4.3% on the day (swing would halt), but the day-trade book
    # is only down 0.5% -> the day trade is NOT halted by the swing drawdown.
    account = AccountSnapshot(equity=100_000, cash=50_000, buying_power=50_000, last_equity=104_500)
    decision = engine.evaluate(_dt_candidate(-0.5), account, [])
    assert decision.approved


def test_day_trade_halts_on_its_own_daily_loss() -> None:
    engine = RiskEngine(Settings.model_validate({"day_trading": {"enabled": True}}))
    account = AccountSnapshot(equity=100_000, cash=50_000, buying_power=50_000, last_equity=100_000)
    decision = engine.evaluate(_dt_candidate(-1.5), account, [])  # past the 1% day-trade limit
    assert not decision.approved
    assert "Daily loss stop" in decision.reason


def test_apply_halt_filter_swing_keeps_day_trade() -> None:
    agent = TradingAgent.__new__(TradingAgent)
    swing = _candidate("AAPL", AssetClass.EQUITY, "equity_momentum")
    dt = _candidate("MSFT", AssetClass.EQUITY, "day_trade_entry", day_trade=True)
    kept = agent._apply_halt_filter([swing, dt], swing_halt=True, day_trade_halt=False)
    assert {c.symbol for c in kept} == {"MSFT"}
    kept2 = agent._apply_halt_filter([swing, dt], swing_halt=False, day_trade_halt=True)
    assert {c.symbol for c in kept2} == {"AAPL"}
    assert len(agent._apply_halt_filter([swing, dt], False, False)) == 2


def test_day_trade_daily_pl_pct_combines_open_and_realized(tmp_path) -> None:
    agent = TradingAgent.__new__(TradingAgent)
    agent.settings = Settings.model_validate({"day_trading": {"enabled": True}})
    agent.audit = AuditStore(tmp_path / "a.sqlite3")
    agent.audit.record_event(
        cycle_id=None,
        event_type="order",
        payload={"intent": {"symbol": "MSFT", "metadata": {"day_trade": True, "day_trade_entry": True}}},
        symbol="MSFT",
        strategy="day_trade_entry",
        status="submitted",
    )
    agent.audit.record_event(
        cycle_id=None,
        event_type="order",
        payload={"intent": {"metadata": {"exit": True, "day_trade": True, "position": {"unrealized_pl": -200}}}},
        symbol="AAPL",
        strategy="day_trade_exit",
        status="submitted",
    )
    account = AccountSnapshot(equity=100_000, cash=0, buying_power=0, last_equity=100_000)
    positions = [Position(symbol="MSFT", asset_class=AssetClass.EQUITY, qty=10, market_value=1_000, unrealized_pl=100)]
    # open MSFT +100 plus realized AAPL -200 = -100 -> -0.1% of 100k.
    assert round(agent._day_trade_daily_pl_pct(account, positions), 4) == -0.1


@pytest.mark.anyio
async def test_swing_halt_preserves_working_day_trade_orders() -> None:
    class _Broker:
        def __init__(self):
            self.canceled: list[str] = []

        async def get_open_orders(self):
            return [
                {"id": "s1", "symbol": "AAPL", "side": "buy", "type": "limit",
                 "client_order_id": "ta-equity_momentum-x", "position_intent": "buy_to_open"},
                {"id": "d1", "symbol": "MSFT", "side": "buy", "type": "limit",
                 "client_order_id": "ta-day_trade_entry-y", "position_intent": "buy_to_open"},
            ]

        async def cancel_order(self, order_id):
            self.canceled.append(order_id)

    agent = TradingAgent.__new__(TradingAgent)
    agent.console = _SilentConsole()
    agent.audit = None
    agent.broker = _Broker()
    await agent._cancel_opening_orders_for_halt(None, cancel_swing=True, cancel_day_trade=False)
    assert agent.broker.canceled == ["s1"]  # swing entry canceled, day-trade entry preserved
