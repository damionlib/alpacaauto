from datetime import UTC, datetime

import pytest

from trading_agent.agent import TradingAgent
from trading_agent.config import Settings
from trading_agent.models import (
    AccountSnapshot,
    AssetClass,
    OrderIntent,
    OrderSide,
    OrderType,
    RiskDecision,
    TradeCandidate,
)


class _Silent:
    def print(self, *a, **k) -> None:
        return None


def _agent(settings: Settings | None = None) -> TradingAgent:
    agent = TradingAgent.__new__(TradingAgent)
    agent.settings = settings or Settings()
    agent.console = _Silent()
    agent.audit = None
    return agent


# Summer (EDT) -> ET = UTC-4. 9:30 ET = 13:30 UTC; 16:00 ET = 20:00 UTC.
def _utc(h, m):
    return datetime(2026, 6, 8, h, m, tzinfo=UTC)


CLOSE_2000 = "2026-06-08T20:00:00Z"


def test_options_window_blocked_when_market_closed() -> None:
    ok, info = _agent()._evaluate_options_window({"is_open": False}, _utc(2, 0))
    assert ok is False and "closed" in info["note"]


def test_options_window_blocked_in_open_buffer() -> None:
    # 9:35 ET, default 15m open buffer -> blocked until 9:45 ET.
    ok, info = _agent()._evaluate_options_window(
        {"is_open": True, "next_close": CLOSE_2000}, _utc(13, 35)
    )
    assert ok is False and "open" in info["note"]


def test_options_window_open_midsession() -> None:
    ok, _ = _agent()._evaluate_options_window(
        {"is_open": True, "next_close": CLOSE_2000}, _utc(14, 30)  # 10:30 ET
    )
    assert ok is True


def test_options_window_blocked_in_close_buffer() -> None:
    # 15:55 ET, default 10m close buffer vs 16:00 close -> blocked.
    ok, info = _agent()._evaluate_options_window(
        {"is_open": True, "next_close": CLOSE_2000}, _utc(19, 55)
    )
    assert ok is False and "close" in info["note"]


@pytest.mark.anyio
async def test_options_window_disabled_passes_without_clock() -> None:
    agent = _agent(Settings.model_validate({"execution": {"market_hours_only_options": False}}))

    class _Broker:
        async def get_clock(self):
            raise AssertionError("clock should not be fetched when gate disabled")

    agent.broker = _Broker()
    ok, _ = await agent._options_window()
    assert ok is True


@pytest.mark.anyio
async def test_options_window_fails_closed_when_clock_errors() -> None:
    agent = _agent()

    class _Broker:
        async def get_clock(self):
            raise RuntimeError("clock down")

    agent.broker = _Broker()
    ok, info = await agent._options_window()
    assert ok is False and "Clock unavailable" in info["note"]


def test_option_spread_too_wide() -> None:
    agent = _agent()  # max_option_spread_pct default 25
    assert agent._option_spread_too_wide(1.00, 1.10) is False  # ~9.5%
    assert agent._option_spread_too_wide(1.00, 1.60) is True  # ~46%
    assert agent._option_spread_too_wide(0, 1.0) is True  # missing bid
    assert agent._option_spread_too_wide(None, None) is True


# --- submission gate --------------------------------------------------------
class _FakeBroker:
    def __init__(self) -> None:
        self.submitted: list[OrderIntent] = []

    async def get_open_orders(self):
        return []

    async def submit_order(self, intent: OrderIntent):
        self.submitted.append(intent)
        return {"id": f"o{len(self.submitted)}", "status": "accepted"}

    async def cancel_order(self, order_id):
        return None


def _equity_buy() -> RiskDecision:
    c = TradeCandidate(
        symbol="MSFT", asset_class=AssetClass.EQUITY, side=OrderSide.BUY,
        strategy="equity_momentum", score=80, entry_price=100,
    )
    i = OrderIntent(
        symbol="MSFT", asset_class=AssetClass.EQUITY, side=OrderSide.BUY,
        qty=10, order_type=OrderType.LIMIT, limit_price=100,
    )
    return RiskDecision(approved=True, reason="ok", intent=i, candidate=c)


def _option_buy() -> RiskDecision:
    c = TradeCandidate(
        symbol="MSFT260101C00400000", asset_class=AssetClass.OPTION, side=OrderSide.BUY,
        strategy="long_call", score=80, entry_price=2.0,
    )
    i = OrderIntent(
        symbol="MSFT260101C00400000", asset_class=AssetClass.OPTION, side=OrderSide.BUY,
        qty=1, order_type=OrderType.LIMIT, limit_price=2.0,
    )
    return RiskDecision(approved=True, reason="ok", intent=i, candidate=c)


@pytest.mark.anyio
async def test_submit_defers_options_outside_window_but_keeps_equity() -> None:
    agent = _agent()
    agent.broker = _FakeBroker()
    account = AccountSnapshot(equity=100_000, cash=50_000, buying_power=50_000, last_equity=100_000)

    await agent._submit_decisions(
        [_option_buy(), _equity_buy()], account, None, options_window_ok=False
    )

    submitted = {i.symbol for i in agent.broker.submitted}
    assert submitted == {"MSFT"}  # the option was deferred, the equity went through


@pytest.mark.anyio
async def test_submit_allows_options_inside_window() -> None:
    agent = _agent()
    agent.broker = _FakeBroker()
    account = AccountSnapshot(equity=100_000, cash=50_000, buying_power=50_000, last_equity=100_000)

    await agent._submit_decisions([_option_buy()], account, None, options_window_ok=True)
    assert any(i.symbol == "MSFT260101C00400000" for i in agent.broker.submitted)


# --- market-closed gate (weekend churn prevention) -------------------------

def _crypto_buy() -> RiskDecision:
    c = TradeCandidate(
        symbol="BTC/USD", asset_class=AssetClass.CRYPTO, side=OrderSide.BUY,
        strategy="crypto_momentum", score=80, entry_price=60000,
    )
    i = OrderIntent(
        symbol="BTC/USD", asset_class=AssetClass.CRYPTO, side=OrderSide.BUY,
        qty=0.1, order_type=OrderType.LIMIT, limit_price=60000,
    )
    return RiskDecision(approved=True, reason="ok", intent=i, candidate=c)


@pytest.mark.anyio
async def test_market_closed_blocks_equity_but_allows_crypto() -> None:
    agent = _agent()
    agent.broker = _FakeBroker()
    account = AccountSnapshot(equity=100_000, cash=50_000, buying_power=50_000, last_equity=100_000)

    await agent._submit_decisions(
        [_equity_buy(), _crypto_buy()], account, None, market_open=False
    )

    submitted = {i.symbol for i in agent.broker.submitted}
    assert "MSFT" not in submitted
    assert "BTC/USD" in submitted


@pytest.mark.anyio
async def test_market_open_allows_equity() -> None:
    agent = _agent()
    agent.broker = _FakeBroker()
    account = AccountSnapshot(equity=100_000, cash=50_000, buying_power=50_000, last_equity=100_000)

    await agent._submit_decisions(
        [_equity_buy()], account, None, market_open=True
    )

    assert any(i.symbol == "MSFT" for i in agent.broker.submitted)
