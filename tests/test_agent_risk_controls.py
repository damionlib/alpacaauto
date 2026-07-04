import pytest

from trading_agent.agent import TradingAgent
from trading_agent.audit import AuditStore
from trading_agent.config import Settings
from trading_agent.models import (
    AccountSnapshot,
    AssetClass,
    OrderIntent,
    OrderSide,
    OrderType,
    Position,
    RiskDecision,
    TimeInForce,
    TradeCandidate,
)
from trading_agent.position_manager import PositionManager
from trading_agent.risk import RiskEngine


@pytest.fixture(autouse=True)
def _fast_exit_retry(monkeypatch):
    # The exit-retry loop sleeps between attempts to let broker-side cancels
    # settle; tests should not pay that wall-clock cost.
    monkeypatch.setattr("trading_agent.agent._EXIT_RETRY_DELAY_SECONDS", 0)


class SilentConsole:
    def print(self, *args, **kwargs) -> None:  # noqa: D401 - test stub
        return None


class FakeBroker:
    def __init__(self, *, open_orders=None, submit_result=None) -> None:
        self._open_orders = open_orders or []
        self._submit_result = submit_result or {}
        self.submitted: list[OrderIntent] = []
        self.canceled: list[str] = []
        self.calls: list[str] = []
        self.cancel_all_called = False
        self.account: AccountSnapshot | None = None
        self.positions: list[Position] = []

    async def get_open_orders(self):
        return self._open_orders

    async def submit_order(self, intent: OrderIntent):
        self.calls.append(f"submit:{intent.symbol}")
        self.submitted.append(intent)
        result = {"id": f"ord-{len(self.submitted)}", "status": "accepted"}
        result.update(self._submit_result)
        return result

    async def cancel_order(self, order_id: str):
        self.calls.append(f"cancel:{order_id}")
        self.canceled.append(order_id)

    async def cancel_all_orders(self):
        self.cancel_all_called = True
        return []

    async def get_clock(self):
        return {"is_open": True, "next_close": "2026-01-01T20:00:00Z"}

    async def get_account(self):
        return self.account

    async def get_positions(self):
        return self.positions


def _agent(broker: FakeBroker, settings: Settings | None = None) -> TradingAgent:
    agent = TradingAgent.__new__(TradingAgent)
    agent.settings = settings or Settings()
    agent.console = SilentConsole()
    agent.audit = None
    agent.broker = broker
    return agent


def _entry_decision(
    symbol: str,
    *,
    qty: float = 30,
    limit_price: float = 100.0,
    asset_class: AssetClass = AssetClass.EQUITY,
    strategy: str = "equity_momentum",
    stop_price: float | None = None,
    notional: float | None = None,
    time_in_force: TimeInForce = TimeInForce.DAY,
) -> RiskDecision:
    candidate = TradeCandidate(
        symbol=symbol,
        asset_class=asset_class,
        side=OrderSide.BUY,
        strategy=strategy,
        score=90,
        entry_price=limit_price,
        stop_price=stop_price,
    )
    intent = OrderIntent(
        symbol=symbol,
        asset_class=asset_class,
        side=OrderSide.BUY,
        qty=None if notional is not None else qty,
        notional=notional,
        order_type=OrderType.LIMIT,
        time_in_force=time_in_force,
        limit_price=limit_price,
    )
    return RiskDecision(approved=True, reason="ok", intent=intent, candidate=candidate)


def _exit_decision(symbol: str = "AAPL", qty: int = 10) -> RiskDecision:
    candidate = TradeCandidate(
        symbol=symbol,
        asset_class=AssetClass.EQUITY,
        side=OrderSide.SELL,
        strategy="stop_loss_exit",
        score=100,
        entry_price=90,
        metadata={"exit": True, "exit_qty": qty},
    )
    intent = OrderIntent(
        symbol=symbol,
        asset_class=AssetClass.EQUITY,
        side=OrderSide.SELL,
        qty=qty,
        metadata=candidate.metadata,
    )
    return RiskDecision(approved=True, reason="exit", intent=intent, candidate=candidate)


# --- #4: marketable-limit entries -------------------------------------------------


def test_equity_entry_is_marketable_limit_order() -> None:
    engine = RiskEngine(Settings())
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
    assert decision.intent.order_type == OrderType.LIMIT
    assert decision.intent.limit_price == 100.5  # entry * (1 + 0.5% slippage)
    assert decision.intent.qty == 120
    # Bracket protection still carried for the broker side.
    assert decision.intent.stop_loss_price == 95


def test_crypto_entry_is_quantity_limit_not_notional() -> None:
    engine = RiskEngine(Settings())
    decision = engine.evaluate(
        TradeCandidate(
            symbol="BTC/USD",
            asset_class=AssetClass.CRYPTO,
            side=OrderSide.BUY,
            strategy="crypto_momentum",
            score=90,
            entry_price=100,
        ),
        AccountSnapshot(equity=100_000, cash=100_000, buying_power=100_000, last_equity=100_000),
        [],
    )
    assert decision.approved
    assert decision.intent is not None
    assert decision.intent.order_type == OrderType.LIMIT
    assert decision.intent.time_in_force == TimeInForce.GTC
    assert decision.intent.notional is None
    assert decision.intent.qty == 20.0
    assert decision.intent.limit_price == 100.5


# --- #2: cycle-aggregate cash budget ----------------------------------------------


def test_estimated_cash_requirement_by_order_kind() -> None:
    agent = _agent(FakeBroker())
    assert agent._estimated_cash_requirement(_entry_decision("AAPL", qty=30, limit_price=100)) == 3000.0
    assert agent._estimated_cash_requirement(_exit_decision()) == 0.0

    csp = RiskDecision(
        approved=True,
        reason="ok",
        intent=OrderIntent(symbol="AAPL...P", asset_class=AssetClass.OPTION, side=OrderSide.SELL, qty=1),
        candidate=TradeCandidate(
            symbol="AAPL...P",
            asset_class=AssetClass.OPTION,
            side=OrderSide.SELL,
            strategy="cash_secured_put",
            score=70,
            entry_price=1.0,
            metadata={"contract": {"strike_price": "200"}},
        ),
    )
    assert agent._estimated_cash_requirement(csp) == 200 * 100


@pytest.mark.anyio
async def test_cycle_cash_budget_blocks_second_overrunning_entry() -> None:
    broker = FakeBroker()
    agent = _agent(broker)
    account = AccountSnapshot(equity=100_000, cash=10_000, buying_power=10_000, last_equity=100_000)
    # Budget = min(cash, bp) - 5% buffer = 10_000 - 5_000 = 5_000.
    # Each order needs 3_000, so only the first one fits.
    decisions = [
        _entry_decision("AAA", qty=30, limit_price=100),
        _entry_decision("BBB", qty=30, limit_price=100),
    ]

    await agent._submit_decisions(decisions, account, None)

    assert len(broker.submitted) == 1
    assert broker.submitted[0].symbol == "AAA"


# --- #3: crypto broker-side protective stop & exit supersedes resting orders ------


@pytest.mark.anyio
async def test_crypto_entry_places_protective_stop_limit() -> None:
    broker = FakeBroker(submit_result={"status": "filled", "filled_qty": "2.0"})
    agent = _agent(broker)
    account = AccountSnapshot(equity=100_000, cash=100_000, buying_power=100_000, last_equity=100_000)
    decision = _entry_decision(
        "BTC/USD",
        qty=2.0,
        limit_price=100,
        asset_class=AssetClass.CRYPTO,
        strategy="crypto_momentum",
        stop_price=95,
        time_in_force=TimeInForce.GTC,
    )

    await agent._submit_decisions([decision], account, None)

    assert len(broker.submitted) == 2
    entry, protective = broker.submitted
    assert entry.side == OrderSide.BUY
    assert protective.side == OrderSide.SELL
    assert protective.order_type == OrderType.STOP_LIMIT
    assert protective.time_in_force == TimeInForce.GTC
    assert protective.stop_price == 95
    assert protective.qty == 2.0


@pytest.mark.anyio
async def test_crypto_protective_stop_waits_for_entry_fill() -> None:
    broker = FakeBroker(submit_result={"status": "accepted", "filled_qty": "0"})
    agent = _agent(broker)
    account = AccountSnapshot(equity=100_000, cash=100_000, buying_power=100_000, last_equity=100_000)
    decision = _entry_decision(
        "BTC/USD",
        qty=2.0,
        limit_price=100,
        asset_class=AssetClass.CRYPTO,
        strategy="crypto_momentum",
        stop_price=95,
        time_in_force=TimeInForce.GTC,
    )

    await agent._submit_decisions([decision], account, None)

    assert len(broker.submitted) == 1
    assert broker.submitted[0].side == OrderSide.BUY


@pytest.mark.anyio
async def test_crypto_protective_stop_is_placed_after_fill_sync(tmp_path) -> None:
    broker = FakeBroker()
    broker.positions = [
        Position(symbol="BTC/USD", asset_class=AssetClass.CRYPTO, qty=2, market_value=200)
    ]
    agent = _agent(broker)
    agent.audit = AuditStore(tmp_path / "audit.sqlite3")
    client_order_id = "ta-crypto_momentum-fill1"
    agent.audit.record_event(
        cycle_id=None,
        event_type="order",
        payload={
            "intent": OrderIntent(
                symbol="BTC/USD",
                asset_class=AssetClass.CRYPTO,
                side=OrderSide.BUY,
                qty=2,
                order_type=OrderType.LIMIT,
                limit_price=100,
                stop_loss_price=95,
                client_order_id=client_order_id,
            ),
            "broker_order": {"id": "entry-1"},
        },
        symbol="BTC/USD",
        strategy="crypto_momentum",
        status="submitted",
    )

    await agent._place_crypto_protective_stop_from_fill(
        {
            "id": "entry-1",
            "symbol": "BTC/USD",
            "asset_class": "crypto",
            "side": "buy",
            "filled_qty": "2",
            "client_order_id": client_order_id,
        },
        None,
    )

    assert len(broker.submitted) == 1
    protective = broker.submitted[0]
    assert protective.order_type == OrderType.STOP_LIMIT
    assert protective.side == OrderSide.SELL
    assert protective.qty == 2
    assert protective.stop_price == 95


@pytest.mark.anyio
async def test_crypto_fill_sync_skips_stop_when_position_is_already_closed(tmp_path) -> None:
    broker = FakeBroker()
    agent = _agent(broker)
    agent.audit = AuditStore(tmp_path / "audit.sqlite3")
    client_order_id = "ta-crypto_momentum-closed"
    agent.audit.record_event(
        cycle_id=None,
        event_type="order",
        payload={
            "intent": OrderIntent(
                symbol="BTC/USD",
                asset_class=AssetClass.CRYPTO,
                side=OrderSide.BUY,
                qty=2,
                order_type=OrderType.LIMIT,
                limit_price=100,
                stop_loss_price=95,
                client_order_id=client_order_id,
            )
        },
        symbol="BTC/USD",
        strategy="crypto_momentum",
        status="submitted",
    )

    await agent._place_crypto_protective_stop_from_fill(
        {
            "id": "entry-1",
            "symbol": "BTC/USD",
            "asset_class": "crypto",
            "side": "buy",
            "filled_qty": "2",
            "client_order_id": client_order_id,
        },
        None,
    )

    assert broker.submitted == []


@pytest.mark.anyio
async def test_exit_cancels_conflicting_open_order_then_submits() -> None:
    broker = FakeBroker(open_orders=[{"id": "x1", "symbol": "AAPL", "side": "sell", "qty": "10"}])
    agent = _agent(broker)
    account = AccountSnapshot(equity=100_000, cash=50_000, buying_power=50_000, last_equity=100_000)

    await agent._submit_decisions([_exit_decision("AAPL", qty=10)], account, None)

    assert broker.canceled == ["x1"]
    assert len(broker.submitted) == 1
    assert broker.submitted[0].side == OrderSide.SELL
    assert broker.calls == ["submit:AAPL", "cancel:x1"]
    assert broker.cancel_all_called is False


# --- #1: daily loss stop manages exits instead of wiping protective orders --------


@pytest.mark.anyio
async def test_daily_loss_stop_manages_exits_without_canceling_all() -> None:
    settings = Settings()
    broker = FakeBroker(
        open_orders=[
            {
                "id": "entry-1",
                "symbol": "MSFT",
                "side": "buy",
                "type": "limit",
                "client_order_id": "ta-equity_momentum-abc",
                "position_intent": "buy_to_open",
            },
            {
                "id": "protect-1",
                "symbol": "TSLA",
                "side": "sell",
                "type": "stop_limit",
                "client_order_id": "ta-crypto-protect-def",
                "position_intent": "sell_to_close",
            },
        ]
    )
    broker.account = AccountSnapshot(
        equity=96_000, cash=20_000, buying_power=20_000, last_equity=100_000
    )  # -4% day, past the 3% stop
    broker.positions = [
        Position(
            symbol="AAPL",
            asset_class=AssetClass.EQUITY,
            qty=10,
            market_value=9_000,
            avg_entry_price=100,
            current_price=90,  # -10% -> triggers position-manager stop loss
            unrealized_pl=-100,
        )
    ]
    agent = _agent(broker, settings)
    agent.broker_sync = None
    agent.position_manager = PositionManager(settings, None)
    agent.risk = RiskEngine(settings)

    decisions = await agent.run_once()

    assert broker.cancel_all_called is False
    assert "entry-1" in broker.canceled
    assert "protect-1" not in broker.canceled
    assert len(broker.submitted) == 1
    assert broker.submitted[0].side == OrderSide.SELL
    assert any(decision.candidate.metadata.get("exit") for decision in decisions)


# --- exit retry path: conflict-gated cancel + re-arm on retry failure -------------


class ScriptedBroker(FakeBroker):
    """FakeBroker whose submit_order can raise a scripted sequence of errors."""

    def __init__(self, *, submit_outcomes=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._submit_outcomes = list(submit_outcomes or [])

    async def submit_order(self, intent: OrderIntent):
        if self._submit_outcomes:
            outcome = self._submit_outcomes.pop(0)
            if isinstance(outcome, Exception):
                self.calls.append(f"submit-fail:{intent.symbol}")
                raise outcome
        return await super().submit_order(intent)


def _crypto_exit_decision(symbol: str = "BTC/USD", qty: float = 2.0) -> RiskDecision:
    candidate = TradeCandidate(
        symbol=symbol,
        asset_class=AssetClass.CRYPTO,
        side=OrderSide.SELL,
        strategy="stop_loss_exit",
        score=100,
        entry_price=90,
        metadata={"exit": True, "exit_qty": qty},
    )
    intent = OrderIntent(
        symbol=symbol,
        asset_class=AssetClass.CRYPTO,
        side=OrderSide.SELL,
        qty=qty,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.GTC,
        metadata=candidate.metadata,
    )
    return RiskDecision(approved=True, reason="exit", intent=intent, candidate=candidate)


def _spread_exit_decision() -> RiskDecision:
    legs = [
        {
            "symbol": "CDNS260618C00410000",
            "ratio_qty": "1",
            "side": "sell",
            "position_intent": "sell_to_close",
        },
        {
            "symbol": "CDNS260618C00430000",
            "ratio_qty": "1",
            "side": "buy",
            "position_intent": "buy_to_close",
        },
    ]
    candidate = TradeCandidate(
        symbol="CDNS_call_debit_spread",
        asset_class=AssetClass.OPTION,
        side=OrderSide.SELL,
        strategy="spread_stop_loss_exit",
        score=100,
        entry_price=6.4,
        metadata={
            "exit": True,
            "spread_exit": True,
            "exit_qty": 1,
            "legs": legs,
            "paired_symbols": ["CDNS260618C00410000", "CDNS260618C00430000"],
        },
    )
    intent = OrderIntent(
        symbol="CDNS_call_debit_spread",
        asset_class=AssetClass.OPTION,
        side=OrderSide.SELL,
        qty=1,
        order_type=OrderType.LIMIT,
        limit_price=6.4,
        order_class="mleg",
        legs=legs,
        metadata=candidate.metadata,
    )
    return RiskDecision(approved=True, reason="exit", intent=intent, candidate=candidate)


_CONFLICT = RuntimeError("POST /v2/orders failed with 403: 40310000: insufficient qty available")
_TRANSIENT = RuntimeError("POST /v2/orders failed with 503: service temporarily unavailable")


@pytest.mark.anyio
async def test_exit_conflict_error_cancels_then_retries_successfully() -> None:
    broker = ScriptedBroker(
        open_orders=[{"id": "x1", "symbol": "AAPL", "side": "sell", "qty": "10"}],
        submit_outcomes=[_CONFLICT],  # first exit submit fails on a qty conflict
    )
    agent = _agent(broker)
    account = AccountSnapshot(equity=100_000, cash=50_000, buying_power=50_000, last_equity=100_000)

    await agent._submit_decisions([_exit_decision("AAPL", qty=10)], account, None)

    # Conflict -> cancel resting order -> retry -> success.
    assert broker.canceled == ["x1"]
    assert broker.calls == ["submit-fail:AAPL", "cancel:x1", "submit:AAPL"]
    assert len(broker.submitted) == 1
    assert broker.cancel_all_called is False


@pytest.mark.anyio
async def test_spread_exit_conflict_cancels_parent_mleg_order_then_retries() -> None:
    stale_opening_spread = {
        "id": "mleg-1",
        "symbol": "",
        "client_order_id": "ta-call_debit_spread-stale",
        "order_class": "mleg",
        "qty": "1",
        "legs": [
            {
                "symbol": "CDNS260618C00410000",
                "side": "buy",
                "ratio_qty": "1",
                "position_intent": "buy_to_open",
            },
            {
                "symbol": "CDNS260618C00430000",
                "side": "sell",
                "ratio_qty": "1",
                "position_intent": "sell_to_open",
            },
        ],
    }
    broker = ScriptedBroker(
        open_orders=[stale_opening_spread],
        submit_outcomes=[RuntimeError("POST /v2/orders failed with 403: potential wash trade detected")],
    )
    agent = _agent(broker)
    account = AccountSnapshot(equity=100_000, cash=50_000, buying_power=50_000, last_equity=100_000)

    await agent._submit_decisions([_spread_exit_decision()], account, None)

    assert broker.canceled == ["mleg-1"]
    assert broker.calls == [
        "submit-fail:CDNS_call_debit_spread",
        "cancel:mleg-1",
        "submit:CDNS_call_debit_spread",
    ]
    assert len(broker.submitted) == 1
    assert broker.submitted[0].order_class == "mleg"


@pytest.mark.anyio
async def test_exit_transient_error_preserves_protective_orders() -> None:
    broker = ScriptedBroker(
        open_orders=[{"id": "x1", "symbol": "AAPL", "side": "sell", "qty": "10"}],
        submit_outcomes=[_TRANSIENT],  # not a conflict -> must NOT cancel anything
    )
    agent = _agent(broker)
    account = AccountSnapshot(equity=100_000, cash=50_000, buying_power=50_000, last_equity=100_000)

    await agent._submit_decisions([_exit_decision("AAPL", qty=10)], account, None)

    # Protective/resting order is left in place; nothing was canceled or submitted.
    assert broker.canceled == []
    assert broker.submitted == []
    assert broker.calls == ["submit-fail:AAPL"]
    assert broker.cancel_all_called is False


@pytest.mark.anyio
async def test_exit_retry_failure_rearms_crypto_protective_stop() -> None:
    protective = {
        "id": "p1",
        "symbol": "BTC/USD",
        "side": "sell",
        "type": "stop_limit",
        "qty": "2",
        "stop_price": "95",
        "client_order_id": "ta-crypto-protect-abc",
    }
    broker = ScriptedBroker(
        open_orders=[protective],
        submit_outcomes=[_CONFLICT] * 4,  # exit fails, all retries fail too
    )
    agent = _agent(broker)
    account = AccountSnapshot(equity=100_000, cash=100_000, buying_power=100_000, last_equity=100_000)

    await agent._submit_decisions([_crypto_exit_decision("BTC/USD", qty=2.0)], account, None)

    # The protective stop was canceled for the retry, then restored after it failed.
    assert broker.canceled == ["p1"]
    assert broker.calls == [
        "submit-fail:BTC/USD",
        "cancel:p1",
        *["submit-fail:BTC/USD"] * 3,
        "submit:BTC/USD",
    ]
    assert len(broker.submitted) == 1
    restored = broker.submitted[0]
    assert restored.order_type == OrderType.STOP_LIMIT
    assert restored.side == OrderSide.SELL
    assert restored.qty == 2.0
    assert restored.stop_price == 95


@pytest.mark.anyio
async def test_exit_retry_failure_rearms_equity_bracket_as_oco() -> None:
    stop_leg = {
        "id": "sl",
        "symbol": "AAPL",
        "side": "sell",
        "type": "stop",
        "qty": "10",
        "stop_price": "94",
    }
    take_profit_leg = {
        "id": "tp",
        "symbol": "AAPL",
        "side": "sell",
        "type": "limit",
        "qty": "10",
        "limit_price": "112",
    }
    broker = ScriptedBroker(
        open_orders=[stop_leg, take_profit_leg],
        submit_outcomes=[_CONFLICT] * 4,  # exit fails, all retries fail too
    )
    agent = _agent(broker)
    account = AccountSnapshot(equity=100_000, cash=50_000, buying_power=50_000, last_equity=100_000)

    await agent._submit_decisions([_exit_decision("AAPL", qty=10)], account, None)

    # Both bracket legs were canceled for the retry, then rebuilt as one OCO.
    assert broker.canceled == ["sl", "tp"]
    assert len(broker.submitted) == 1
    restored = broker.submitted[0]
    assert restored.order_class == "oco"
    assert restored.side == OrderSide.SELL
    assert restored.qty == 10
    assert restored.stop_loss_price == 94
    assert restored.take_profit_price == 112


@pytest.mark.anyio
async def test_exit_retry_does_not_rearm_unfilled_entry_order() -> None:
    # A resting BUY entry on the same symbol must not be "restored" as protection.
    entry_order = {
        "id": "e1",
        "symbol": "AAPL",
        "side": "buy",
        "type": "limit",
        "qty": "5",
        "limit_price": "101",
        "client_order_id": "ta-equity_momentum-xyz",
        "position_intent": "buy_to_open",
    }
    broker = ScriptedBroker(
        open_orders=[entry_order],
        submit_outcomes=[_CONFLICT] * 4,
    )
    agent = _agent(broker)
    account = AccountSnapshot(equity=100_000, cash=50_000, buying_power=50_000, last_equity=100_000)

    await agent._submit_decisions([_exit_decision("AAPL", qty=10)], account, None)

    assert broker.canceled == ["e1"]  # canceled to clear the conflict
    assert broker.submitted == []  # but NOT re-armed as protection
