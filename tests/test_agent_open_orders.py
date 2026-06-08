from trading_agent.agent import TradingAgent
from trading_agent.config import Settings
from trading_agent.models import (
    AssetClass,
    OrderIntent,
    OrderSide,
    Position,
    RiskDecision,
    TradeCandidate,
)


class FakeAudit:
    def __init__(self, *, submitted=None, events=None) -> None:
        self.submitted = submitted or []
        self._events = events or []

    def submitted_orders_since(self, since):
        return self.submitted

    def events(self, **kwargs):
        return self._events


def test_parse_option_symbol_extracts_underlying_and_type() -> None:
    agent = TradingAgent.__new__(TradingAgent)

    parsed = agent._parse_option_symbol("AAPL260612C00322500")

    assert parsed == {
        "underlying": "AAPL",
        "expiration": "2026-06-12",
        "type": "C",
    }


def test_skip_duplicate_open_order_symbol() -> None:
    agent = TradingAgent.__new__(TradingAgent)
    decision = _covered_call_decision("AAPL260612C00322500")

    reason = agent._skip_due_to_open_orders(
        decision,
        {
            "symbols": {"AAPL260612C00322500"},
            "covered_call_contracts_by_underlying": {},
            "orders": [],
        },
    )

    assert reason == "Open order already exists for this symbol."


def test_skip_covered_call_when_open_orders_reserve_underlying_shares() -> None:
    agent = TradingAgent.__new__(TradingAgent)
    decision = _covered_call_decision("AAPL260612C00330000")

    reason = agent._skip_due_to_open_orders(
        decision,
        {
            "symbols": set(),
            "covered_call_contracts_by_underlying": {"AAPL": 2},
            "orders": [],
        },
    )

    assert "reserve 200 of 200 available AAPL shares" in reason


def _covered_call_decision(symbol: str) -> RiskDecision:
    candidate = TradeCandidate(
        symbol=symbol,
        asset_class=AssetClass.OPTION,
        side=OrderSide.SELL,
        strategy="covered_call",
        score=72,
        entry_price=1.0,
        metadata={"underlying": "AAPL", "contracts_per_100_shares": 2},
    )
    return RiskDecision(
        approved=True,
        reason="Approved option trade.",
        intent=OrderIntent(
            symbol=symbol,
            asset_class=AssetClass.OPTION,
            side=OrderSide.SELL,
            qty=1,
        ),
        candidate=candidate,
    )


def _spread_decision() -> RiskDecision:
    legs = [
        {
            "symbol": "CDNS260618C00410000",
            "ratio_qty": "1",
            "side": "buy",
            "position_intent": "buy_to_open",
        },
        {
            "symbol": "CDNS260618C00430000",
            "ratio_qty": "1",
            "side": "sell",
            "position_intent": "sell_to_open",
        },
    ]
    candidate = TradeCandidate(
        symbol="CDNS_call_debit_spread",
        asset_class=AssetClass.OPTION,
        side=OrderSide.BUY,
        strategy="call_debit_spread",
        score=72,
        entry_price=10.8,
        metadata={"underlying": "CDNS", "legs": legs},
    )
    return RiskDecision(
        approved=True,
        reason="Approved multi-leg option trade.",
        intent=OrderIntent(
            symbol="CDNS_call_debit_spread",
            asset_class=AssetClass.OPTION,
            side=OrderSide.BUY,
            qty=1,
            order_class="mleg",
            legs=legs,
            metadata=candidate.metadata,
        ),
        candidate=candidate,
    )


def _aapl_positions(short_calls: int) -> list[Position]:
    positions = [Position(symbol="AAPL", asset_class=AssetClass.EQUITY, qty=228, market_value=72_000)]
    if short_calls:
        positions.append(
            Position(
                symbol="AAPL260612C00322500",
                asset_class=AssetClass.OPTION,
                qty=-short_calls,
                market_value=-184 * short_calls,
            )
        )
    return positions


def test_skip_covered_call_when_existing_short_calls_use_coverage() -> None:
    agent = TradingAgent.__new__(TradingAgent)
    decision = _covered_call_decision("AAPL260612C00330000")

    # 228 shares cover 2 calls; 1 is already written and 1 order is still working,
    # so a third would overrun the coverage and create a naked call.
    reason = agent._skip_due_to_open_orders(
        decision,
        {"symbols": set(), "covered_call_contracts_by_underlying": {"AAPL": 1}, "orders": []},
        _aapl_positions(short_calls=1),
    )

    assert reason is not None
    assert "already written" in reason


def test_covered_call_allowed_when_coverage_remains_after_short_calls() -> None:
    agent = TradingAgent.__new__(TradingAgent)
    decision = _covered_call_decision("AAPL260612C00330000")

    # 228 shares cover 2 calls, only 1 written and none working -> room for one more.
    reason = agent._skip_due_to_open_orders(
        decision,
        {"symbols": set(), "covered_call_contracts_by_underlying": {}, "orders": []},
        _aapl_positions(short_calls=1),
    )

    assert reason is None


def test_existing_option_exposure_blocks_same_underlying_option_entries() -> None:
    agent = TradingAgent.__new__(TradingAgent)
    positions = [
        Position(
            symbol="CDNS260618C00400000",
            asset_class=AssetClass.OPTION,
            qty=1,
            market_value=1_960,
        ),
        Position(
            symbol="CDNS260618C00420000",
            asset_class=AssetClass.OPTION,
            qty=-1,
            market_value=-820,
        ),
    ]

    assert agent._has_open_option_exposure("CDNS", positions) is True
    assert agent._has_open_option_exposure("AAPL", positions) is False


def test_pending_option_order_blocks_same_underlying_option_entry() -> None:
    agent = TradingAgent.__new__(TradingAgent)
    agent.settings = Settings()
    agent.audit = None

    reason = agent._skip_due_to_open_orders(
        _spread_decision(),
        {
            "symbols": {"CDNS260618C00405000", "CDNS260618C00430000"},
            "option_underlyings": {"CDNS"},
            "covered_call_contracts_by_underlying": {},
            "orders": [],
        },
    )

    assert reason == "Open option order already exists for CDNS."


def test_daily_option_underlying_cap_blocks_repeat_entry() -> None:
    agent = TradingAgent.__new__(TradingAgent)
    agent.settings = Settings()
    agent.audit = FakeAudit(
        submitted=[
            {
                "payload": {
                    "intent": {
                        "asset_class": "option",
                        "symbol": "CDNS_call_debit_spread",
                        "metadata": {"underlying": "CDNS"},
                    }
                }
            }
        ]
    )

    reason = agent._skip_due_to_open_orders(
        _spread_decision(),
        {"symbols": set(), "option_underlyings": set(), "covered_call_contracts_by_underlying": {}, "orders": []},
    )

    assert reason == "Daily option entry cap reached for CDNS (1/1 already submitted today)."


def test_option_loss_cooldown_blocks_repeat_entry() -> None:
    agent = TradingAgent.__new__(TradingAgent)
    agent.settings = Settings()
    agent.audit = FakeAudit(
        events=[
            {
                "created_at": "2999-01-01T15:00:00+00:00",
                "payload": {
                    "candidate": {
                        "asset_class": "option",
                        "symbol": "CDNS_call_debit_spread",
                        "metadata": {
                            "exit": True,
                            "underlying": "CDNS",
                            "metrics": {"pnl_pct": -41.2},
                        },
                    }
                },
            }
        ]
    )

    reason = agent._skip_due_to_open_orders(
        _spread_decision(),
        {"symbols": set(), "option_underlyings": set(), "covered_call_contracts_by_underlying": {}, "orders": []},
    )

    assert reason is not None
    assert "Option loss cooldown active for CDNS" in reason
