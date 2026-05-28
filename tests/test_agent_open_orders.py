from trading_agent.agent import TradingAgent
from trading_agent.models import AssetClass, OrderIntent, OrderSide, RiskDecision, TradeCandidate


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
