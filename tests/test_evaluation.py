from trading_agent.evaluation import book_stats, invariant_violations, round_trips
from trading_agent.models import AssetClass, Position


def _order(symbol, side, qty, price, filled_at, client):
    return {
        "symbol": symbol, "side": side, "status": "filled", "filled_qty": str(qty),
        "filled_avg_price": str(price), "filled_at": filled_at, "client_order_id": client,
        "asset_class": "us_equity",
    }


def test_round_trips_pair_fifo_and_attribute_book_by_entry() -> None:
    orders = [
        _order("AAA", "buy", 10, 100, "2026-11-05T14:00:00Z", "ta-equity_momentum-1"),
        _order("BBB", "buy", 5, 50, "2026-11-05T14:05:00Z", "ta-day_trade_entry-2"),
        _order("BBB", "sell", 5, 51, "2026-11-05T15:00:00Z", "leg-uuid"),
        _order("AAA", "sell", 4, 110, "2026-11-06T15:00:00Z", "ta-equity-protect-3"),
    ]
    trips = round_trips(orders)
    by_symbol = {trip.symbol: trip for trip in trips}
    assert by_symbol["BBB"].book == "day" and by_symbol["BBB"].pnl == 5
    assert by_symbol["AAA"].book == "swing" and by_symbol["AAA"].qty == 4 and by_symbol["AAA"].pnl == 40
    assert by_symbol["AAA"].exit_kind == "equity-protect"
    stats = book_stats(trips)
    assert stats["trades"] == 2 and stats["win_rate"] == 1.0


def _position(symbol, qty):
    return Position(symbol=symbol, asset_class=AssetClass.EQUITY, qty=qty, market_value=qty * 10,
                    avg_entry_price=10, current_price=10)


def test_invariants_flag_missing_stop_and_overnight_day_trade() -> None:
    stops = [{"symbol": "AAA", "side": "sell", "type": "stop", "qty": "6", "time_in_force": "gtc"}]
    problems = invariant_violations(
        [_position("AAA", 10), _position("DT", 5)], stops, {"DT"}, after_close=True
    )
    assert any("AAA" in p and "only 6" in p for p in problems)
    assert any("DT" in p and "after the close" in p for p in problems)
    full = [{"symbol": "AAA", "side": "sell", "type": "stop", "qty": "10", "time_in_force": "gtc"}]
    assert invariant_violations([_position("AAA", 10)], full, set(), after_close=True) == []


def test_cross_book_conflict_is_reported() -> None:
    from trading_agent.evaluation import cross_book_conflicts

    orders = [
        _order("AAA", "buy", 5, 100, "2026-11-05T14:00:00Z", "ta-day_trade_entry-1"),
        _order("AAA", "sell", 5, 101, "2026-11-05T15:00:00Z", "leg"),
        _order("AAA", "buy", 5, 101, "2026-11-05T20:30:00Z", "ta-equity_momentum-2"),
        _order("AAA", "sell", 5, 102, "2026-11-06T15:00:00Z", "ta-equity-protect-3"),
    ]
    assert cross_book_conflicts(round_trips(orders)) == [
        "AAA: entered by both the swing and day-trade books on 2026-11-05."
    ]
