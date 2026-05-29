from trading_agent.audit import AuditStore
from trading_agent.models import AccountSnapshot, AssetClass, OrderIntent, OrderSide, Position


def test_performance_report_calculates_equity_drawdown(tmp_path) -> None:
    store = AuditStore(tmp_path / "audit.sqlite3")
    for equity in [100_000, 105_000, 99_000, 102_000]:
        cycle_id = store.start_cycle(
            AccountSnapshot(equity=equity, cash=50_000, buying_power=50_000),
            [],
        )
        store.finish_cycle(cycle_id)

    report = store.performance_report()

    assert report["summary"]["total_pl"] == 2_000
    assert round(report["summary"]["max_drawdown_pct"], 2) == 5.71


def test_performance_report_maps_open_positions_to_entry_strategy(tmp_path) -> None:
    store = AuditStore(tmp_path / "audit.sqlite3")
    first_cycle_id = store.start_cycle(
        AccountSnapshot(equity=100_000, cash=50_000, buying_power=50_000),
        [],
    )
    store.record_event(
        cycle_id=first_cycle_id,
        event_type="order",
        payload={
            "intent": OrderIntent(
                symbol="AAPL",
                asset_class=AssetClass.EQUITY,
                side=OrderSide.BUY,
                qty=10,
            )
        },
        symbol="AAPL",
        strategy="equity_momentum",
        status="submitted",
    )
    store.finish_cycle(first_cycle_id)
    second_cycle_id = store.start_cycle(
        AccountSnapshot(equity=100_250, cash=50_000, buying_power=50_000),
        [
            Position(
                symbol="AAPL",
                asset_class=AssetClass.EQUITY,
                qty=10,
                market_value=2_000,
                unrealized_pl=250,
            )
        ],
    )
    store.finish_cycle(second_cycle_id)

    report = store.performance_report()
    strategy = report["strategies"][0]

    assert strategy["strategy"] == "equity_momentum"
    assert strategy["open_unrealized_pl"] == 250
    assert strategy["assessment"] == "helping"


def test_performance_report_uses_exit_signal_pl_when_fills_are_missing(tmp_path) -> None:
    store = AuditStore(tmp_path / "audit.sqlite3")
    cycle_id = store.start_cycle(
        AccountSnapshot(equity=100_000, cash=50_000, buying_power=50_000),
        [],
    )
    store.record_event(
        cycle_id=cycle_id,
        event_type="order",
        payload={
            "intent": OrderIntent(
                symbol="AAPL",
                asset_class=AssetClass.EQUITY,
                side=OrderSide.BUY,
                qty=10,
            )
        },
        symbol="AAPL",
        strategy="equity_momentum",
        status="submitted",
    )
    store.record_event(
        cycle_id=cycle_id,
        event_type="order",
        payload={
            "intent": OrderIntent(
                symbol="AAPL",
                asset_class=AssetClass.EQUITY,
                side=OrderSide.SELL,
                qty=10,
                metadata={
                    "exit": True,
                    "position": {"unrealized_pl": -80},
                },
            )
        },
        symbol="AAPL",
        strategy="stop_loss_exit",
        status="submitted",
    )
    store.finish_cycle(cycle_id)

    report = store.performance_report()
    strategy = report["strategies"][0]

    assert strategy["strategy"] == "equity_momentum"
    assert strategy["exit_signals"] == 1
    assert strategy["win_rate_pct"] == 0
    assert strategy["total_pl_estimate"] == -80
    assert strategy["assessment"] == "hurting"


def test_performance_report_calculates_realized_pl_from_filled_updates(tmp_path) -> None:
    store = AuditStore(tmp_path / "audit.sqlite3")
    cycle_id = store.start_cycle(
        AccountSnapshot(equity=100_000, cash=50_000, buying_power=50_000),
        [],
    )
    store.record_event(
        cycle_id=cycle_id,
        event_type="broker_order_update",
        payload={
            "broker_order": {
                "id": "entry-1",
                "client_order_id": "ta-equity_momentum-abc",
                "symbol": "AAPL",
                "side": "buy",
                "asset_class": "us_equity",
                "filled_qty": "10",
                "filled_avg_price": "100",
                "position_intent": "buy_to_open",
            }
        },
        symbol="AAPL",
        strategy="equity_momentum",
        status="filled",
    )
    store.record_event(
        cycle_id=cycle_id,
        event_type="broker_order_update",
        payload={
            "broker_order": {
                "id": "exit-1",
                "client_order_id": "ta-take_profit_exit-def",
                "symbol": "AAPL",
                "side": "sell",
                "asset_class": "us_equity",
                "filled_qty": "10",
                "filled_avg_price": "110",
                "position_intent": "sell_to_close",
            }
        },
        symbol="AAPL",
        strategy="take_profit_exit",
        status="filled",
    )
    store.finish_cycle(cycle_id)

    report = store.performance_report()
    strategy = report["strategies"][0]

    assert strategy["strategy"] == "equity_momentum"
    assert strategy["closed_trades"] == 1
    assert strategy["wins"] == 1
    assert strategy["realized_pl"] == 100
    assert strategy["win_rate_pct"] == 100
