from datetime import UTC, datetime, timedelta

from trading_agent.audit import AuditStore, start_of_trading_day
from trading_agent.models import AccountSnapshot, AssetClass, MarketSnapshot, OrderIntent, OrderSide, Position


def test_audit_store_records_cycle_and_events(tmp_path) -> None:
    store = AuditStore(tmp_path / "audit.sqlite3")
    cycle_id = store.start_cycle(
        AccountSnapshot(equity=100_000, cash=50_000, buying_power=50_000),
        [Position(symbol="AAPL", asset_class=AssetClass.EQUITY, qty=1, market_value=200)],
    )

    store.record_event(
        cycle_id=cycle_id,
        event_type="market_snapshot",
        payload=MarketSnapshot(symbol="AAPL", asset_class=AssetClass.EQUITY, price=200),
        symbol="AAPL",
        status="captured",
    )
    store.finish_cycle(cycle_id)

    summary = store.latest_summary()

    assert summary["cycle"]["id"] == cycle_id
    assert summary["cycle"]["status"] == "completed"
    assert summary["market_snapshots"][0]["symbol"] == "AAPL"
    assert summary["market_snapshots"][0]["payload"]["price"] == 200


def test_audit_store_preserves_existing_cycles_when_reopened(tmp_path) -> None:
    database_path = tmp_path / "audit.sqlite3"
    first_store = AuditStore(database_path)
    first_cycle_id = first_store.start_cycle(
        AccountSnapshot(equity=100_000, cash=50_000, buying_power=50_000),
        [],
    )
    first_store.finish_cycle(first_cycle_id)

    second_store = AuditStore(database_path)
    second_cycle_id = second_store.start_cycle(
        AccountSnapshot(equity=101_000, cash=51_000, buying_power=51_000),
        [],
    )
    second_store.finish_cycle(second_cycle_id)

    cycles = second_store.recent_cycles(limit=10)

    assert [cycle["id"] for cycle in cycles] == [second_cycle_id, first_cycle_id]
    assert second_store.counts()["cycles"] == 2


def test_audit_backup_and_export_preserve_data(tmp_path) -> None:
    database_path = tmp_path / "audit.sqlite3"
    store = AuditStore(database_path)
    cycle_id = store.start_cycle(
        AccountSnapshot(equity=100_000, cash=50_000, buying_power=50_000),
        [],
    )
    store.finish_cycle(cycle_id)

    backup_path = store.backup(tmp_path / "backup.sqlite3")
    export_path = store.export_json(tmp_path / "export.json")

    backup_store = AuditStore(backup_path)
    assert backup_store.counts()["cycles"] == 1
    assert export_path.read_text(encoding="utf-8").count("cycles") >= 1


def test_position_state_tracks_peak_and_trough(tmp_path) -> None:
    store = AuditStore(tmp_path / "audit.sqlite3")

    first = store.update_position_state(symbol="AAPL", asset_class="equity", current_price=100)
    second = store.update_position_state(symbol="AAPL", asset_class="equity", current_price=110)
    third = store.update_position_state(symbol="AAPL", asset_class="equity", current_price=95)

    assert first["peak_price"] == 100
    assert second["peak_price"] == 110
    assert third["peak_price"] == 110
    assert third["trough_price"] == 95


def test_reconcile_position_states_removes_closed_symbols(tmp_path) -> None:
    store = AuditStore(tmp_path / "audit.sqlite3")
    store.update_position_state(symbol="AAPL", asset_class="equity", current_price=100)
    store.update_position_state(symbol="MSFT", asset_class="equity", current_price=200)

    store.reconcile_position_states({"AAPL"})

    with store._connect() as connection:
        symbols = [
            row["symbol"]
            for row in connection.execute("select symbol from position_state order by symbol").fetchall()
        ]
    assert symbols == ["AAPL"]


def test_order_counts_since_splits_entries_and_exits(tmp_path) -> None:
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
                qty=1,
            )
        },
        symbol="AAPL",
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
                qty=1,
                metadata={"exit": True},
            )
        },
        symbol="AAPL",
        status="submitted",
    )

    counts = store.order_counts_since(datetime.now(UTC) - timedelta(minutes=1))

    assert counts == {"total_orders": 2, "entry_orders": 1, "exit_orders": 1}


def test_start_of_trading_day_uses_central_midnight() -> None:
    start = start_of_trading_day(now=datetime(2026, 5, 28, 15, 30, tzinfo=UTC))

    assert start == datetime(2026, 5, 28, 5, 0, tzinfo=UTC)
