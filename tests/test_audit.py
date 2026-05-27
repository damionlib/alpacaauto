from trading_agent.audit import AuditStore
from trading_agent.models import AccountSnapshot, AssetClass, MarketSnapshot, Position


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
