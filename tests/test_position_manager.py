from datetime import UTC, datetime, timedelta

from trading_agent.audit import AuditStore
from trading_agent.config import Settings
from trading_agent.models import AssetClass, Position
from trading_agent.position_manager import PositionManager


def test_position_manager_creates_stop_loss_exit() -> None:
    manager = PositionManager(Settings())

    candidates = manager.evaluate(
        [
            Position(
                symbol="AAPL",
                asset_class=AssetClass.EQUITY,
                qty=10,
                market_value=9_300,
                avg_entry_price=100,
                current_price=93,
                unrealized_pl=-70,
            )
        ]
    )

    assert len(candidates) == 1
    assert candidates[0].strategy == "stop_loss_exit"
    assert candidates[0].metadata["exit_qty"] == 10


def test_position_manager_creates_take_profit_exit() -> None:
    manager = PositionManager(Settings())

    candidates = manager.evaluate(
        [
            Position(
                symbol="AAPL",
                asset_class=AssetClass.EQUITY,
                qty=10,
                market_value=11_300,
                avg_entry_price=100,
                current_price=113,
                unrealized_pl=130,
            )
        ]
    )

    assert len(candidates) == 1
    assert candidates[0].strategy == "take_profit_exit"


def test_position_manager_creates_trailing_stop_exit(tmp_path) -> None:
    store = AuditStore(tmp_path / "audit.sqlite3")
    manager = PositionManager(Settings(), store)
    store.update_position_state(symbol="AAPL", asset_class="equity", current_price=120)

    candidates = manager.evaluate(
        [
            Position(
                symbol="AAPL",
                asset_class=AssetClass.EQUITY,
                qty=10,
                market_value=1_090,
                avg_entry_price=100,
                current_price=109,
                unrealized_pl=90,
            )
        ]
    )

    assert len(candidates) == 1
    assert candidates[0].strategy == "trailing_stop_exit"


def test_position_manager_creates_time_exit(tmp_path) -> None:
    store = AuditStore(tmp_path / "audit.sqlite3")
    manager = PositionManager(Settings(), store)
    old_date = (datetime.now(UTC) - timedelta(days=25)).isoformat()
    with store._connect() as connection:
        connection.execute(
            """
            insert into position_state
                (symbol, asset_class, first_seen_at, last_seen_at, peak_price, trough_price)
            values (?, ?, ?, ?, ?, ?)
            """,
            ("AAPL", "equity", old_date, old_date, 103, 100),
        )

    candidates = manager.evaluate(
        [
            Position(
                symbol="AAPL",
                asset_class=AssetClass.EQUITY,
                qty=10,
                market_value=1_030,
                avg_entry_price=100,
                current_price=103,
                unrealized_pl=30,
            )
        ]
    )

    assert len(candidates) == 1
    assert candidates[0].strategy == "time_exit"
