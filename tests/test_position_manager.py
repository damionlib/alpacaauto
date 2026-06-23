from datetime import UTC, datetime, timedelta

from trading_agent.audit import AuditStore
from trading_agent.config import Settings
from trading_agent.models import AssetClass, OrderSide, Position
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


def test_position_manager_creates_profit_lock_exit(tmp_path) -> None:
    store = AuditStore(tmp_path / "audit.sqlite3")
    manager = PositionManager(Settings(), store)
    # Peak profit was 9%, activating the 8% -> lock 5% ladder tier.
    store.update_position_state(symbol="KLAC", asset_class="equity", current_price=109)

    candidates = manager.evaluate(
        [
            Position(
                symbol="KLAC",
                asset_class=AssetClass.EQUITY,
                qty=10,
                market_value=1_040,
                avg_entry_price=100,
                current_price=104,
                unrealized_pl=40,
            )
        ]
    )

    assert len(candidates) == 1
    assert candidates[0].strategy == "profit_lock_exit"
    assert "profit-lock tier" in candidates[0].rationale[0]


def test_position_manager_keeps_winner_above_locked_profit(tmp_path) -> None:
    store = AuditStore(tmp_path / "audit.sqlite3")
    manager = PositionManager(Settings(), store)
    store.update_position_state(symbol="KLAC", asset_class="equity", current_price=109)

    candidates = manager.evaluate(
        [
            Position(
                symbol="KLAC",
                asset_class=AssetClass.EQUITY,
                qty=10,
                market_value=1_060,
                avg_entry_price=100,
                current_price=106,
                unrealized_pl=60,
            )
        ]
    )

    assert candidates == []


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


def test_position_manager_creates_stale_loser_exit(tmp_path) -> None:
    settings = Settings.model_validate(
        {"position_manager": {"max_days_without_profit": 7}}
    )
    store = AuditStore(tmp_path / "audit.sqlite3")
    manager = PositionManager(settings, store)
    old_date = (datetime.now(UTC) - timedelta(days=10)).isoformat()
    with store._connect() as connection:
        connection.execute(
            """
            insert into position_state
                (symbol, asset_class, first_seen_at, last_seen_at, peak_price, trough_price)
            values (?, ?, ?, ?, ?, ?)
            """,
            ("LLY", "equity", old_date, old_date, 100.3, 95),
        )

    candidates = manager.evaluate(
        [
            Position(
                symbol="LLY",
                asset_class=AssetClass.EQUITY,
                qty=10,
                market_value=9_700,
                avg_entry_price=100,
                current_price=97,
                unrealized_pl=-30,
            )
        ]
    )

    assert len(candidates) == 1
    assert candidates[0].strategy == "stale_loser_exit"


def test_stale_loser_spares_position_that_was_profitable(tmp_path) -> None:
    settings = Settings.model_validate(
        {"position_manager": {"max_days_without_profit": 7}}
    )
    store = AuditStore(tmp_path / "audit.sqlite3")
    manager = PositionManager(settings, store)
    old_date = (datetime.now(UTC) - timedelta(days=10)).isoformat()
    with store._connect() as connection:
        connection.execute(
            """
            insert into position_state
                (symbol, asset_class, first_seen_at, last_seen_at, peak_price, trough_price)
            values (?, ?, ?, ?, ?, ?)
            """,
            ("AAPL", "equity", old_date, old_date, 105, 98),
        )

    candidates = manager.evaluate(
        [
            Position(
                symbol="AAPL",
                asset_class=AssetClass.EQUITY,
                qty=10,
                market_value=9_900,
                avg_entry_price=100,
                current_price=99,
                unrealized_pl=-10,
            )
        ]
    )

    stale = [c for c in candidates if c.strategy == "stale_loser_exit"]
    assert len(stale) == 0


def test_position_manager_creates_short_option_stop_loss_exit() -> None:
    manager = PositionManager(Settings())

    candidates = manager.evaluate(
        [
            Position(
                symbol="AAPL260612C00322500",
                asset_class=AssetClass.OPTION,
                qty=-1,
                market_value=-225,
                avg_entry_price=1.0,
                current_price=2.25,
                unrealized_pl=-125,
            )
        ]
    )

    assert len(candidates) == 1
    assert candidates[0].side == OrderSide.BUY
    assert candidates[0].strategy == "short_option_stop_loss_exit"


def test_position_manager_closes_debit_spread_as_one_order() -> None:
    manager = PositionManager(Settings())

    candidates = manager.evaluate(
        [
            Position(
                symbol="NVDA260617C00225000",
                asset_class=AssetClass.OPTION,
                qty=1,
                market_value=330,
                avg_entry_price=10.25,
                current_price=3.30,
                unrealized_pl=-695,
            ),
            Position(
                symbol="NVDA260617C00240000",
                asset_class=AssetClass.OPTION,
                qty=-1,
                market_value=-105,
                avg_entry_price=3.55,
                current_price=1.05,
                unrealized_pl=250,
            ),
        ]
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.symbol == "NVDA_call_debit_spread"
    assert candidate.strategy == "spread_stop_loss_exit"
    assert candidate.metadata["spread_exit"] is True
    assert candidate.metadata["metrics"]["pnl_pct"] < -40
    assert candidate.metadata["legs"] == [
        {
            "symbol": "NVDA260617C00225000",
            "ratio_qty": "1",
            "side": "sell",
            "position_intent": "sell_to_close",
        },
        {
            "symbol": "NVDA260617C00240000",
            "ratio_qty": "1",
            "side": "buy",
            "position_intent": "buy_to_close",
        },
    ]


def test_position_manager_does_not_close_one_losing_spread_leg() -> None:
    manager = PositionManager(Settings())

    candidates = manager.evaluate(
        [
            Position(
                symbol="AAPL260617C00305000",
                asset_class=AssetClass.OPTION,
                qty=1,
                market_value=1_105,
                avg_entry_price=11.05,
                current_price=11.05,
                unrealized_pl=0,
            ),
            Position(
                symbol="AAPL260617C00320000",
                asset_class=AssetClass.OPTION,
                qty=-1,
                market_value=-295,
                avg_entry_price=2.05,
                current_price=2.95,
                unrealized_pl=-90,
            ),
        ]
    )

    assert candidates == []
