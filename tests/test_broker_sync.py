import asyncio

from trading_agent.audit import AuditStore
from trading_agent.broker_sync import BrokerOrderSync


class FakeBroker:
    def __init__(self, orders) -> None:
        self.orders = orders
        self.calls = []

    async def get_recent_orders(
        self,
        *,
        status: str = "closed",
        limit: int = 100,
        after: str | None = None,
        until: str | None = None,
    ):
        self.calls.append({"status": status, "limit": limit, "after": after, "until": until})
        return self.orders


def test_broker_sync_records_closed_orders_and_dedupes(tmp_path) -> None:
    store = AuditStore(tmp_path / "audit.sqlite3")
    broker = FakeBroker(
        [
            {
                "id": "order-1",
                "client_order_id": "ta-equity_momentum-abc",
                "symbol": "AAPL",
                "status": "filled",
                "filled_at": "2026-05-29T15:00:00Z",
            },
            {
                "id": "order-2",
                "client_order_id": "not-agent-order",
                "symbol": "MSFT",
                "status": "canceled",
                "canceled_at": "2026-05-29T16:00:00Z",
            },
        ]
    )
    sync = BrokerOrderSync(broker, store)

    first = asyncio.run(sync.sync_closed_orders(after="2026-05-01T00:00:00+00:00", limit=500))
    second = asyncio.run(sync.sync_closed_orders(after="2026-05-01T00:00:00+00:00", limit=500))
    updates = store.events(event_type="broker_order_update", limit=10)

    assert first == {"fetched": 2, "synced": 2, "skipped": 0}
    assert second == {"fetched": 2, "synced": 0, "skipped": 2}
    assert len(updates) == 2
    assert updates[1]["symbol"] == "AAPL"
    assert updates[1]["strategy"] == "equity_momentum"
    assert updates[0]["symbol"] == "MSFT"
    assert updates[0]["strategy"] is None
    assert broker.calls[0]["status"] == "closed"
