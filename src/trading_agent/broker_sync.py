from __future__ import annotations

from typing import Any, Protocol

from rich.console import Console

from trading_agent.audit import AuditStore


class OrderHistoryBroker(Protocol):
    async def get_recent_orders(
        self,
        *,
        status: str = "closed",
        limit: int = 100,
        after: str | None = None,
        until: str | None = None,
    ) -> list[dict[str, Any]]:
        ...


class BrokerOrderSync:
    def __init__(
        self,
        broker: OrderHistoryBroker,
        audit: AuditStore,
        console: Console | None = None,
    ) -> None:
        self.broker = broker
        self.audit = audit
        self.console = console or Console()

    async def sync_closed_orders(
        self,
        *,
        cycle_id: int | None = None,
        after: str | None = None,
        until: str | None = None,
        limit: int = 500,
    ) -> dict[str, int]:
        orders = await self.broker.get_recent_orders(
            status="closed",
            limit=limit,
            after=after,
            until=until,
        )
        synced = 0
        skipped = 0
        for order in orders:
            order_id = str(order.get("id") or "")
            if order_id and self.audit.broker_order_update_exists(order_id):
                skipped += 1
                continue
            self.audit.record_event(
                cycle_id=cycle_id,
                event_type="broker_order_update",
                payload={"broker_order": order},
                symbol=order.get("symbol"),
                strategy=strategy_from_client_order_id(order),
                status=order.get("status"),
                reason=order.get("filled_at") or order.get("canceled_at") or order.get("expired_at"),
            )
            synced += 1
        return {"fetched": len(orders), "synced": synced, "skipped": skipped}


def strategy_from_client_order_id(order: dict[str, Any]) -> str | None:
    client_order_id = str(order.get("client_order_id") or "")
    if not client_order_id.startswith("ta-"):
        return None
    strategy = client_order_id[3:].rsplit("-", 1)[0]
    return strategy or None
