from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from trading_agent.agent import TradingAgent
from trading_agent.audit import AuditStore
from trading_agent.broker_sync import BrokerOrderSync
from trading_agent.brokers.alpaca import AlpacaBroker
from trading_agent.config import load_settings
from trading_agent.dashboard import serve_dashboard
from trading_agent.screener.service import MarketScreener


app = typer.Typer(help="Research, risk, and execution agent for API-first trading.")
console = Console()


@app.command()
def account(config: str = "config/settings.toml") -> None:
    async def _run() -> None:
        broker = AlpacaBroker(load_settings(config))
        snapshot = await broker.get_account()
        console.print(snapshot.model_dump())

    asyncio.run(_run())


@app.command()
def positions(config: str = "config/settings.toml") -> None:
    async def _run() -> None:
        broker = AlpacaBroker(load_settings(config))
        rows = await broker.get_positions()
        table = Table("Symbol", "Class", "Qty", "Market Value", "Unrealized P/L")
        for row in rows:
            table.add_row(
                row.symbol,
                row.asset_class.value,
                str(row.qty),
                f"${row.market_value:,.2f}",
                "" if row.unrealized_pl is None else f"${row.unrealized_pl:,.2f}",
            )
        console.print(table)

    asyncio.run(_run())


@app.command("open-orders")
def open_orders(config: str = "config/settings.toml") -> None:
    async def _run() -> None:
        broker = AlpacaBroker(load_settings(config))
        rows = await broker.get_open_orders()
        table = Table("Symbol", "Side", "Qty", "Type", "Limit", "Status", "Created")
        for row in rows:
            table.add_row(
                str(row.get("symbol", "")),
                str(row.get("side", "")),
                str(row.get("qty", "")),
                str(row.get("type", "")),
                str(row.get("limit_price") or ""),
                str(row.get("status", "")),
                str(row.get("created_at", "")),
            )
        console.print(table if rows else "No open orders.")

    asyncio.run(_run())


@app.command("cancel-open-orders")
def cancel_open_orders(config: str = "config/settings.toml") -> None:
    async def _run() -> None:
        broker = AlpacaBroker(load_settings(config))
        result = await broker.cancel_all_orders()
        console.print(result)

    asyncio.run(_run())


@app.command("run-once")
def run_once(config: str = "config/settings.toml") -> None:
    settings = load_settings(config)
    agent = TradingAgent(settings, console=console)
    asyncio.run(agent.run_once())


@app.command()
def screen(config: str = "config/settings.toml") -> None:
    async def _run() -> None:
        settings = load_settings(config)
        broker = AlpacaBroker(settings)
        screener = MarketScreener(settings, broker)
        rows = await screener.top_symbols()
        table = Table("Symbol", "Class", "Score", "Reasons")
        for row in rows:
            table.add_row(
                row.symbol,
                row.asset_class.value,
                f"{row.score:.2f}",
                "; ".join(row.reasons),
            )
        console.print(table if rows else "No screener candidates.")

    asyncio.run(_run())


@app.command()
def loop(config: str = "config/settings.toml") -> None:
    settings = load_settings(config)
    agent = TradingAgent(settings, console=console)
    asyncio.run(agent.loop())


@app.command()
def dashboard(
    config: str = "config/settings.toml",
    host: str = "127.0.0.1",
    port: int = 8080,
) -> None:
    settings = load_settings(config)
    serve_dashboard(settings.audit.database_path, host=host, port=port)


@app.command("audit-status")
def audit_status(config: str = "config/settings.toml") -> None:
    settings = load_settings(config)
    store = AuditStore(settings.audit.database_path)
    counts = store.counts()
    console.print(
        {
            "database_path": str(Path(settings.audit.database_path).resolve()),
            "cycles": counts["cycles"],
            "events": counts["events"],
        }
    )


@app.command("audit-backup")
def audit_backup(
    config: str = "config/settings.toml",
    output: str | None = None,
) -> None:
    settings = load_settings(config)
    store = AuditStore(settings.audit.database_path)
    backup_path = store.backup(output)
    console.print(f"Audit backup written to {backup_path.resolve()}")


@app.command("audit-export")
def audit_export(
    config: str = "config/settings.toml",
    output: str = "data/trading_agent_audit_export.json",
) -> None:
    settings = load_settings(config)
    store = AuditStore(settings.audit.database_path)
    export_path = store.export_json(output)
    console.print(f"Audit export written to {export_path.resolve()}")


@app.command("broker-sync")
def broker_sync(
    config: str = "config/settings.toml",
    days: int = 30,
    limit: int = 500,
) -> None:
    async def _run() -> None:
        settings = load_settings(config)
        store = AuditStore(settings.audit.database_path)
        broker = AlpacaBroker(settings)
        sync = BrokerOrderSync(broker, store, console)
        after = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        result = await sync.sync_closed_orders(after=after, limit=limit)
        console.print(
            {
                "database_path": str(Path(settings.audit.database_path).resolve()),
                "mode": settings.broker.mode,
                "after": after,
                **result,
            }
        )

    asyncio.run(_run())


if __name__ == "__main__":
    app()
