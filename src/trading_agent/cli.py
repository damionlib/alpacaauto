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


@app.command("paper-reset")
def paper_reset(
    config: str = "config/settings.toml",
    backup_audit: bool = True,
) -> None:
    """Cancel paper orders and close paper positions before a clean evaluation run."""

    async def _run() -> None:
        settings = load_settings(config)
        if settings.broker.mode != "paper":
            raise typer.BadParameter("paper-reset refuses to run unless broker.mode is paper.")

        if backup_audit and settings.audit.enabled:
            store = AuditStore(settings.audit.database_path)
            backup_path = store.backup()
            console.print(f"Audit backup written to {backup_path.resolve()}")

        broker = AlpacaBroker(settings)
        account_before = await broker.get_account()
        positions_before = await broker.get_positions()
        open_orders_before = await broker.get_open_orders()
        cancelled = await broker.cancel_all_orders()
        closed: list[dict] = []
        close_errors: list[str] = []
        try:
            closed = await broker.close_all_positions(cancel_orders=True)
        except RuntimeError as error:
            close_errors.append(str(error))

        open_orders_mid = await broker.get_open_orders()
        pending_symbols = {str(order.get("symbol", "")) for order in open_orders_mid}
        remaining_positions = sorted(
            await broker.get_positions(),
            key=lambda position: 0 if position.asset_class.value == "option" else 1,
        )
        individual_closes: list[dict] = []
        for position in remaining_positions:
            if position.symbol in pending_symbols:
                continue
            try:
                result = await broker.close_position(position.symbol)
            except RuntimeError as error:
                close_errors.append(f"{position.symbol}: {error}")
                continue
            individual_closes.append(result)
            open_orders_mid = await broker.get_open_orders()
            pending_symbols = {str(order.get("symbol", "")) for order in open_orders_mid}

        account_after = await broker.get_account()
        positions_after = await broker.get_positions()
        open_orders_after = await broker.get_open_orders()

        console.print(
            {
                "mode": settings.broker.mode,
                "account_before": account_before.model_dump(),
                "positions_before": len(positions_before),
                "open_orders_before": len(open_orders_before),
                "cancelled_orders": cancelled,
                "close_positions": closed,
                "individual_closes": individual_closes,
                "close_errors": close_errors,
                "account_after": account_after.model_dump(),
                "positions_after": len(positions_after),
                "open_orders_after": len(open_orders_after),
            }
        )

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


@app.command("session-window")
def session_window(config: str = "config/settings.toml", stop_after_close_minutes: int = 20) -> None:
    """Print JSON: whether today is a trading session and when the runner should stop."""
    import json

    from trading_agent.day_trading.features import EASTERN

    async def _run() -> None:
        clock = await AlpacaBroker(load_settings(config)).get_clock()
        today = datetime.now(UTC).astimezone(EASTERN).date()
        next_open = datetime.fromisoformat(str(clock["next_open"]).replace("Z", "+00:00"))
        next_close = datetime.fromisoformat(str(clock["next_close"]).replace("Z", "+00:00"))
        trading_day = bool(clock.get("is_open")) or next_open.astimezone(EASTERN).date() == today
        stop = next_close + timedelta(minutes=stop_after_close_minutes)
        print(json.dumps({
            "trading_day": trading_day,
            "is_open": bool(clock.get("is_open")),
            "close": next_close.isoformat(),
            "stop_epoch": int(stop.timestamp()) if trading_day else 0,
        }))

    asyncio.run(_run())


@app.command("eval-report")
def eval_report(
    config: str = "config/settings.toml",
    since: str = typer.Option("", help="Evaluation start date (YYYY-MM-DD); defaults to today."),
    day: str = typer.Option("", help="Report date (YYYY-MM-DD); defaults to today (ET)."),
    output_dir: str = "reports/paper-eval",
) -> None:
    """Write the daily paper-evaluation report (Markdown + JSON summary)."""
    import json
    from datetime import date as date_type

    from trading_agent.evaluation import build_daily_report

    async def _run() -> None:
        settings = load_settings(config)
        report_day = date_type.fromisoformat(day) if day else None
        since_day = date_type.fromisoformat(since) if since else None
        report, summary = await build_daily_report(settings, day=report_day, since=since_day)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = summary["date"]
        (out / f"{stem}.md").write_text(report)
        (out / f"{stem}.json").write_text(json.dumps(summary, indent=2, default=str))
        console.print(report)
        console.print(f"[green]wrote[/green] {out / (stem + '.md')}")

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
