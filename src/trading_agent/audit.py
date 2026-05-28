from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(to_jsonable(item) for item in value)
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    return value


class AuditStore:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("pragma journal_mode = wal")
        connection.execute("pragma synchronous = normal")
        connection.execute("pragma foreign_keys = on")
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                create table if not exists cycles (
                    id integer primary key autoincrement,
                    started_at text not null,
                    completed_at text,
                    status text not null,
                    account_json text,
                    positions_json text,
                    error text
                );

                create table if not exists audit_events (
                    id integer primary key autoincrement,
                    cycle_id integer,
                    created_at text not null,
                    event_type text not null,
                    symbol text,
                    strategy text,
                    approved integer,
                    score real,
                    status text,
                    reason text,
                    payload_json text not null,
                    foreign key(cycle_id) references cycles(id)
                );

                create table if not exists position_state (
                    symbol text primary key,
                    asset_class text not null,
                    first_seen_at text not null,
                    last_seen_at text not null,
                    peak_price real,
                    trough_price real
                );

                create index if not exists idx_audit_events_cycle_id on audit_events(cycle_id);
                create index if not exists idx_audit_events_type on audit_events(event_type);
                create index if not exists idx_audit_events_created_at on audit_events(created_at);
                """
            )

    def start_cycle(self, account: Any, positions: Any) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                insert into cycles (started_at, status, account_json, positions_json)
                values (?, ?, ?, ?)
                """,
                (
                    utc_now(),
                    "running",
                    self._dumps(account),
                    self._dumps(positions),
                ),
            )
            return int(cursor.lastrowid)

    def finish_cycle(self, cycle_id: int, *, status: str = "completed", error: str | None = None) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                update cycles
                set completed_at = ?, status = ?, error = ?
                where id = ?
                """,
                (utc_now(), status, error, cycle_id),
            )

    def record_event(
        self,
        *,
        cycle_id: int | None,
        event_type: str,
        payload: Any,
        symbol: str | None = None,
        strategy: str | None = None,
        approved: bool | None = None,
        score: float | None = None,
        status: str | None = None,
        reason: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                insert into audit_events
                    (cycle_id, created_at, event_type, symbol, strategy, approved, score, status, reason, payload_json)
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cycle_id,
                    utc_now(),
                    event_type,
                    symbol,
                    strategy,
                    None if approved is None else int(approved),
                    score,
                    status,
                    reason,
                    self._dumps(payload),
                ),
            )

    def latest_cycle(self) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "select * from cycles order by id desc limit 1"
            ).fetchone()
        return self._cycle_row(row) if row else None

    def recent_cycles(self, limit: int = 25) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "select * from cycles order by id desc limit ?",
                (limit,),
            ).fetchall()
        return [self._cycle_row(row) for row in rows]

    def events(
        self,
        *,
        cycle_id: int | None = None,
        event_type: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        query = "select * from audit_events"
        clauses = []
        params: list[Any] = []
        if cycle_id is not None:
            clauses.append("cycle_id = ?")
            params.append(cycle_id)
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)
        if clauses:
            query += " where " + " and ".join(clauses)
        query += " order by id desc limit ?"
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._event_row(row) for row in rows]

    def latest_summary(self) -> dict[str, Any]:
        cycle = self.latest_cycle()
        if not cycle:
            return {
                "cycle": None,
                "decisions": [],
                "orders": [],
                "market_snapshots": [],
                "research_results": [],
            }
        cycle_id = int(cycle["id"])
        return {
            "cycle": cycle,
            "decisions": self.events(cycle_id=cycle_id, event_type="risk_decision", limit=500),
            "orders": self.events(cycle_id=cycle_id, event_type="order", limit=500),
            "market_snapshots": self.events(cycle_id=cycle_id, event_type="market_snapshot", limit=500),
            "research_results": self.events(cycle_id=cycle_id, event_type="research_result", limit=500),
        }

    def backup(self, backup_path: str | Path | None = None) -> Path:
        if backup_path is None:
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            backup_path = self.database_path.with_name(f"{self.database_path.stem}-{timestamp}.backup.sqlite3")
        destination = Path(backup_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as source:
            with sqlite3.connect(destination) as target:
                source.backup(target)
        return destination

    def export_json(self, output_path: str | Path) -> Path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "exported_at": utc_now(),
            "database_path": str(self.database_path),
            "cycles": self.recent_cycles(limit=1_000_000),
            "events": self.events(limit=1_000_000),
        }
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return destination

    def counts(self) -> dict[str, int]:
        with self._connect() as connection:
            cycles = connection.execute("select count(*) from cycles").fetchone()[0]
            events = connection.execute("select count(*) from audit_events").fetchone()[0]
        return {"cycles": int(cycles), "events": int(events)}

    def order_counts_since(self, since: datetime) -> dict[str, int]:
        since_text = since.astimezone(UTC).isoformat()
        with self._connect() as connection:
            rows = connection.execute(
                """
                select payload_json
                from audit_events
                where event_type = 'order'
                  and status = 'submitted'
                  and created_at >= ?
                """,
                (since_text,),
            ).fetchall()

        total_orders = 0
        entry_orders = 0
        exit_orders = 0
        for row in rows:
            payload = self._loads(row["payload_json"]) or {}
            intent = payload.get("intent") or {}
            metadata = intent.get("metadata") or {}
            total_orders += 1
            if metadata.get("exit"):
                exit_orders += 1
            else:
                entry_orders += 1
        return {
            "total_orders": total_orders,
            "entry_orders": entry_orders,
            "exit_orders": exit_orders,
        }

    def update_position_state(
        self,
        *,
        symbol: str,
        asset_class: str,
        current_price: float,
    ) -> dict[str, Any]:
        now = utc_now()
        with self._connect() as connection:
            row = connection.execute(
                "select * from position_state where symbol = ?",
                (symbol,),
            ).fetchone()
            if row:
                peak_price = max(float(row["peak_price"] or current_price), current_price)
                trough_price = min(float(row["trough_price"] or current_price), current_price)
                connection.execute(
                    """
                    update position_state
                    set asset_class = ?, last_seen_at = ?, peak_price = ?, trough_price = ?
                    where symbol = ?
                    """,
                    (asset_class, now, peak_price, trough_price, symbol),
                )
                first_seen_at = row["first_seen_at"]
            else:
                first_seen_at = now
                peak_price = current_price
                trough_price = current_price
                connection.execute(
                    """
                    insert into position_state
                        (symbol, asset_class, first_seen_at, last_seen_at, peak_price, trough_price)
                    values (?, ?, ?, ?, ?, ?)
                    """,
                    (symbol, asset_class, first_seen_at, now, peak_price, trough_price),
                )
        return {
            "symbol": symbol,
            "asset_class": asset_class,
            "first_seen_at": first_seen_at,
            "last_seen_at": now,
            "peak_price": peak_price,
            "trough_price": trough_price,
        }

    def reconcile_position_states(self, active_symbols: set[str]) -> None:
        with self._connect() as connection:
            if not active_symbols:
                connection.execute("delete from position_state")
                return
            placeholders = ",".join("?" for _ in active_symbols)
            connection.execute(
                f"delete from position_state where symbol not in ({placeholders})",
                tuple(active_symbols),
            )

    def _dumps(self, value: Any) -> str:
        return json.dumps(to_jsonable(value), sort_keys=True, default=str)

    def _loads(self, value: str | None) -> Any:
        if not value:
            return None
        return json.loads(value)

    def _cycle_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "status": row["status"],
            "account": self._loads(row["account_json"]),
            "positions": self._loads(row["positions_json"]),
            "error": row["error"],
        }

    def _event_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "cycle_id": row["cycle_id"],
            "created_at": row["created_at"],
            "event_type": row["event_type"],
            "symbol": row["symbol"],
            "strategy": row["strategy"],
            "approved": None if row["approved"] is None else bool(row["approved"]),
            "score": row["score"],
            "status": row["status"],
            "reason": row["reason"],
            "payload": self._loads(row["payload_json"]),
        }


def start_of_trading_day(
    *,
    now: datetime | None = None,
    timezone_name: str = "America/Chicago",
) -> datetime:
    timezone = ZoneInfo(timezone_name)
    local_now = (now or datetime.now(UTC)).astimezone(timezone)
    return local_now.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(UTC)
