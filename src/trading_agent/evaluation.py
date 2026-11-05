"""Daily paper-trading evaluation report.

Builds a Markdown report from the broker (authoritative fills, positions, open
orders) and the audit log (decision funnel, errors): round trips per book,
safety invariants, and performance against SPY since the evaluation start.
"""
from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")


@dataclass
class RoundTrip:
    book: str
    symbol: str
    entry_strategy: str
    exit_kind: str
    qty: float
    entry_price: float
    exit_price: float
    entry_at: datetime
    exit_at: datetime

    @property
    def pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.qty

    @property
    def ret_pct(self) -> float:
        return (self.exit_price / self.entry_price - 1) * 100


def strategy_of(order: dict[str, Any]) -> str:
    client = str(order.get("client_order_id") or "")
    if client.startswith("ta-"):
        return client[3:].rsplit("-", 1)[0]
    return "bracket_leg"


def book_of(strategy: str) -> str:
    if strategy.startswith("day_trade"):
        return "day"
    if strategy.endswith("_momentum") or strategy.endswith("_catalyst"):
        return "swing"
    return "other"


def round_trips(closed_orders: list[dict[str, Any]]) -> list[RoundTrip]:
    """FIFO-match filled equity buys and sells per symbol (long-only books)."""
    fills = []
    for order in closed_orders:
        if order.get("status") not in {"filled", "partially_filled"} or not order.get("filled_at"):
            continue
        if str(order.get("asset_class") or "") not in {"us_equity", ""}:
            continue
        qty = float(order.get("filled_qty") or 0)
        price = float(order.get("filled_avg_price") or 0)
        if qty <= 0 or price <= 0:
            continue
        fills.append((datetime.fromisoformat(str(order["filled_at"]).replace("Z", "+00:00")), order, qty, price))
    fills.sort(key=lambda row: row[0])
    lots: dict[str, deque] = defaultdict(deque)
    trips: list[RoundTrip] = []
    for filled_at, order, qty, price in fills:
        symbol = str(order.get("symbol"))
        strategy = strategy_of(order)
        if order.get("side") == "buy":
            lots[symbol].append([qty, price, strategy, filled_at])
            continue
        remaining = qty
        while remaining > 1e-9 and lots[symbol]:
            lot = lots[symbol][0]
            matched = min(remaining, lot[0])
            trips.append(RoundTrip(book_of(lot[2]), symbol, lot[2], strategy, matched, lot[1], price, lot[3], filled_at))
            lot[0] -= matched
            remaining -= matched
            if lot[0] <= 1e-9:
                lots[symbol].popleft()
    return trips


def book_stats(trips: list[RoundTrip]) -> dict[str, Any]:
    pnls = [trip.pnl for trip in trips]
    wins = [p for p in pnls if p > 0]
    losses = [-p for p in pnls if p <= 0]
    mean = sum(pnls) / len(pnls) if pnls else 0.0
    sd = (sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)) ** 0.5 if len(pnls) > 1 else 0.0
    cum = peak = drawdown = 0.0
    for trip in sorted(trips, key=lambda t: t.exit_at):
        cum += trip.pnl
        peak = max(peak, cum)
        drawdown = max(drawdown, peak - cum)
    return {
        "trades": len(pnls),
        "pnl": sum(pnls),
        "win_rate": len(wins) / len(pnls) if pnls else 0.0,
        "profit_factor": (sum(wins) / sum(losses)) if losses else (math.inf if wins else 0.0),
        "expectancy": mean,
        "t_stat": mean / sd * math.sqrt(len(pnls)) if sd else 0.0,
        "max_drawdown": drawdown,
        "avg_win_pct": sum(t.ret_pct for t in trips if t.pnl > 0) / len(wins) if wins else 0.0,
        "avg_loss_pct": sum(t.ret_pct for t in trips if t.pnl <= 0) / len(losses) if losses else 0.0,
    }


def invariant_violations(
    positions: list[Any], open_orders: list[dict[str, Any]], day_trade_symbols: set[str], *, after_close: bool,
) -> list[str]:
    problems: list[str] = []
    for position in positions:
        symbol, qty = position.symbol, float(position.qty)
        if getattr(position.asset_class, "value", position.asset_class) not in {"equity", "etf"} or qty <= 0:
            continue
        if after_close and symbol in day_trade_symbols:
            problems.append(f"{symbol}: day-trade position still open after the close ({qty:g} shares).")
            continue
        stops = [
            order for order in open_orders
            if order.get("symbol") == symbol and order.get("side") == "sell"
            and str(order.get("type") or order.get("order_type")) in {"stop", "stop_limit"}
        ]
        gtc_qty = sum(float(order.get("qty") or 0) for order in stops if order.get("time_in_force") == "gtc")
        if gtc_qty + 1e-9 < int(qty):
            problems.append(
                f"{symbol}: {int(qty)} shares but only {gtc_qty:g} covered by a resting GTC stop."
            )
    return problems


def cross_book_conflicts(trips: list[RoundTrip]) -> list[str]:
    """Symbols entered by both the swing and the day-trade book on the same day."""
    books: dict[tuple[str, date], set[str]] = defaultdict(set)
    for trip in trips:
        books[(trip.symbol, trip.entry_at.astimezone(EASTERN).date())].add(trip.book)
    return [
        f"{symbol}: entered by both the swing and day-trade books on {day.isoformat()}."
        for (symbol, day), seen in sorted(books.items())
        if {"swing", "day"} <= seen
    ]


def audit_funnel(db_path: str | Path, day: date) -> dict[str, Any]:
    start = datetime.combine(day, time(0, 0), tzinfo=EASTERN).astimezone(UTC).isoformat()
    end = datetime.combine(day + timedelta(days=1), time(0, 0), tzinfo=EASTERN).astimezone(UTC).isoformat()
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        cycles = connection.execute(
            "select status, error from cycles where started_at >= ? and started_at < ?", (start, end)
        ).fetchall()
        rows = connection.execute(
            "select event_type, strategy, status, reason from audit_events where created_at >= ? and created_at < ?",
            (start, end),
        ).fetchall()
    finally:
        connection.close()
    funnel: dict[str, Any] = {
        "cycles": Counter(row["status"] for row in cycles),
        "cycle_errors": Counter((row["error"] or "")[:120] for row in cycles if row["error"]),
        "catalyst_blocks": Counter(),
        "day_trade_signals": Counter(),
        "day_trade_block_reasons": Counter(),
        "swing_candidates": Counter(),
        "orders": Counter(),
        "order_rejections": Counter(),
        "screener_errors": 0,
    }
    for row in rows:
        kind, strategy, status, reason = row["event_type"], row["strategy"] or "", row["status"] or "", row["reason"] or ""
        if kind == "catalyst_prediction":
            funnel["catalyst_blocks"][(reason or "allowed")[:70]] += 1
        elif kind == "day_trade_signal":
            funnel["day_trade_signals"][status] += 1
            if status == "blocked":
                funnel["day_trade_block_reasons"][reason[:70]] += 1
        elif kind == "trade_candidate" and strategy.endswith("_momentum"):
            funnel["swing_candidates"][status] += 1
        elif kind == "order":
            funnel["orders"][f"{strategy}:{status}"] += 1
            if status in {"rejected", "unprotected", "replace_failed", "failed"}:
                funnel["order_rejections"][f"{strategy}: {reason[:90]}"] += 1
    return funnel


def _table(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _stats_row(name: str, stats: dict[str, Any]) -> list[str]:
    pf = stats["profit_factor"]
    return [
        name, str(stats["trades"]), f"${stats['pnl']:+,.2f}", f"{stats['win_rate'] * 100:.0f}%",
        "inf" if pf == math.inf else f"{pf:.2f}", f"${stats['expectancy']:+,.2f}",
        f"{stats['avg_win_pct']:+.2f}% / {stats['avg_loss_pct']:+.2f}%", f"${stats['max_drawdown']:,.2f}",
        f"{stats['t_stat']:+.2f}",
    ]


def render_report(
    *,
    day: date,
    since: date,
    account: Any,
    start_equity: float | None,
    spy_change_since_pct: float | None,
    spy_change_day_pct: float | None,
    trips: list[RoundTrip],
    violations: list[str],
    funnel: dict[str, Any],
    positions: list[Any],
) -> str:
    today = [trip for trip in trips if trip.exit_at.astimezone(EASTERN).date() == day]
    header = ["Book", "Trades", "P/L", "Win", "PF", "Expectancy", "Avg win / loss", "Max DD", "t-stat"]
    out = [f"# Paper evaluation — {day.isoformat()}", ""]
    day_pl = account.equity - (account.last_equity or account.equity)
    out.append(
        f"**Equity** ${account.equity:,.2f} (day {day_pl:+,.2f}, {account.daily_pl_pct:+.2f}%; "
        f"SPY day {spy_change_day_pct:+.2f}%)" if spy_change_day_pct is not None else
        f"**Equity** ${account.equity:,.2f} (day {day_pl:+,.2f})"
    )
    if start_equity:
        since_pct = (account.equity / start_equity - 1) * 100
        spy_text = f" vs SPY {spy_change_since_pct:+.2f}%" if spy_change_since_pct is not None else ""
        out.append(f"**Since {since.isoformat()}** {since_pct:+.2f}%{spy_text}")
    out += ["", "## Safety invariants", ""]
    out += [f"- VIOLATION: {problem}" for problem in violations] or ["- All checks passed: no day trade held past the close; every swing position has a full-size GTC stop."]
    out += ["", "## Today", ""]
    out.append(_table([_stats_row(book, book_stats([t for t in today if t.book == book])) for book in ("day", "swing")], header))
    out += ["", f"## Since {since.isoformat()} (cumulative)", ""]
    out.append(_table([_stats_row(book, book_stats([t for t in trips if t.book == book])) for book in ("day", "swing")], header))
    out += ["", "## Round trips today", ""]
    if today:
        out.append(_table(
            [[t.book, t.symbol, t.entry_strategy, t.exit_kind, f"{t.qty:g}", f"{t.entry_price:.2f}", f"{t.exit_price:.2f}",
              f"{t.ret_pct:+.2f}%", f"${t.pnl:+,.2f}",
              f"{t.entry_at.astimezone(EASTERN):%H:%M}->{t.exit_at.astimezone(EASTERN):%m-%d %H:%M}"] for t in today],
            ["Book", "Symbol", "Entry", "Exit", "Qty", "Entry px", "Exit px", "Return", "P/L", "Time (ET)"],
        ))
    else:
        out.append("No round trips closed today.")
    out += ["", "## Open positions", ""]
    if positions:
        out.append(_table(
            [[p.symbol, f"{p.qty:g}", f"{p.avg_entry_price or 0:.2f}", f"{p.current_price or 0:.2f}", f"${p.unrealized_pl or 0:+,.2f}"]
             for p in positions],
            ["Symbol", "Qty", "Entry", "Last", "Unrealized"],
        ))
    else:
        out.append("None.")
    out += ["", "## Decision funnel (audit log)", ""]
    out.append(f"- Cycles: {dict(funnel['cycles'])}")
    for label, key in (("Cycle errors", "cycle_errors"), ("Order problems", "order_rejections")):
        if funnel[key]:
            out.append(f"- {label}:")
            out += [f"  - {count}× {text}" for text, count in funnel[key].most_common(8)]
    out.append(f"- Swing candidates: {dict(funnel['swing_candidates'])}")
    out.append(f"- Orders: {dict(funnel['orders'])}")
    out.append(f"- Day-trade signals: {dict(funnel['day_trade_signals'])}")
    if funnel["day_trade_block_reasons"]:
        out.append("- Top day-trade block reasons:")
        out += [f"  - {count}× {text}" for text, count in funnel["day_trade_block_reasons"].most_common(6)]
    if funnel["catalyst_blocks"]:
        total = sum(funnel["catalyst_blocks"].values())
        out.append("- Catalyst outcomes:")
        out += [f"  - {count / total * 100:.0f}% {text}" for text, count in funnel["catalyst_blocks"].most_common(5)]
    return "\n".join(out) + "\n"


async def build_daily_report(settings: Any, *, day: date | None = None, since: date | None = None) -> tuple[str, dict]:
    from trading_agent.audit import AuditStore, start_of_trading_day
    from trading_agent.brokers.alpaca import AlpacaBroker

    broker = AlpacaBroker(settings)
    now = datetime.now(UTC)
    day = day or now.astimezone(EASTERN).date()
    since = since or day
    account = await broker.get_account()
    positions = await broker.get_positions()
    open_orders = await broker.get_open_orders()
    after = datetime.combine(since, time(0, 0), tzinfo=EASTERN).astimezone(UTC).isoformat()
    until = datetime.combine(day + timedelta(days=1), time(0, 0), tzinfo=EASTERN).astimezone(UTC).isoformat()
    closed = await broker.get_recent_orders(status="closed", limit=500, after=after, until=until)
    trips = round_trips(closed)
    audit = AuditStore(settings.audit.database_path)
    day_trade_symbols = {
        str(event.get("symbol") or "") for event in audit.day_trade_entries_since(start_of_trading_day())
    }
    clock = await broker.get_clock()
    after_close = not clock.get("is_open") and now.astimezone(EASTERN).time() >= time(16, 0)
    violations = invariant_violations(positions, open_orders, day_trade_symbols, after_close=after_close)
    violations += cross_book_conflicts(trips)

    start_equity = None
    connection = sqlite3.connect(f"file:{settings.audit.database_path}?mode=ro", uri=True)
    try:
        row = connection.execute(
            "select account_json from cycles where started_at >= ? order by id limit 1", (after,)
        ).fetchone()
        if row and row[0]:
            start_equity = float(json.loads(row[0]).get("equity") or 0) or None
    finally:
        connection.close()
    spy = (await broker.get_daily_bars(["SPY"], lookback_days=max((day - since).days + 10, 15))).get("SPY") or {}
    spy_day = spy_since = None
    closes = spy.get("closes") or []
    if len(closes) >= 2:
        spy_day = (closes[-1] / closes[-2] - 1) * 100
        sessions = max(1, sum(1 for offset in range((day - since).days + 1)
                              if (since + timedelta(days=offset)).weekday() < 5))
        if len(closes) > sessions:
            spy_since = (closes[-1] / closes[-1 - sessions] - 1) * 100
    funnel = audit_funnel(settings.audit.database_path, day)
    report = render_report(
        day=day, since=since, account=account, start_equity=start_equity,
        spy_change_since_pct=spy_since, spy_change_day_pct=spy_day, trips=trips,
        violations=violations, funnel=funnel, positions=positions,
    )
    summary = {
        "date": day.isoformat(),
        "equity": account.equity,
        "violations": violations,
        "books": {book: book_stats([t for t in trips if t.book == book]) for book in ("day", "swing")},
        "today": {book: book_stats([t for t in trips if t.book == book and t.exit_at.astimezone(EASTERN).date() == day])
                  for book in ("day", "swing")},
        "cycle_errors": dict(funnel["cycle_errors"]),
        "order_problems": dict(funnel["order_rejections"]),
    }
    return report, summary
