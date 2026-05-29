from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


UNKNOWN_STRATEGY = "manual_or_unknown"


@dataclass
class Lot:
    symbol: str
    strategy: str
    qty: float
    price: float
    direction: int
    multiplier: float


def build_performance_report(
    cycles: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    chronological_cycles = sorted(cycles, key=lambda row: row.get("id") or 0)
    chronological_events = sorted(events, key=lambda row: row.get("id") or 0)
    strategy_by_symbol = _latest_entry_strategy_by_symbol(chronological_events)
    strategy_rows = _base_strategy_rows(chronological_events, strategy_by_symbol)
    closed_stats = _closed_trade_stats(chronological_events)
    open_positions = _open_position_stats(chronological_cycles, strategy_by_symbol)

    for strategy, stats in closed_stats.items():
        row = strategy_rows[strategy]
        row["closed_trades"] += stats["closed_trades"]
        row["wins"] += stats["wins"]
        row["losses"] += stats["losses"]
        row["realized_pl"] += stats["realized_pl"]
        row["gross_profit"] += stats["gross_profit"]
        row["gross_loss"] += stats["gross_loss"]

    for item in open_positions:
        row = strategy_rows[item["strategy"]]
        row["open_positions"] += 1
        row["open_unrealized_pl"] += item["unrealized_pl"] or 0.0

    strategies = [_finalize_strategy(strategy, row) for strategy, row in strategy_rows.items()]
    strategies.sort(key=lambda row: row["total_pl_estimate"], reverse=True)

    equity_curve = _equity_curve(chronological_cycles)
    summary = _summary(equity_curve, strategies, open_positions)
    warnings = _data_quality_warnings(chronological_events, strategies)

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "summary": summary,
        "strategies": strategies,
        "open_positions": open_positions,
        "equity_curve": equity_curve,
        "data_quality": {
            "warnings": warnings,
            "filled_order_count": len(_filled_order_ids(chronological_events)),
            "exit_signal_count": sum(row["exit_signals"] for row in strategies),
            "cycle_count": len(equity_curve),
        },
    }


def _base_strategy_rows(
    events: list[dict[str, Any]],
    strategy_by_symbol: dict[str, str],
) -> defaultdict[str, dict[str, Any]]:
    rows: defaultdict[str, dict[str, Any]] = defaultdict(_empty_strategy_row)
    for event in events:
        if event.get("event_type") != "order":
            continue
        payload = event.get("payload") or {}
        intent = payload.get("intent") or {}
        metadata = intent.get("metadata") or {}
        symbol = str(event.get("symbol") or intent.get("symbol") or "")
        strategy = (
            strategy_by_symbol.get(symbol)
            if metadata.get("exit")
            else event.get("strategy")
        ) or event.get("strategy") or UNKNOWN_STRATEGY
        row = rows[strategy]
        status = event.get("status") or ""
        if status == "submitted":
            row["submitted_orders"] += 1
        elif status == "rejected":
            row["rejected_orders"] += 1
        elif status == "skipped":
            row["skipped_orders"] += 1

        if _filled_order(event):
            row["filled_orders"] += 1

        if metadata.get("exit") and status in {"submitted", "completed"}:
            pl = _exit_signal_pl(metadata)
            if pl is None:
                continue
            row["exit_signals"] += 1
            row["exit_signal_pl"] += pl
            if pl > 0:
                row["exit_signal_wins"] += 1
            elif pl < 0:
                row["exit_signal_losses"] += 1
    return rows


def _empty_strategy_row() -> dict[str, Any]:
    return {
        "submitted_orders": 0,
        "rejected_orders": 0,
        "skipped_orders": 0,
        "filled_orders": 0,
        "closed_trades": 0,
        "wins": 0,
        "losses": 0,
        "realized_pl": 0.0,
        "gross_profit": 0.0,
        "gross_loss": 0.0,
        "exit_signals": 0,
        "exit_signal_wins": 0,
        "exit_signal_losses": 0,
        "exit_signal_pl": 0.0,
        "open_positions": 0,
        "open_unrealized_pl": 0.0,
    }


def _latest_entry_strategy_by_symbol(events: list[dict[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for event in events:
        if event.get("event_type") != "order" or event.get("status") != "submitted":
            continue
        payload = event.get("payload") or {}
        intent = payload.get("intent") or {}
        metadata = intent.get("metadata") or {}
        symbol = str(event.get("symbol") or intent.get("symbol") or "")
        if not symbol or metadata.get("exit"):
            continue
        mapping[symbol] = str(event.get("strategy") or UNKNOWN_STRATEGY)
    return mapping


def _closed_trade_stats(events: list[dict[str, Any]]) -> defaultdict[str, dict[str, Any]]:
    open_lots: defaultdict[str, deque[Lot]] = defaultdict(deque)
    seen_order_ids: set[str] = set()
    stats: defaultdict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "closed_trades": 0,
            "wins": 0,
            "losses": 0,
            "realized_pl": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
        }
    )
    for event in events:
        if event.get("event_type") not in {"order", "broker_order_update"}:
            continue
        order = _filled_order(event)
        if not order:
            continue
        if order["id"] and order["id"] in seen_order_ids:
            continue
        if order["id"]:
            seen_order_ids.add(order["id"])
        symbol = order["symbol"]
        qty = order["qty"]
        price = order["price"]
        direction = order["direction"]
        multiplier = order["multiplier"]
        closing = order["closing"]
        if not closing:
            open_lots[symbol].append(
                Lot(
                    symbol=symbol,
                    strategy=order["strategy"],
                    qty=qty,
                    price=price,
                    direction=direction,
                    multiplier=multiplier,
                )
            )
            continue
        remaining = qty
        while remaining > 0 and open_lots[symbol]:
            lot = open_lots[symbol][0]
            matched_qty = min(remaining, lot.qty)
            if lot.direction > 0:
                pl = (price - lot.price) * matched_qty * lot.multiplier
            else:
                pl = (lot.price - price) * matched_qty * lot.multiplier
            row = stats[lot.strategy]
            row["closed_trades"] += 1
            row["realized_pl"] += pl
            if pl > 0:
                row["wins"] += 1
                row["gross_profit"] += pl
            elif pl < 0:
                row["losses"] += 1
                row["gross_loss"] += abs(pl)
            lot.qty -= matched_qty
            remaining -= matched_qty
            if lot.qty <= 0:
                open_lots[symbol].popleft()
    return stats


def _filled_order(event: dict[str, Any]) -> dict[str, Any] | None:
    payload = event.get("payload") or {}
    broker_order = payload.get("broker_order") or {}
    intent = payload.get("intent") or {}
    symbol = str(event.get("symbol") or broker_order.get("symbol") or intent.get("symbol") or "")
    qty = _to_float(broker_order.get("filled_qty"))
    price = _to_float(broker_order.get("filled_avg_price"))
    if not symbol or not qty or not price:
        return None
    side = str(broker_order.get("side") or intent.get("side") or "")
    position_intent = str(broker_order.get("position_intent") or "")
    metadata = intent.get("metadata") or {}
    closing = bool(metadata.get("exit")) or position_intent.endswith("_to_close")
    direction = -1 if side == "sell" else 1
    return {
        "id": str(broker_order.get("id") or ""),
        "symbol": symbol,
        "qty": abs(qty),
        "price": price,
        "direction": direction,
        "closing": closing,
        "strategy": str(event.get("strategy") or _strategy_from_client_order_id(broker_order) or UNKNOWN_STRATEGY),
        "multiplier": _order_multiplier(intent, broker_order),
    }


def _order_multiplier(intent: dict[str, Any], broker_order: dict[str, Any]) -> float:
    asset_class = str(intent.get("asset_class") or broker_order.get("asset_class") or "")
    if asset_class in {"option", "us_option"}:
        return 100.0
    return 1.0


def _strategy_from_client_order_id(broker_order: dict[str, Any]) -> str | None:
    client_order_id = str(broker_order.get("client_order_id") or "")
    if not client_order_id.startswith("ta-"):
        return None
    strategy = client_order_id[3:].rsplit("-", 1)[0]
    return strategy or None


def _exit_signal_pl(metadata: dict[str, Any]) -> float | None:
    position = metadata.get("position") or {}
    pl = _to_float(position.get("unrealized_pl"))
    if pl is not None:
        return pl
    metrics = metadata.get("metrics") or {}
    cost_basis = _to_float(metrics.get("cost_basis"))
    pnl_pct = _to_float(metrics.get("pnl_pct"))
    if cost_basis is None or pnl_pct is None:
        return None
    return cost_basis * (pnl_pct / 100.0)


def _open_position_stats(
    cycles: list[dict[str, Any]],
    strategy_by_symbol: dict[str, str],
) -> list[dict[str, Any]]:
    if not cycles:
        return []
    latest = cycles[-1]
    positions = latest.get("positions") or []
    rows = []
    for position in positions:
        symbol = str(position.get("symbol") or "")
        rows.append(
            {
                "symbol": symbol,
                "strategy": strategy_by_symbol.get(symbol, UNKNOWN_STRATEGY),
                "asset_class": position.get("asset_class"),
                "qty": _to_float(position.get("qty")) or 0.0,
                "market_value": _to_float(position.get("market_value")) or 0.0,
                "avg_entry_price": _to_float(position.get("avg_entry_price")),
                "current_price": _to_float(position.get("current_price")),
                "unrealized_pl": _to_float(position.get("unrealized_pl")) or 0.0,
            }
        )
    rows.sort(key=lambda row: abs(row["unrealized_pl"]), reverse=True)
    return rows


def _equity_curve(cycles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    curve = []
    peak: float | None = None
    for cycle in cycles:
        account = cycle.get("account") or {}
        equity = _to_float(account.get("equity"))
        if equity is None:
            continue
        peak = equity if peak is None else max(peak, equity)
        drawdown_amount = max(0.0, peak - equity)
        drawdown_pct = (drawdown_amount / peak * 100.0) if peak else 0.0
        curve.append(
            {
                "cycle_id": cycle.get("id"),
                "started_at": cycle.get("started_at"),
                "equity": equity,
                "drawdown_amount": drawdown_amount,
                "drawdown_pct": drawdown_pct,
            }
        )
    return curve


def _summary(
    equity_curve: list[dict[str, Any]],
    strategies: list[dict[str, Any]],
    open_positions: list[dict[str, Any]],
) -> dict[str, Any]:
    first_equity = equity_curve[0]["equity"] if equity_curve else None
    last_equity = equity_curve[-1]["equity"] if equity_curve else None
    total_pl = (last_equity - first_equity) if first_equity is not None and last_equity is not None else 0.0
    total_pl_pct = (total_pl / first_equity * 100.0) if first_equity else 0.0
    max_drawdown = max(equity_curve, key=lambda row: row["drawdown_pct"], default={})

    closed_trades = sum(row["closed_trades"] for row in strategies)
    wins = sum(row["wins"] for row in strategies)
    exit_signals = sum(row["exit_signals"] for row in strategies)
    exit_wins = sum(row["exit_signal_wins"] for row in strategies)
    win_rate = (wins / closed_trades * 100.0) if closed_trades else (
        exit_wins / exit_signals * 100.0 if exit_signals else None
    )

    return {
        "first_equity": first_equity,
        "last_equity": last_equity,
        "total_pl": total_pl,
        "total_pl_pct": total_pl_pct,
        "max_drawdown_amount": max_drawdown.get("drawdown_amount", 0.0),
        "max_drawdown_pct": max_drawdown.get("drawdown_pct", 0.0),
        "win_rate_pct": win_rate,
        "closed_trades": closed_trades,
        "exit_signals": exit_signals,
        "realized_pl": sum(row["realized_pl"] for row in strategies),
        "open_unrealized_pl": sum(row["unrealized_pl"] for row in open_positions),
        "strategies_helping": sum(1 for row in strategies if row["assessment"] == "helping"),
        "strategies_hurting": sum(1 for row in strategies if row["assessment"] == "hurting"),
        "strategy_count": len(strategies),
    }


def _finalize_strategy(strategy: str, row: dict[str, Any]) -> dict[str, Any]:
    observed_trades = row["closed_trades"] or row["exit_signals"]
    wins = row["wins"] if row["closed_trades"] else row["exit_signal_wins"]
    losses = row["losses"] if row["closed_trades"] else row["exit_signal_losses"]
    win_rate = (wins / observed_trades * 100.0) if observed_trades else None
    avg_win = row["gross_profit"] / row["wins"] if row["wins"] else None
    avg_loss = row["gross_loss"] / row["losses"] if row["losses"] else None
    profit_factor = (
        row["gross_profit"] / row["gross_loss"]
        if row["gross_loss"]
        else (None if not row["gross_profit"] else float("inf"))
    )
    total_pl_estimate = row["realized_pl"] + row["exit_signal_pl"] + row["open_unrealized_pl"]
    assessment = _assessment(total_pl_estimate, observed_trades, row["open_positions"])
    return {
        "strategy": strategy,
        "assessment": assessment,
        "submitted_orders": row["submitted_orders"],
        "rejected_orders": row["rejected_orders"],
        "skipped_orders": row["skipped_orders"],
        "filled_orders": row["filled_orders"],
        "closed_trades": row["closed_trades"],
        "wins": wins,
        "losses": losses,
        "win_rate_pct": win_rate,
        "realized_pl": row["realized_pl"],
        "open_positions": row["open_positions"],
        "open_unrealized_pl": row["open_unrealized_pl"],
        "exit_signals": row["exit_signals"],
        "exit_signal_wins": row["exit_signal_wins"],
        "exit_signal_losses": row["exit_signal_losses"],
        "exit_signal_pl": row["exit_signal_pl"],
        "total_pl_estimate": total_pl_estimate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
    }


def _assessment(total_pl_estimate: float, observed_trades: int, open_positions: int) -> str:
    if observed_trades == 0 and open_positions == 0:
        return "needs_data"
    if total_pl_estimate > 0:
        return "helping"
    if total_pl_estimate < 0:
        return "hurting"
    return "neutral"


def _data_quality_warnings(
    events: list[dict[str, Any]],
    strategies: list[dict[str, Any]],
) -> list[str]:
    warnings = []
    submitted_orders = [
        event
        for event in events
        if event.get("event_type") == "order" and event.get("status") == "submitted"
    ]
    if submitted_orders and not _filled_order_ids(events):
        warnings.append(
            "No filled broker orders are recorded in audit yet; realized P/L and closed-trade win rate will stay empty until fills are captured."
        )
    if any(row["exit_signals"] for row in strategies):
        warnings.append(
            "Exit-signal P/L uses the position snapshot available when the agent proposed an exit; final fill P/L can differ."
        )
    if not strategies:
        warnings.append("No strategy order history is available yet.")
    return warnings


def _filled_order_ids(events: list[dict[str, Any]]) -> set[str]:
    ids = set()
    synthetic_count = 0
    for event in events:
        order = _filled_order(event)
        if not order:
            continue
        if order["id"]:
            ids.add(order["id"])
        else:
            synthetic_count += 1
            ids.add(f"synthetic-{synthetic_count}")
    return ids


def _to_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
