"""Swing backtest: same entry signals (the agent's real screener + momentum code), old vs new exits.

Daily bars, SIP, split/dividend adjusted. Entries at the signal day's close (matches the
pre-close entry window). Exits simulated on the next days' OHLC: gap through a stop fills
at the open, otherwise at the stop; stop checked before any target on the same bar.
Known biases: universe is today's large caps (survivorship); no news/catalyst history, so
the catalyst gate is not applied; holidays treated as trading days.
"""
from __future__ import annotations

import asyncio
import json
import math
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path

from trading_agent.brokers.alpaca import AlpacaBroker
from trading_agent.config import load_settings
from trading_agent.indicators import atr, sma
from trading_agent.models import AssetClass, MarketSnapshot, ResearchSnapshot
from trading_agent.position_manager import swing_stop_level
from trading_agent.screener.service import MarketScreener
from trading_agent.screener.universe import symbols_for_universes
from trading_agent.strategies.momentum import MomentumStrategy

SCRATCH = Path(__file__).parent / "cache"
CACHE = SCRATCH / "bt_bars.pkl"
COST = 0.0005  # 5 bps per side
WINDOW = 290   # bars visible to the live agent (420 calendar days)


async def load_bars():
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    settings = load_settings()
    broker = AlpacaBroker(settings)
    symbols = [s for s in symbols_for_universes(["nasdaq100", "sp500_core"]) if "/" not in s]
    if "SPY" not in symbols:
        symbols.append("SPY")
    bars = {}
    for start in range(0, len(symbols), 50):
        chunk = symbols[start:start + 50]
        got = await broker.get_daily_bars(chunk, lookback_days=365 * 6)
        bars.update(got)
        print(f"fetched {len(got)}/{len(chunk)} symbols", file=sys.stderr)
    # dates: refetch timestamps for SPY only to align
    import httpx  # noqa
    CACHE.write_bytes(pickle.dumps(bars))
    return bars


def precompute_signals(bars, settings):
    """For each day index t (aligned on SPY's length), list of (score, symbol, close, atr, vol_stop)."""
    screener = MarketScreener(settings, broker=None)
    strategy = MomentumStrategy(settings)
    research = ResearchSnapshot(symbol="X", sec_summary={"latest_net_income": {"value": 1}})
    n = len(bars["SPY"]["closes"])
    signals = [[] for _ in range(n)]
    for symbol, s in bars.items():
        if symbol in {"SPY", "QQQ"}:
            continue
        m = len(s["closes"])
        offset = n - m  # align right edges (all series end today)
        for t in range(60, m):
            lo = max(0, t + 1 - WINDOW)
            closes = s["closes"][lo:t + 1]
            snap = MarketSnapshot(
                symbol=symbol, asset_class=AssetClass.EQUITY, price=closes[-1], closes=closes,
                metadata={"volumes": s["volumes"][lo:t + 1], "highs": s["highs"][lo:t + 1], "lows": s["lows"][lo:t + 1]},
            )
            if not screener.score_snapshot(snap):
                continue
            cands = strategy.evaluate(snap, research)
            if not cands:
                continue
            c = cands[0]
            signals[t + offset].append((c.score, symbol, closes[-1], c.metadata.get("atr"), c.metadata.get("vol20")))
    return signals


@dataclass
class Pos:
    symbol: str
    qty: int
    entry: float
    entry_t: int
    atr: float
    initial_stop: float
    peak: float
    exit_reason: str = ""


@dataclass
class Params:
    name: str
    mode: str  # "old" or "atr"
    risk_pct: float
    atr_stop: float = 2.5
    chandelier: float = 3.0
    breakeven: float = 1.5
    max_hold: int = 20
    stale_days: int = 7
    hard_stop_pct: float = 12.0
    # old rules
    old_stop_pct: float = 6.0
    old_tp_pct: float = 8.0
    old_trail_pct: float = 8.0
    old_locks: tuple = ((2.0, 0.75), (3.0, 2.0), (5.0, 3.5), (7.0, 5.5))


GROUPS = [
    {"NVDA", "AVGO", "AMD", "MU", "AMAT", "LRCX", "KLAC", "ASML", "MRVL", "TSM", "SMCI", "ORCL", "MSFT",
     "GOOGL", "GOOG", "META", "AAPL", "AMZN", "TSLA"},
    {"PANW", "CRWD", "ZS", "FTNT", "NET", "OKTA", "S", "DDOG", "NOW", "CRM", "ADBE", "SNOW", "MDB", "TEAM", "WDAY"},
]


def run(bars, signals, p: Params, start_t: int, end_t: int):
    spy = bars["SPY"]["closes"]
    n = len(spy)
    idx = {sym: n - len(s["closes"]) for sym, s in bars.items()}
    cash, positions, trades, curve = 100_000.0, {}, [], []

    def px(sym, t, key):
        i = t - idx[sym]
        return bars[sym][key][i] if 0 <= i < len(bars[sym][key]) else None

    for t in range(start_t, end_t):
        # ---- exits on day t's bar
        for sym in list(positions):
            pos = positions[sym]
            o, h, l, c = px(sym, t, "opens"), px(sym, t, "highs"), px(sym, t, "lows"), px(sym, t, "closes")
            if c is None:
                continue
            fill, reason = None, ""
            if p.mode == "atr":
                level, basis = swing_stop_level(entry_price=pos.entry, atr_value=pos.atr, peak_price=pos.peak,
                                                initial_stop=pos.initial_stop, breakeven_trigger_atr=p.breakeven,
                                                chandelier_atr_multiple=p.chandelier)
                level = max(level, pos.entry * (1 - p.hard_stop_pct / 100))
                if o <= level:
                    fill, reason = o, basis + "_gap"
                elif l <= level:
                    fill, reason = level, basis
            else:
                stop = pos.entry * (1 - p.old_stop_pct / 100)
                peak_pnl = (pos.peak / pos.entry - 1) * 100
                for trig, lock in p.old_locks:
                    if peak_pnl >= trig:
                        stop = max(stop, pos.entry * (1 + lock / 100))
                stop = max(stop, pos.peak * (1 - p.old_trail_pct / 100))
                tp = pos.entry * (1 + p.old_tp_pct / 100)
                if o <= stop:
                    fill, reason = o, "stop_gap"
                elif l <= stop:
                    fill, reason = stop, "stop"
                elif o >= tp:
                    fill, reason = o, "take_profit"
                elif h >= tp:
                    fill, reason = tp, "take_profit"
            held = t - pos.entry_t
            if fill is None:
                pos.peak = max(pos.peak, h)
                if p.stale_days and held >= p.stale_days and (pos.peak / pos.entry - 1) * 100 < 0.5:
                    fill, reason = c, "stale"
                elif p.max_hold and held >= p.max_hold:
                    fill, reason = c, "time"
            if fill is not None:
                proceeds = fill * pos.qty * (1 - COST)
                cash += proceeds
                trades.append({"sym": sym, "pnl": proceeds - pos.entry * pos.qty * (1 + COST),
                               "ret": fill / pos.entry - 1, "held": held, "reason": reason})
                del positions[sym]
        # ---- mark to market
        equity = cash + sum((px(s, t, "closes") or ps.entry) * ps.qty for s, ps in positions.items())
        curve.append(equity)
        # ---- entries at close t (regime gate: SPY > SMA50)
        spy_sma = sma(spy[: t + 1], 50)
        if spy_sma is None or spy[t] <= spy_sma:
            continue
        budget = max(cash - equity * 0.05, 0)
        for score, sym, close, atr_v, vol20 in sorted(signals[t], reverse=True):
            if sym in positions or budget <= 0:
                continue
            if p.mode == "atr" and atr_v:
                stop_pct = max(3.0, min(12.0, p.atr_stop * atr_v / close * 100))
            else:
                stop_pct = max(4.0, min(15.0, ((vol20 or 30) / 100) * 75))
            stop = close * (1 - stop_pct / 100)
            notional = min(equity * p.risk_pct / 100 / (stop_pct / 100), equity * 0.12, budget)
            grp = next((g for g in GROUPS if sym in g), None)
            if grp:
                exposure = sum((px(s, t, "closes") or 0) * ps.qty for s, ps in positions.items() if s in grp)
                if exposure + notional > equity * 0.25:
                    continue
            qty = int(notional / close)
            if qty < 1:
                continue
            cost = close * qty * (1 + COST)
            cash -= cost
            budget -= cost
            positions[sym] = Pos(sym, qty, close, t, atr_v or close * 0.02, stop, close)
    # close out at the end
    for sym, pos in positions.items():
        c = px(sym, end_t - 1, "closes")
        trades.append({"sym": sym, "pnl": (c - pos.entry) * pos.qty, "ret": c / pos.entry - 1,
                       "held": end_t - 1 - pos.entry_t, "reason": "open_at_end"})
    return trades, curve


def stats(trades, curve, days):
    if not curve:
        return {}
    peak, mdd = curve[0], 0.0
    for v in curve:
        peak = max(peak, v)
        mdd = max(mdd, (peak - v) / peak)
    rets = [curve[i] / curve[i - 1] - 1 for i in range(1, len(curve))]
    mean = sum(rets) / len(rets)
    sd = (sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)) ** 0.5
    wins = [x["pnl"] for x in trades if x["pnl"] > 0]
    losses = [-x["pnl"] for x in trades if x["pnl"] <= 0]
    years = days / 252
    return {
        "cagr": (curve[-1] / curve[0]) ** (1 / years) - 1 if years > 0 else 0,
        "mdd": mdd,
        "sharpe": mean / sd * math.sqrt(252) if sd else 0,
        "pf": sum(wins) / sum(losses) if losses else float("inf"),
        "win": len(wins) / len(trades) if trades else 0,
        "n": len(trades),
        "avg_win": sum(x["ret"] for x in trades if x["pnl"] > 0) / max(len(wins), 1),
        "avg_loss": sum(x["ret"] for x in trades if x["pnl"] <= 0) / max(len(losses), 1),
        "hold": sum(x["held"] for x in trades) / max(len(trades), 1),
        "gap_losses": sum(1 for x in trades if x["reason"].endswith("_gap") and x["pnl"] < 0),
    }


def fmt(name, s):
    return (f"{name:<34} CAGR {s['cagr']*100:6.1f}%  MaxDD {s['mdd']*100:5.1f}%  Sharpe {s['sharpe']:4.2f}  "
            f"PF {s['pf']:4.2f}  win {s['win']*100:4.1f}%  avgW {s['avg_win']*100:5.2f}%  avgL {s['avg_loss']*100:6.2f}%  "
            f"n {s['n']:4d}  hold {s['hold']:4.1f}d")


async def main():
    settings = load_settings()
    settings.strategy.enabled = ["equity_momentum"]  # disabled in live config; the backtest evaluates it
    bars = await load_bars()
    sig_cache = SCRATCH / "bt_signals.pkl"
    if sig_cache.exists():
        signals = pickle.loads(sig_cache.read_bytes())
    else:
        signals = precompute_signals(bars, settings)
        sig_cache.write_bytes(pickle.dumps(signals))
    n = len(bars["SPY"]["closes"])
    start = 260
    spy = bars["SPY"]["closes"]
    years = (n - start) / 252
    print(f"symbols {len(bars)}  bars {n}  test window {n - start} days (~{years:.1f}y)  "
          f"signal-days {sum(1 for x in signals[start:] if x)}")
    print(f"SPY buy&hold CAGR {((spy[-1]/spy[start])**(1/years)-1)*100:.1f}%")
    print()
    configs = [Params("OLD exits (6%/8%/locks), risk 2%", "old", 2.0)]
    for a in (2.0, 2.5, 3.0):
        for ch in (2.5, 3.0, 4.0):
            for be in (0.0, 1.5):
                for mh in (20, 40, 60):
                    configs.append(Params(f"ATR stop{a} ch{ch} be{be} hold{mh}", "atr", 0.75, a, ch, be, mh))
    results = []
    for cfg in configs:
        trades, curve = run(bars, signals, cfg, start, n)
        results.append((cfg, stats(trades, curve, n - start), trades))
    old = results[0]
    print(fmt(old[0].name, old[1]))
    ranked = sorted(results[1:], key=lambda r: r[1]["sharpe"], reverse=True)
    print("--- top 8 ATR configs by Sharpe")
    for cfg, s, _ in ranked[:8]:
        print(fmt(cfg.name, s))
    print("--- bottom 3")
    for cfg, s, _ in ranked[-3:]:
        print(fmt(cfg.name, s))
    chosen = next(r for r in results if r[0].name == "ATR stop2.5 ch3.0 be1.5 hold40")
    print("--- chosen default"); print(fmt(chosen[0].name, chosen[1]))
    # robustness by sub-period
    print("\n--- sub-periods (OLD vs chosen)")
    third = (n - start) // 3
    for k in range(3):
        a, b = start + k * third, start + (k + 1) * third if k < 2 else n
        for cfg in (old[0], chosen[0]):
            tr, cv = run(bars, signals, cfg, a, b)
            print(f"  period {k+1} ({b-a}d) " + fmt(cfg.name, stats(tr, cv, b - a)))
    from collections import Counter
    for label, (_, _, tr) in (("OLD", old), ("chosen", chosen)):
        print(f"exit reasons {label}:", dict(Counter(x['reason'] for x in tr).most_common()))
    json.dump([{"name": c.name, **s} for c, s, _ in results], open(SCRATCH / "bt_results.json", "w"), indent=1)


asyncio.run(main())
