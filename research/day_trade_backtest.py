"""Day-trade backtest on SIP minute bars, driving the real DayTradingEngine.

Entries: every 5 minutes from 09:45, the engine's evaluate_entry on every universe
symbol (catalyst assumed bullish 75 and research neutral, since there is no news
history - so this tests the intraday edge, not the news gate). Up to 3 trades/day,
3 per cycle, one per symbol per day, best combined score first. Fill at the next
minute's open + slippage.
Exits: bracket floor/ceiling checked intrabar (stop first if both touch); partial
exit and intraday-score exit checked at 5-minute cycles; force exit at 15:45 ET.
Sizing: 0.25% equity risk / floor, capped at 8% of equity (x setup multiplier).
"""
from __future__ import annotations

import math
import pickle
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, time, timedelta
from pathlib import Path

from trading_agent.catalyst.models import CatalystPrediction
from trading_agent.config import load_settings
from trading_agent.day_trading import DayTradingEngine
from trading_agent.day_trading.features import EASTERN, intraday_features
from trading_agent.models import AssetClass, MarketSnapshot, NewsItem, ResearchSnapshot

HERE = Path(__file__).parent / "cache"
EQUITY = 100_000.0
SLIP = 0.0002
NAN = float("nan")
FORCE_MINUTE = 375  # 15:45 ET
CYCLE_MINUTES = list(range(15, FORCE_MINUTE, 5))


def is_nan(x):
    return x != x


@dataclass
class Day:
    o: list
    h: list
    l: list
    c: list           # forward-filled close
    cum_v: list
    cum_pv: list
    or_high: float
    profile: list     # avg cumulative volume by minute over prior 10 sessions
    prior_closes: list
    adr_pct: float


def prepare(data):
    minute, daily = data["minute"], data["daily"]
    days_by_symbol = {}
    for sym, by_day in minute.items():
        dailies = daily.get(sym, [])
        ordered = sorted(by_day)
        out = {}
        cum_history = []  # cumulative volume arrays of prior sessions
        for d in ordered:
            o, h, l, c, v, vw = by_day[d]
            close_ff, last = [], NAN
            cum_v, cum_pv, run_v, run_pv = [], [], 0.0, 0.0
            for m in range(390):
                if not is_nan(c[m]):
                    last = c[m]
                    run_v += v[m]
                    run_pv += (vw[m] if vw[m] else c[m]) * v[m]
                close_ff.append(last)
                cum_v.append(run_v)
                cum_pv.append(run_pv)
            highs = [x for x in h[:15] if not is_nan(x)]
            profile = []
            if len(cum_history) >= 5:
                hist = cum_history[-10:]
                profile = [sum(day_cum[m] for day_cum in hist) / len(hist) for m in range(390)]
            prior = [row for row in dailies if row[0] < d]
            if prior and profile and len(prior) >= 60:
                ranges = [(r[2] - r[3]) / r[1] * 100 for r in prior[-20:] if r[1] > 0]
                out[d] = Day(list(o), list(h), list(l), close_ff, cum_v, cum_pv,
                             max(highs) if highs else NAN, profile,
                             [r[4] for r in prior[-289:]], sum(ranges) / len(ranges))
            cum_history.append(cum_v)
        days_by_symbol[sym] = out
    return days_by_symbol


def fast_features(day: Day, M: int, orb_valid: int = 120) -> dict:
    """Same definitions as day_trading.features.intraday_features at now = 09:30 + M."""
    f = {}
    if day.cum_v[M - 1] > 0:
        f["vwap"] = round(day.cum_pv[M - 1] / day.cum_v[M - 1], 4)
    last, base = day.c[M - 1], day.c[M - 6] if M >= 6 else NAN
    if not is_nan(base) and base:
        f["minute_trend_pct"] = round((last - base) / base * 100, 4)
    if M >= 15 and not is_nan(day.or_high):
        f["opening_range_high"] = day.or_high
        f["opening_range_break"] = bool(last > day.or_high and M <= orb_valid)
    expected = day.profile[max(0, min(389, M - 1))]
    if expected > 0:
        f["relative_volume"] = round(day.cum_v[M - 1] / expected, 3)
    f["minutes_since_open"] = M
    return f


def parity_check(days_by_symbol, samples=40):
    rng = random.Random(7)
    checked = 0
    for sym in rng.sample(sorted(days_by_symbol), 10):
        for d, day in list(days_by_symbol[sym].items())[:4]:
            bars = []
            for m in range(390):
                if is_nan(day.o[m]):
                    continue
                t = datetime.combine(d, time(9, 30), tzinfo=EASTERN) + timedelta(minutes=m)
                prev = day.cum_v[m] - (day.cum_v[m - 1] if m else 0)
                prev_pv = day.cum_pv[m] - (day.cum_pv[m - 1] if m else 0)
                bars.append({"t": t.isoformat(), "o": day.o[m], "h": day.h[m], "l": day.l[m], "c": day.c[m],
                             "v": prev, "vw": prev_pv / prev if prev else day.c[m]})
            for M in (16, 47, 130, 300):
                now = datetime.combine(d, time(9, 30), tzinfo=EASTERN) + timedelta(minutes=M)
                ref = intraday_features(bars, now=now, volume_profile=day.profile)
                got = fast_features(day, M)
                for key in ("vwap", "minute_trend_pct", "opening_range_break", "relative_volume"):
                    a, b = ref.get(key), got.get(key)
                    if isinstance(a, float) and isinstance(b, float):
                        assert abs(a - b) <= 1e-3 * max(1, abs(a)), (sym, d, M, key, a, b)
                    else:
                        assert a == b, (sym, d, M, key, a, b)
                checked += 1
                if checked >= samples:
                    return checked
    return checked


PREDICTION = CatalystPrediction(symbol="X", asset_class="equity", direction="bullish", prediction_score=75,
                                bullish_score=75, confidence="medium", entry_allowed=True, market_regime="risk_on")
RESEARCH = ResearchSnapshot(symbol="X", news=[NewsItem(title=f"Company update {i}") for i in range(5)],
                            sec_summary={"latest_net_income": {"value": 1}})


def snapshot(sym, day, M, feats):
    price = day.c[M - 1]
    meta = dict(feats)
    meta.update({"spread_pct": 0.02, "adr_pct": day.adr_pct, "volumes": []})
    return MarketSnapshot(symbol=sym, asset_class=AssetClass.EQUITY, price=price,
                          closes=day.prior_closes + [price], metadata=meta)


def entry_signals(days_by_symbol, settings):
    engine = DayTradingEngine(settings)
    signals = defaultdict(list)  # date -> [(M, combined, sym, setup)]
    for sym, days in days_by_symbol.items():
        for d, day in days.items():
            for M in CYCLE_MINUTES:
                if is_nan(day.c[M - 1]):
                    continue
                market = snapshot(sym, day, M, fast_features(day, M))
                now = datetime.combine(d, time(9, 30), tzinfo=EASTERN) + timedelta(minutes=M)
                cand, sig = engine.evaluate_entry(market, RESEARCH, PREDICTION, [], trades_used_today=0, now=now)
                if cand:
                    signals[d].append((M, sig["combined_score"], sym, sig["setup"]))
    return signals


@dataclass(frozen=True)
class V:
    name: str
    intraday_exit: bool = True
    partial: bool = True
    cutoff: int = 0               # no entries within N minutes of the close
    ceiling: float = 0.55
    floor: float = 0.40
    partial_frac: float = 0.45
    setups: tuple = ("trend_continuation", "opening_range_break", "intraday_trend")
    orb_mult: float = 0.5
    max_per_day: int = 3


def simulate(days_by_symbol, signals, v: V, settings):
    engine = DayTradingEngine(settings)
    exit_score = settings.day_trading.exit_intraday_score
    trades = []
    for d in sorted(signals):
        taken, used = set(), 0
        by_cycle = defaultdict(list)
        for M, score, sym, setup in signals[d]:
            by_cycle[M].append((score, sym, setup))
        for M in sorted(by_cycle):
            if used >= v.max_per_day or M > 390 - v.cutoff:
                continue
            per_cycle = 0
            for score, sym, setup in sorted(by_cycle[M], reverse=True):
                if used >= v.max_per_day or per_cycle >= 3 or sym in taken or setup not in v.setups:
                    continue
                day = days_by_symbol[sym][d]
                trade = run_trade(engine, sym, d, day, M, setup, v, exit_score)
                if trade:
                    trades.append(trade)
                    taken.add(sym)
                    used += 1
                    per_cycle += 1
    return trades


def run_trade(engine, sym, d, day, M, setup, v, exit_score):
    entry = day.o[M] if not is_nan(day.o[M]) else day.c[M - 1]
    entry *= 1 + SLIP
    adr = day.adr_pct
    floor = min(adr * v.floor, 1.5)
    ceiling = min(adr * v.ceiling, 3.0)
    partial = min(adr * v.partial_frac, 1.0)
    mult = v.orb_mult if setup == "opening_range_break" else 1.0
    notional = min(EQUITY * 0.0025 / (floor / 100), EQUITY * 0.08 * mult)
    qty = int(notional / entry)
    if qty < 1:
        return None
    stop, tp = entry * (1 - floor / 100), entry * (1 + ceiling / 100)
    remaining, pnl, partial_done, reason, mfe, mae = qty, 0.0, False, "", 0.0, 0.0

    def close_all(px, why):
        nonlocal pnl, remaining, reason
        pnl += (px * (1 - SLIP) - entry) * remaining
        remaining, reason = 0, why

    for k in range(M, 390):
        o, h, l = day.o[k], day.h[k], day.l[k]
        if k >= FORCE_MINUTE:
            px = o if not is_nan(o) else day.c[k - 1]
            close_all(px, "force_close")
            break
        if k > M and k % 5 == 0:  # agent cycle boundary (cycles run at minutes 15, 20, 25, ...)
            px = day.c[k - 1]
            if v.partial and not partial_done and remaining >= 2 and px >= entry * (1 + partial / 100):
                sell = remaining // 2
                pnl += (px * (1 - SLIP) - entry) * sell
                remaining -= sell
                partial_done, stop = True, entry
            if v.intraday_exit:
                feats = fast_features(day, k)
                score = engine._intraday_score(snapshot(sym, day, k, feats))["score"]
                if score <= exit_score:
                    close_all(px, "intraday_score_exit")
                    break
        if is_nan(o):
            continue
        mfe, mae = max(mfe, (h / entry - 1) * 100), min(mae, (l / entry - 1) * 100)
        if o <= stop:
            close_all(o, "breakeven_stop" if partial_done else "floor_stop")
            break
        if l <= stop:
            close_all(stop, "breakeven_stop" if partial_done else "floor_stop")
            break
        if o >= tp:
            close_all(o, "ceiling")
            break
        if h >= tp:
            close_all(tp, "ceiling")
            break
    return {"date": d, "sym": sym, "M": M, "setup": setup, "pnl": pnl, "notional": qty * entry,
            "ret": pnl / (qty * entry) * 100, "reason": reason, "adr": adr, "mfe": mfe, "mae": mae,
            "partial": partial_done}


def summarize(name, trades):
    if not trades:
        return f"{name:<44} no trades"
    pnls = [t["pnl"] for t in trades]
    wins, losses = [p for p in pnls if p > 0], [-p for p in pnls if p <= 0]
    mean = sum(pnls) / len(pnls)
    sd = (sum((p - mean) ** 2 for p in pnls) / max(len(pnls) - 1, 1)) ** 0.5
    cum = peak = dd = 0.0
    for t in sorted(trades, key=lambda x: (x["date"], x["M"])):
        cum += t["pnl"]
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    pf = sum(wins) / sum(losses) if losses else float("inf")
    tstat = mean / sd * math.sqrt(len(pnls)) if sd else 0
    return (f"{name:<44} n {len(pnls):3d}  win {len(wins)/len(pnls)*100:4.0f}%  PF {pf:4.2f}  "
            f"total ${sum(pnls):+8.0f}  exp ${mean:+6.1f}/trade  t {tstat:+5.2f}  maxDD ${dd:6.0f}")


def main():
    data = pickle.loads((HERE / "dt_data.pkl").read_bytes())
    days_by_symbol = prepare(data)
    n_days = len({d for days in days_by_symbol.values() for d in days})
    print(f"symbols {len(days_by_symbol)}  test sessions {n_days}")
    print(f"feature parity vs live module: {parity_check(days_by_symbol)} samples OK")
    settings = load_settings()
    sig_path = HERE / "dt_signals.pkl"
    if sig_path.exists():
        signals = pickle.loads(sig_path.read_bytes())
    else:
        signals = entry_signals(days_by_symbol, settings)
        sig_path.write_bytes(pickle.dumps(dict(signals)))
    print(f"entry signals: {sum(len(x) for x in signals.values())} across {len(signals)} sessions")
    base = V("CURRENT (score-exit on, partial, no cutoff)")
    variants = [
        base,
        replace(base, name="score-exit OFF", intraday_exit=False),
        replace(base, name="score-exit OFF + cutoff 90m", intraday_exit=False, cutoff=90),
        replace(base, name="score-exit OFF + cutoff 90m, no partial", intraday_exit=False, cutoff=90, partial=False),
        replace(base, name="score-exit OFF, cutoff 90m, trend only", intraday_exit=False, cutoff=90,
                setups=("trend_continuation",)),
        replace(base, name="score-exit OFF, cutoff 90m, ORB only", intraday_exit=False, cutoff=90,
                setups=("opening_range_break",)),
    ]
    results = {}
    for v in variants:
        results[v.name] = simulate(days_by_symbol, signals, v, settings)
        print(summarize(v.name, results[v.name]))
    print("--- ceiling/floor grid (score-exit OFF, cutoff 90m, partial on)")
    for c in (0.40, 0.55, 0.75, 1.0):
        for f in (0.30, 0.40, 0.60):
            tr = simulate(days_by_symbol, signals, replace(base, name="", intraday_exit=False, cutoff=90, ceiling=c, floor=f), settings)
            print(summarize(f"  ceiling {c:.2f} / floor {f:.2f}", tr))
    for name in (variants[0].name, variants[2].name):
        tr = results[name]
        print(f"\n[{name}] exit reasons:", dict(sorted(((r, sum(1 for t in tr if t['reason'] == r)) for r in {t['reason'] for t in tr}), key=lambda x: -x[1])))
        buckets = defaultdict(list)
        for t in tr:
            buckets["09:45-10:30" if t["M"] < 60 else "10:30-12:00" if t["M"] < 150 else "12:00-14:00" if t["M"] < 270 else "14:00+"].append(t)
        for k in sorted(buckets):
            print("   ", summarize(f"entry {k}", buckets[k]))
        for k, grp in (("ADR < 2.5%", [t for t in tr if t["adr"] < 2.5]), ("ADR >= 2.5%", [t for t in tr if t["adr"] >= 2.5])):
            print("   ", summarize(k, grp))
    pickle.dump(results, open(HERE / "dt_results.pkl", "wb"))


if __name__ == "__main__":
    main()
