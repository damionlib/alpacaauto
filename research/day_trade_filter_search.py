"""Entry-filter search for day trades, chosen in-sample (first half), judged out-of-sample (second half)."""
import pickle, sys, itertools, math
from dataclasses import replace
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
import day_trade_backtest as bt
from trading_agent.config import load_settings

data = pickle.loads((bt.HERE / "dt_data.pkl").read_bytes())
days_by_symbol = bt.prepare(data)
signals = pickle.loads((bt.HERE / "dt_signals.pkl").read_bytes())
settings = load_settings()
sessions = sorted(signals)
half = len(sessions) // 2
first, second = set(sessions[:half]), set(sessions[half:])

def filtered(min_adr, start, min_combined, min_rvol):
    out = {}
    for d, rows in signals.items():
        keep = []
        for M, score, sym, setup in rows:
            day = days_by_symbol[sym][d]
            if M < start or score < min_combined or day.adr_pct < min_adr:
                continue
            if min_rvol and (bt.fast_features(day, M).get("relative_volume") or 0) < min_rvol:
                continue
            keep.append((M, score, sym, setup))
        out[d] = keep
    return out

base = bt.V("x", intraday_exit=False, partial=False, cutoff=90)
def stats(trades):
    p = [t["pnl"] for t in trades]
    if len(p) < 2: return (len(p), 0, 0, 0)
    w = sum(x for x in p if x > 0); l = -sum(x for x in p if x <= 0)
    m = sum(p) / len(p); sd = (sum((x - m) ** 2 for x in p) / (len(p) - 1)) ** .5
    return (len(p), w / l if l else 9.99, sum(p), m / sd * math.sqrt(len(p)) if sd else 0)

grid = list(itertools.product((0, 2.5, 3.0), (15, 30, 60), (75, 80, 85), (0, 1.5, 2.0)))
rows = []
for combo in grid:
    sig = filtered(*combo)
    ins = bt.simulate(days_by_symbol, {d: s for d, s in sig.items() if d in first}, base, settings)
    oos = bt.simulate(days_by_symbol, {d: s for d, s in sig.items() if d in second}, base, settings)
    rows.append((combo, stats(ins), stats(oos)))
rows.sort(key=lambda r: r[1][3], reverse=True)
print(f"sessions: in-sample {len(first)}, out-of-sample {len(second)}; combos {len(grid)}")
print("top 10 by IN-SAMPLE t-stat, with their OUT-OF-SAMPLE result:")
print(f"{'minADR start minComb minRVOL':<30}{'IS n  PF   P/L     t':<30}{'OOS n  PF   P/L     t'}")
for combo, a, b in rows[:10]:
    print(f"{str(combo):<30}{a[0]:3d} {a[1]:4.2f} {a[2]:+7.0f} {a[3]:+5.2f}      {b[0]:3d} {b[1]:4.2f} {b[2]:+7.0f} {b[3]:+5.2f}")
pos_oos = sum(1 for _, _, b in rows if b[2] > 0)
print(f"\ncombos profitable out-of-sample: {pos_oos}/{len(rows)}")
top = rows[:10]
print(f"mean OOS P/L of the 10 best in-sample combos: {sum(b[2] for _, _, b in top) / 10:+.0f}")
cur = [r for r in rows if r[0] == (0, 15, 75, 0)][0]
print(f"current entry rules (0,15,75,0): IS {cur[1]}  OOS {cur[2]}")
