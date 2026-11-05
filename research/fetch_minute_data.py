"""Fetch SIP minute bars + dated daily bars for the day-trade backtest (compact arrays)."""
import asyncio, pickle, sys
from array import array
from datetime import datetime, timedelta, UTC
from pathlib import Path
from trading_agent.brokers.alpaca import AlpacaBroker, DATA_API_BASE
from trading_agent.config import load_settings
from trading_agent.day_trading.features import bar_time, minute_of_session, SESSION_OPEN, SESSION_CLOSE
from trading_agent.screener.universe import symbols_for_universes

OUT = Path(__file__).parent / "cache" / "dt_data.pkl"
import os
from datetime import date as _date
# Override the window with BT_START / BT_END (ISO dates) to extend the sample.
START = os.environ.get("BT_START", "2026-07-15") + "T00:00:00Z"
END = os.environ.get("BT_END", _date.today().isoformat()) + "T23:59:00Z"

async def paged(b, params):
    rows = {}
    for _ in range(2000):
        data = await b._request("GET", f"{DATA_API_BASE}/v2/stocks/bars", params=params)
        for sym, bars in (data.get("bars") or {}).items():
            rows.setdefault(sym, []).extend(bars)
        tok = data.get("next_page_token")
        if not tok:
            return rows
        params["page_token"] = tok
    return rows

async def main():
    b = AlpacaBroker(load_settings())
    syms = [s for s in symbols_for_universes(["nasdaq100", "sp500_core"]) if "/" not in s]
    minute = {}  # sym -> {date: [o,h,l,c,v,vw arrays of 390 (nan = missing)]}
    for i in range(0, len(syms), 8):
        chunk = syms[i:i + 8]
        rows = await paged(b, {"symbols": ",".join(chunk), "timeframe": "1Min", "start": START, "end": END,
                               "feed": "sip", "adjustment": "all", "limit": 10000})
        for sym, bars in rows.items():
            days = minute.setdefault(sym, {})
            for bar in bars:
                t = bar_time(bar)
                if not (SESSION_OPEN <= t.time() < SESSION_CLOSE):
                    continue
                d = days.get(t.date())
                if d is None:
                    d = days[t.date()] = [array("d", [float("nan")] * 390) for _ in range(6)]
                m = minute_of_session(t)
                for k, key in enumerate(("o", "h", "l", "c", "v", "vw")):
                    d[k][m] = float(bar.get(key) or bar.get("c") or 0)
        print(f"minute bars: {i + len(chunk)}/{len(syms)} symbols", file=sys.stderr, flush=True)
    daily_rows = await paged(b, {"symbols": ",".join(syms), "timeframe": "1Day", "start": "2025-10-01T00:00:00Z",
                                 "end": END, "feed": "sip", "adjustment": "all", "limit": 10000})
    daily = {s: [(bar_time(r).date(), float(r["o"]), float(r["h"]), float(r["l"]), float(r["c"]), float(r["v"])) for r in rows]
             for s, rows in daily_rows.items()}
    OUT.write_bytes(pickle.dumps({"minute": minute, "daily": daily}))
    print("saved", OUT, file=sys.stderr)

asyncio.run(main())
