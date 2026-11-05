# Research backtests

Evidence used to choose live parameters. Caches (`cache/`) are gitignored and are
rebuilt automatically (swing) or with `fetch_minute_data.py` (day trading).

| Script | What it answers |
|---|---|
| `swing_backtest.py` | Swing entries (the agent's real screener + momentum code) on ~5 years of SIP daily bars; old fixed-% exits vs the ATR/breakeven/chandelier grid; sub-period robustness. Delete `cache/bt_*.pkl` to refresh data. |
| `fetch_minute_data.py` | Downloads SIP minute bars + dated daily bars for the day-trade backtest (`BT_START`/`BT_END` env vars, ISO dates). |
| `day_trade_backtest.py` | Drives the real `DayTradingEngine` over minute bars; exit variants and ceiling/floor grid; parity-checks its fast features against `day_trading/features.py`. Delete `cache/dt_signals.pkl` after refetching or after changing entry logic. |
| `day_trade_filter_search.py` | Entry-filter search chosen on the first half of sessions and judged on the second half (out-of-sample), to avoid curve fitting. |

Run from the repo root, e.g. `.venv/bin/python research/day_trade_backtest.py`.

Findings (Nov 2026): ATR exits with a breakeven step beat the old fixed-% exits for
swing (PF 1.26 -> 1.41 on equal signals); the day-trade engine showed no robust
edge (PF ~1.0 over 224 trades, out-of-sample filter search ~ coin flip).
