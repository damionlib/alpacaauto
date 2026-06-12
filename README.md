# Trading Agent

This project is a broker-API trading agent scaffold for research, risk checks, and automated execution. It starts with Alpaca because Alpaca currently supports API trading for U.S. stocks/ETFs, options, crypto, and paper trading.

The agent is intentionally paper-first. Live trading requires all of these to be true:

- `config/settings.toml` has `broker.mode = "live"`
- `.env` has `ALLOW_LIVE_TRADING=true`
- valid live API keys are present

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Add Alpaca API keys to `.env`. Start with paper keys.

## Commands

```bash
trading-agent account
trading-agent positions
trading-agent open-orders
trading-agent cancel-open-orders
trading-agent run-once
trading-agent loop
trading-agent dashboard
trading-agent audit-status
trading-agent audit-backup
trading-agent audit-export
```

`run-once` gathers account state, researches configured symbols, generates candidate trades, applies risk gates, and submits allowed orders when `agent.execute_orders = true`.

## Dashboard And Audit Trail

Every agent cycle can be saved to a local SQLite audit database at `data/trading_agent.sqlite3`. The dashboard reads from that database and refreshes automatically.

The audit database is append-only during normal runs. Stopping or rerunning the agent does not delete or overwrite previous cycles.

```bash
trading-agent dashboard
```

Open:

```text
http://127.0.0.1:8080
```

The dashboard shows:

- latest account equity, cash, buying power, and cycle status
- approved/rejected trade decisions with score and reason
- submitted/rejected orders and broker response details
- day-trading signals, decisions, and orders
- market snapshots captured by the agent
- research results from news and SEC company data
- raw audit history with filters for review
- date filtering so each trading day can be reviewed separately

## Position Manager

Before looking for new entries, the agent checks existing positions and can create exit trades. Exits are submitted before new entries and do not count against `max_orders_per_cycle`.

Configured in `config/settings.toml`:

```toml
[position_manager]
enabled = true
stop_loss_pct = 6.0
take_profit_pct = 8.0
trailing_stop_pct = 8.0
max_holding_days = 20
profit_lock_enabled = true
profit_lock_steps = [
  { profit_pct = 3.0, lock_pct = 1.0 },
  { profit_pct = 5.0, lock_pct = 3.0 },
  { profit_pct = 7.0, lock_pct = 5.0 }
]
manage_options = true
option_stop_loss_pct = 40.0
option_take_profit_pct = 50.0
```

The position manager checks:

- unrealized P/L against stop-loss and take-profit thresholds
- profit-lock tiers that protect part of a stock/ETF winner after it reaches configured profit levels
- trailing drawdown from the tracked peak price
- how long the position has been tracked
- option-specific stop-loss/take-profit thresholds

Exit candidates still pass through the risk engine before execution and are saved in the audit dashboard.

## Day Trading

Day trading is an optional paper-first mode for stock/ETF trades that are intended to be opened and closed the same day.

**Intraday data.** The day-trade engine scores setups from intraday inputs — VWAP,
relative volume, short-term trend, opening-range break, and bid/ask spread. When
`day_trading.enabled = true`, the agent fetches a latest quote plus today's 1-minute
bars and attaches these to the snapshot **only for the handful of symbols that reach
day-trade evaluation** (so it stays cheap). Without this data the engine runs blind
on daily candles and its strict gates never clear — do not "fix" that by lowering the
score thresholds; feed it real intraday data instead.

**Regime gate.** Day-trade entries run on their own intraday signals, so the broad
`[regime]` daily-SMA gate does **not** block them by default (`regime.apply_to_day_trades
= false`). Set it true to also require the daily uptrend for day trades.

**Exit ownership.** Day-trade positions are managed only by the day-trade engine; the
swing position manager skips any symbol that has a day-trade entry today, so a single
position is never exited by two engines with conflicting rules.

**Independent daily-loss halt.** Day trading is gated on the day-trade book's *own*
daily P/L (open day-trade positions + today's realized day-trade exits), not the
shared account P/L. A swing drawdown that trips the swing daily-loss stop or the
drawdown circuit breaker halts swing entries but leaves day trading running (and vice
versa), so the two books can be evaluated independently even when run together.

Configured in `config/settings.toml`:

```toml
[day_trading]
enabled = false
paper_only = true
max_trades_per_day = 3
risk_per_trade_pct = 0.25
max_position_pct = 4.0
max_daily_loss_pct = 1.0
max_position_minutes = 120
force_exit_before_close_minutes = 15
min_catalyst_score = 65.0
min_intraday_score = 70.0
min_combined_score = 80.0
exit_intraday_score = 45.0
stop_loss_pct = 1.0
take_profit_pct = 2.0
trailing_stop_pct = 1.0
min_relative_volume = 1.2
max_spread_pct = 0.25
```

The day-trading engine uses the existing research and Catalyst Engine output, then adds intraday and execution checks:

- Catalyst score and direction
- research/news quality and negative-news risk
- intraday trend, VWAP, relative volume, opening-range flags, and short-term trend metadata when available
- spread/execution quality
- tighter day-trading risk sizing and daily-loss limits

Day-trade entries still pass through the normal risk engine and broker execution path. Day-trade exits can be generated when:

- the day-trade stop loss is hit
- the take-profit threshold is hit
- Catalyst flips bearish
- intraday trend score drops below the exit threshold
- max holding time is reached
- the market is approaching close, so the position should not be held overnight

### Swing/Day-Trade Conflict Protection

Alpaca tracks one net position per symbol. If the account already holds 10 AAPL swing shares and the agent buys 10 more AAPL for day trading, the broker can show one 20-share AAPL position with an averaged cost basis. To avoid confusing exits, P/L, and audit records, the agent prevents same-symbol overlap:

- if an existing position exists, the day-trade entry for that symbol is blocked
- if swing and day-trade candidates appear for the same symbol in the same cycle, the swing candidate wins and the day trade is blocked
- if a day-trade entry for a symbol has already been submitted today, later swing entries for that same symbol are blocked

This policy is currently fixed and conservative; there is no `day_trade_conflict_policy` config variable.

### Dashboard

The dashboard includes a **Day Trading** tab with:

- day-trade signals and setup scores
- day-trade risk decisions
- submitted/skipped/rejected day-trade orders
- reasons for entry, waiting, blocking, or exit

Use the dashboard date filter to review exactly what happened on a specific trading day across Day Trading, Trade Decisions, Orders, Catalyst, Market, Research, and Audit History.

## Crypto Research

Crypto symbols such as `BTC/USD` and `ETH/USD` use a separate crypto research path. When enabled, the agent records:

- crypto market regime and score
- BTC and ETH dominance
- total crypto market cap and volume
- asset 24h/7d/30d performance
- crypto fear/greed reading
- stablecoin circulating supply snapshot
- perpetual funding-rate snapshot for supported BTC/ETH pairs
- explicit status for exchange-flow and on-chain feeds

Configured in `config/settings.toml`:

```toml
[research]
crypto_research_enabled = true
crypto_onchain_enabled = false
crypto_onchain_provider = ""
crypto_exchange_flows_enabled = false
```

Exchange-flow and on-chain data are intentionally marked as not configured until a supported provider is added. The crypto momentum strategy uses the crypto regime and risk flags to adjust scores before creating entry candidates.

## Market Screener

By default, the agent uses the fixed `strategy.symbols` list. When the screener is enabled, it scans configured universes, filters for liquidity/trend/volatility, and passes only top candidates into the research and risk pipeline.

```toml
[screener]
enabled = false
max_candidates = 10
max_crypto_candidates = 3
universes = ["nasdaq100", "sp500_core", "crypto_major"]
min_price = 5.0
min_avg_dollar_volume = 25000000.0
max_realized_volatility_pct = 90.0
min_trend_score = 45.0
```

Use the screener for normal paper observation once you are comfortable with the system. Use the fixed `strategy.symbols` list when debugging, testing a small watchlist, or running your first live trials.

Preview screener output without placing trades:

```bash
trading-agent screen
```

Useful audit commands:

```bash
trading-agent audit-status
trading-agent audit-backup
trading-agent audit-export
trading-agent broker-sync --days 30
```

`audit-backup` creates a SQLite backup copy. `audit-export` writes a JSON review file. `broker-sync` is read-only against Alpaca and backfills closed broker orders into SQLite so the performance report can calculate realized P/L from fills.

Start the local dashboard:

```bash
trading-agent dashboard
```

The dashboard includes trade decisions, day trading, orders, market snapshots, research, audit history, and a performance report with equity drawdown, win rate, open P/L, exit-signal P/L, and strategy-level assessment.

## Continuous Run On macOS

For a terminal session:

```bash
scripts/run_agent.sh
```

For `launchd`, copy `scripts/com.local.trading-agent.plist.example` into `~/Library/LaunchAgents/com.local.trading-agent.plist`, create the `logs/` folder, then load it:

```bash
mkdir -p logs
launchctl load ~/Library/LaunchAgents/com.local.trading-agent.plist
```

Use `trading-agent cancel-open-orders` as a manual kill switch for open orders.

## Live Trading Checklist

1. Run paper mode first.
2. Confirm account, positions, and `run-once` output.
3. Edit `config/settings.toml` only after paper testing:

```toml
[broker]
mode = "live"
```

4. Set `ALLOW_LIVE_TRADING=true` in `.env`.
5. Use live Alpaca keys, not paper keys.

## Risk Defaults

- Max risk per trade: 2% of equity
- Max daily loss stop: 3% from prior equity
- Max stock/ETF position: 12% of equity
- Max crypto position: 10% of equity
- Max options premium per trade: 2% of equity
- Max option entry orders per underlying per day: 1
- Option loss cooldown: 1440 minutes before another option entry on the same underlying
- Max entry slippage: 0.5% (entries are submitted as marketable limit orders, not pure market orders, so a gap or thin quote cannot fill far from the price the sizing and stop math assumed)

Per-trade caps are also enforced in aggregate across a single cycle: a shared cash
budget (spendable balance minus the cash buffer) is decremented as each order is
submitted, so the agent cannot approve several entries that each assume the full
buffer.

Option entries are also guarded by underlying, not only by exact contract symbol:
an open or pending option order for `CDNS` blocks another `CDNS` option entry, and
a losing option exit signal starts the same-underlying cooldown. This avoids
repeatedly opening new spreads on the same name after the first idea has already
failed.

### Daily Loss Stop Behavior

When the account is down more than `max_daily_loss_pct` from prior-day equity, the
agent **halts new entries but keeps managing exits**. It no longer cancels all open
orders on the stop: a blanket cancel would also remove the protective stop/take-profit
legs of bracket orders, leaving open positions unguarded on the worst day. The
position manager still runs and can trim or close losers.

`trading-agent cancel-open-orders` remains as a manual kill switch if you want to
clear all working orders yourself.

### Bleed-Stop Controls

Several layers exist to stop a losing streak from compounding:

- **Exits are never throttled.** Closing a position bypasses the daily order caps,
  the cash budget, and the open-order checks. A risk-reducing close can always go
  through, even when the day's order budget is spent on entries.
- **Drawdown circuit breaker** (`risk.max_drawdown_halt_pct`, default 8%). When
  equity falls this far below its trailing peak (over `drawdown_lookback_days`),
  the agent halts *all* new entries until it recovers — stopping it from averaging
  into a sustained drawdown. Distinct from the intraday daily-loss stop.
- **Broad-market regime gate** (`[regime]`). New equity/ETF/option **longs** are
  only opened when the benchmark (`SPY`) is above its `sma_period`-day SMA. In a
  downtrend the agent manages exits and crypto only. Set `enabled = false` to turn
  it off.
- **Correlated-exposure cap** (`risk.max_correlated_exposure_pct`, default 25%).
  Aggregate exposure to one correlated cluster (default: megacap AI/semis) is
  capped, so the book can't load names that all fall together. Option positions
  count toward their underlying's cluster.
- **Pullback entries.** Equity momentum now buys pullbacks *within* an uptrend
  (`SMA20 > SMA50`, price above `SMA50`, near the 20-SMA) instead of chasing
  breakouts at the highs, and requires an uptrend to enter at all.
- **Profit-lock ratchet** (`position_manager.profit_lock_steps`). Once a long
  reaches a profit tier, it exits if the gain gives back to that tier's lock level.
  `lock_pct` must be `< profit_pct` (validated at config load).
- **Options trading window** (`[execution]`). Options are illiquid and badly
  quoted after hours and in the opening auction, which causes stale marks, bad
  fills, and stop whipsaws. Option entries *and* exits are only priced/submitted
  when the market is open and outside the `open_buffer_minutes` /
  `close_buffer_minutes` windows; an option quote wider than `max_option_spread_pct`
  is rejected as untrustworthy. If the market clock can't be fetched, options are
  blocked (fail-closed). Equities and crypto are unaffected.

### Crypto Protective Stops

Alpaca does not support bracket orders for crypto, so after a crypto entry fills the
agent rests a GTC stop-limit sell as a broker-side floor between position-manager
polls. If the broker rejects it, the agent falls back to the position manager's
software stop (which runs every cycle). Verify this in paper before relying on it.

## Cycle And Daily Order Caps

The agent has separate paper/live loop intervals and daily order caps:

```toml
[agent]
paper_cycle_seconds = 300
live_cycle_seconds = 1800
max_orders_per_cycle = 3
paper_max_entry_orders_per_day = 12
paper_max_total_orders_per_day = 24
live_max_entry_orders_per_day = 3
live_max_total_orders_per_day = 6
```

`max_orders_per_cycle` limits new entries per cycle. The daily caps limit submitted orders for the whole Central-time day. Exit orders count toward the total daily cap but not the daily entry cap.

This software is not financial advice. It can lose money, especially if live trading is enabled.

## License

Released under the [MIT License](LICENSE). Copyright (c) 2026 damionlib.
