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
- market snapshots captured by the agent
- research results from news and SEC company data
- raw audit history with filters for review

## Position Manager

Before looking for new entries, the agent checks existing positions and can create exit trades. Exits are submitted before new entries and do not count against `max_orders_per_cycle`.

Configured in `config/settings.toml`:

```toml
[position_manager]
enabled = true
stop_loss_pct = 6.0
take_profit_pct = 12.0
trailing_stop_pct = 8.0
max_holding_days = 20
manage_options = true
option_stop_loss_pct = 40.0
option_take_profit_pct = 80.0
```

The position manager checks:

- unrealized P/L against stop-loss and take-profit thresholds
- trailing drawdown from the tracked peak price
- how long the position has been tracked
- option-specific stop-loss/take-profit thresholds

Exit candidates still pass through the risk engine before execution and are saved in the audit dashboard.

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
```

`audit-backup` creates a SQLite backup copy. `audit-export` writes a JSON review file.

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
