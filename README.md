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

This software is not financial advice. It can lose money, especially if live trading is enabled.
