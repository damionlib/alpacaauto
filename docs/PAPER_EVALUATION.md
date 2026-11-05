# Paper evaluation policy (Nov 5 - Nov 13, 2026)

Unattended evaluation of the swing and day-trading books after the November 2026
fixes. The agent loop is started each trading day by launchd
(`scripts/run_session.sh`); an after-close review follows this policy.

## What a few days of paper trading can and cannot show

Seven sessions produce perhaps 10-20 day trades and a handful of swing exits.
That is far too few to judge edge or tune parameters: the backtests needed
hundreds of trades, and the parameters tuned on 3-15 trades in the past
(SmartConfig runs, the partial scale-out) were reversed by them. Live paper days
answer a different question: **does the system execute what the backtest
assumed?** - fills and slippage, stops resting and ratcheting at the broker,
entries in the right window, exits for the expected reasons, no errors.

Backtest expectations to compare against:

| Book | Expectation (research/README.md) |
|---|---|
| Swing | ~1 entry per qualifying day, holds ~11 trading days, win rate ~40%, avg win ~+6%, avg loss ~-2.8%, profit factor ~1.4. Most exits via breakeven/chandelier stops. Slippage assumed 5 bp per side. |
| Day | Up to 3 trades/day, almost all entered 09:45-10:30 ET, win rate ~42%, profit factor ~1.0 (no proven edge). Exits: floor stop, ceiling, or 15:45 force-close. Slippage assumed 2 bp per side. |

## Daily review (after the close)

1. `trading-agent eval-report --since 2026-11-05` (writes `reports/paper-eval/<date>.md` and `.json`).
2. Read the report, `logs/session-<date>.log`, and audit errors.
3. Check: safety invariants; failed cycles; rejected/unprotected orders; swing
   stops placed after each fill and raised as prices rose; entries only in the
   pre-close window (swing) or before the 90-minute cutoff (day); no symbol
   entered by both books on the same day; fill prices vs
   signal prices (slippage); exit-reason mix vs the table above.
4. Append an entry to `reports/paper-eval/JOURNAL.md`.

## What the review may change on its own

- **Bugs and safety problems, immediately**: crashes, failed cycles, broken data,
  order rejections caused by code, any invariant violation (a swing position
  without a full GTC stop, a day trade held past the close). Every fix needs a
  test, and the full suite (`.venv/bin/python -m pytest -q`) must pass.
- **At most one parameter change per day**, and only when re-running the
  relevant backtest in `research/` (with data refreshed through the latest
  session) shows the improvement in **both halves** of the sample, or out of
  sample. Record the before/after numbers in the journal.

## What it must not do

- Increase risk: `risk.max_risk_per_trade_pct`, any `max_position_pct`, daily
  loss limits, `max_drawdown_halt_pct`, `max_trades_per_day`, order caps. Propose
  these to the user instead.
- Tune anything from live P/L alone.
- Switch away from paper, edit `.env` or credentials, disable stops, halts or the
  regime gate, delete audit data, or commit/push to git.
- Run a second agent loop while the launchd runner is active.

## Changing the schedule

- Session runner: `~/Library/LaunchAgents/com.local.trading-agent.session.plist`
  (remove: `launchctl bootout gui/$(id -u)/com.local.trading-agent.session`).
- Daily review: the "Paper trading daily review" scheduled task in the Claude app.
