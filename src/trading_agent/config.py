from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr, model_validator


class BrokerConfig(BaseModel):
    provider: Literal["alpaca"] = "alpaca"
    mode: Literal["paper", "live"] = "paper"


class AgentConfig(BaseModel):
    cycle_seconds: int | None = Field(default=None, ge=30)
    paper_cycle_seconds: int = Field(default=300, ge=30)
    live_cycle_seconds: int = Field(default=1800, ge=30)
    run_on_weekends_for_crypto: bool = True
    execute_orders: bool = True
    max_orders_per_cycle: int = Field(default=3, ge=0)
    paper_max_entry_orders_per_day: int = Field(default=12, ge=0)
    paper_max_total_orders_per_day: int = Field(default=24, ge=0)
    live_max_entry_orders_per_day: int = Field(default=3, ge=0)
    live_max_total_orders_per_day: int = Field(default=6, ge=0)

    def cycle_interval(self, *, live: bool) -> int:
        if self.cycle_seconds is not None:
            return self.cycle_seconds
        return self.live_cycle_seconds if live else self.paper_cycle_seconds

    def max_entry_orders_per_day(self, *, live: bool) -> int:
        return self.live_max_entry_orders_per_day if live else self.paper_max_entry_orders_per_day

    def max_total_orders_per_day(self, *, live: bool) -> int:
        return self.live_max_total_orders_per_day if live else self.paper_max_total_orders_per_day


class AuditConfig(BaseModel):
    enabled: bool = True
    database_path: str = "data/trading_agent.sqlite3"


class ProfitLockStep(BaseModel):
    profit_pct: float = Field(gt=0, le=500)
    lock_pct: float = Field(ge=0, le=500)


class PositionManagerConfig(BaseModel):
    enabled: bool = True
    stop_loss_pct: float = Field(default=6.0, gt=0, le=100)
    take_profit_pct: float = Field(default=12.0, gt=0, le=500)
    trailing_stop_pct: float = Field(default=8.0, gt=0, le=100)
    max_holding_days: int = Field(default=20, ge=0)
    max_days_without_profit: int = Field(default=7, ge=0)
    profit_lock_enabled: bool = True
    profit_lock_steps: list[ProfitLockStep] = Field(
        default_factory=lambda: [
            ProfitLockStep(profit_pct=5.0, lock_pct=2.0),
            ProfitLockStep(profit_pct=8.0, lock_pct=5.0),
            ProfitLockStep(profit_pct=10.0, lock_pct=8.0),
        ]
    )
    manage_options: bool = True
    option_stop_loss_pct: float = Field(default=40.0, gt=0, le=100)
    option_take_profit_pct: float = Field(default=80.0, gt=0, le=1000)

    @model_validator(mode="after")
    def _validate_profit_lock(self) -> "PositionManagerConfig":
        for step in self.profit_lock_steps:
            if step.lock_pct >= step.profit_pct:
                raise ValueError(
                    f"profit_lock step lock_pct ({step.lock_pct}) must be < profit_pct "
                    f"({step.profit_pct}); otherwise the position exits the instant it reaches the tier."
                )
        # Keep tiers ordered so the highest reached tier is well-defined.
        self.profit_lock_steps = sorted(self.profit_lock_steps, key=lambda item: item.profit_pct)
        return self


class RiskConfig(BaseModel):
    max_risk_per_trade_pct: float = Field(default=2.0, gt=0, le=10)
    max_daily_loss_pct: float = Field(default=3.0, gt=0, le=20)
    max_position_pct: float = Field(default=12.0, gt=0, le=100)
    max_crypto_position_pct: float = Field(default=10.0, gt=0, le=100)
    max_options_premium_pct: float = Field(default=2.0, gt=0, le=10)
    min_cash_buffer_pct: float = Field(default=5.0, ge=0, le=50)
    max_entry_slippage_pct: float = Field(default=0.5, ge=0, le=5)
    # Portfolio circuit breaker: halt NEW entries when equity is down this far
    # from its trailing peak (0 disables). Distinct from the intraday daily-loss
    # stop — this stops averaging into a sustained drawdown.
    max_drawdown_halt_pct: float = Field(default=8.0, ge=0, le=100)
    drawdown_lookback_days: int = Field(default=14, ge=1, le=365)
    # Cap aggregate exposure to a correlated cluster (0 disables).
    max_correlated_exposure_pct: float = Field(default=25.0, ge=0, le=100)
    correlated_groups: list[list[str]] = Field(
        default_factory=lambda: [
            [
                "NVDA", "AVGO", "AMD", "MU", "AMAT", "LRCX", "KLAC", "ASML", "MRVL",
                "TSM", "SMCI", "ORCL", "MSFT", "GOOGL", "GOOG", "META", "AAPL", "AMZN", "TSLA",
            ],
            # Cybersecurity / high-beta software: the Jun 9 cluster loss (AAPL+PANW+CRWD
            # gapping down together) was invisible to the cap because only AAPL was grouped.
            [
                "PANW", "CRWD", "ZS", "FTNT", "NET", "OKTA", "S", "DDOG",
                "NOW", "CRM", "ADBE", "SNOW", "MDB", "TEAM", "WDAY",
            ],
        ]
    )


class StrategyConfig(BaseModel):
    symbols: list[str] = Field(default_factory=lambda: ["SPY", "QQQ"])
    enabled: list[str] = Field(default_factory=lambda: ["equity_momentum"])
    allow_options: bool = True
    allow_crypto: bool = True
    allow_short: bool = False
    min_signal_score: float = Field(default=70.0, ge=0, le=100)
    max_option_entry_orders_per_underlying_per_day: int = Field(default=1, ge=0)
    option_loss_cooldown_minutes: int = Field(default=1440, ge=0)


class ScreenerConfig(BaseModel):
    enabled: bool = False
    max_candidates: int = Field(default=10, ge=1, le=100)
    max_crypto_candidates: int = Field(default=3, ge=0, le=25)
    universes: list[str] = Field(default_factory=lambda: ["nasdaq100", "sp500_core", "crypto_major"])
    min_price: float = Field(default=5.0, ge=0)
    min_avg_dollar_volume: float = Field(default=25_000_000.0, ge=0)
    max_realized_volatility_pct: float = Field(default=90.0, gt=0)
    min_trend_score: float = Field(default=45.0, ge=0, le=100)


class CatalystConfig(BaseModel):
    enabled: bool = True
    min_trade_score: float = Field(default=65.0, ge=0, le=100)
    min_entry_score: float = Field(default=80.0, ge=0, le=100)
    score_weight: float = Field(default=0.35, ge=0, le=1)
    max_volatility_pct: float = Field(default=75.0, gt=0, le=200)
    generate_entry_candidates: bool = True
    block_low_confidence: bool = True
    block_rumor_only: bool = True
    block_high_event_risk: bool = True
    require_medium_confidence_for_options: bool = True


class DayTradingConfig(BaseModel):
    enabled: bool = False
    paper_only: bool = True
    max_trades_per_day: int = Field(default=3, ge=0)
    risk_per_trade_pct: float = Field(default=0.25, gt=0, le=5)
    max_position_pct: float = Field(default=4.0, gt=0, le=25)
    max_daily_loss_pct: float = Field(default=1.0, gt=0, le=10)
    max_position_minutes: int = Field(default=120, ge=1)
    force_exit_before_close_minutes: int = Field(default=15, ge=0)
    min_catalyst_score: float = Field(default=65.0, ge=0, le=100)
    min_intraday_score: float = Field(default=70.0, ge=0, le=100)
    min_combined_score: float = Field(default=80.0, ge=0, le=100)
    exit_intraday_score: float = Field(default=45.0, ge=0, le=100)
    stop_loss_pct: float = Field(default=1.0, gt=0, le=10)
    take_profit_pct: float = Field(default=2.0, gt=0, le=20)
    trailing_stop_pct: float = Field(default=1.0, gt=0, le=10)
    ceiling_adr_fraction: float = Field(default=0.55, gt=0, le=2.0)
    floor_adr_fraction: float = Field(default=0.40, gt=0, le=2.0)
    # Partial scale-out: bank a fraction of the position at partial_exit_adr_fraction
    # of ADR (capped at partial_exit_max_pct), then hold the rest to the ceiling with
    # a breakeven stop. partial_exit_size=0 disables.
    partial_exit_adr_fraction: float = Field(default=0.45, ge=0, le=2.0)
    partial_exit_max_pct: float = Field(default=1.0, gt=0, le=10)
    partial_exit_size: float = Field(default=0.5, ge=0, lt=1)
    # Per-setup risk scaling (applies to both risk budget and position cap). Setups
    # with weak realized expectancy trade smaller until they prove themselves.
    setup_risk_multipliers: dict[str, float] = Field(
        default_factory=lambda: {"opening_range_break": 0.5}
    )
    profit_protect_pct: float = Field(default=0.4, ge=0, le=10)
    profit_trail_pct: float = Field(default=0.5, ge=0, le=10)
    stale_negative_minutes: int = Field(default=30, ge=0)
    time_decay_minutes: int = Field(default=60, ge=0)
    min_relative_volume: float = Field(default=1.2, ge=0)
    max_spread_pct: float = Field(default=0.25, gt=0, le=5)


class ResearchConfig(BaseModel):
    news_headline_limit: int = Field(default=8, ge=0, le=50)
    news_cache_enabled: bool = True
    news_cache_database_path: str = "data/news_cache.sqlite3"
    news_cache_ttl_seconds: int = Field(default=1800, ge=0)
    marketaux_daily_call_limit: int = Field(default=90, ge=0)
    alpaca_news_daily_call_limit: int = Field(default=0, ge=0)
    yahoo_news_daily_call_limit: int = Field(default=0, ge=0)
    sec_companyfacts_enabled: bool = True
    crypto_research_enabled: bool = True
    crypto_onchain_enabled: bool = False
    crypto_onchain_provider: str | None = None
    crypto_exchange_flows_enabled: bool = False


class ExecutionConfig(BaseModel):
    # Options are illiquid and badly quoted after hours and in the opening
    # auction, which produces stale marks, bad fills, and stop-loss whipsaws.
    # Restrict option pricing/orders to regular hours with open/close buffers.
    market_hours_only_options: bool = True
    open_buffer_minutes: int = Field(default=15, ge=0, le=120)
    close_buffer_minutes: int = Field(default=10, ge=0, le=120)
    # Reject an option candidate whose quoted bid/ask is wider than this (% of
    # mid) — a wide quote is untrustworthy to size or fill against.
    max_option_spread_pct: float = Field(default=25.0, gt=0, le=200)


class RegimeConfig(BaseModel):
    # Broad-market trend gate: only open new equity/ETF/option longs when the
    # benchmark is above its trend SMA. Stops buying longs into a falling tape.
    enabled: bool = True
    benchmark_symbol: str = "SPY"
    sma_period: int = Field(default=50, ge=5, le=200)
    block_equity_entries_in_downtrend: bool = True
    # Day trading runs on its own intraday-timeframe signals, so by default the
    # daily-SMA gate does not block day-trade entries. Set true to apply it.
    apply_to_day_trades: bool = False


class Settings(BaseModel):
    broker: BrokerConfig = Field(default_factory=BrokerConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    position_manager: PositionManagerConfig = Field(default_factory=PositionManagerConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    screener: ScreenerConfig = Field(default_factory=ScreenerConfig)
    catalyst: CatalystConfig = Field(default_factory=CatalystConfig)
    day_trading: DayTradingConfig = Field(default_factory=DayTradingConfig)
    regime: RegimeConfig = Field(default_factory=RegimeConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    alpaca_api_key_id: SecretStr | None = None
    alpaca_api_secret_key: SecretStr | None = None
    marketaux_api_token: SecretStr | None = None
    allow_live_trading: bool = False
    sec_user_agent: str = "trading-agent your-email@example.com"

    @property
    def is_live(self) -> bool:
        return self.broker.mode == "live"

    def require_live_confirmation(self) -> None:
        if self.is_live and not self.allow_live_trading:
            raise RuntimeError(
                "Live trading requested, but ALLOW_LIVE_TRADING is not true. "
                "Keep paper mode until you have tested the agent."
            )


def load_settings(config_path: str | Path = "config/settings.toml") -> Settings:
    load_dotenv()
    path = Path(config_path)
    data: dict = {}
    if path.exists():
        with path.open("rb") as handle:
            data = tomllib.load(handle)

    env_data = {
        "alpaca_api_key_id": os.getenv("ALPACA_API_KEY_ID"),
        "alpaca_api_secret_key": os.getenv("ALPACA_API_SECRET_KEY"),
        "marketaux_api_token": os.getenv("MARKETAUX_API_TOKEN"),
        "allow_live_trading": os.getenv("ALLOW_LIVE_TRADING", "false").lower()
        in {"1", "true", "yes", "on"},
        "sec_user_agent": os.getenv("SEC_USER_AGENT", data.get("sec_user_agent")),
    }
    clean_env_data = {key: value for key, value in env_data.items() if value is not None}
    return Settings.model_validate({**data, **clean_env_data})
