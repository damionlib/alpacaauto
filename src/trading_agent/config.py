from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr


class BrokerConfig(BaseModel):
    provider: Literal["alpaca"] = "alpaca"
    mode: Literal["paper", "live"] = "paper"


class AgentConfig(BaseModel):
    cycle_seconds: int = Field(default=300, ge=30)
    run_on_weekends_for_crypto: bool = True
    execute_orders: bool = True
    max_orders_per_cycle: int = Field(default=3, ge=0)


class AuditConfig(BaseModel):
    enabled: bool = True
    database_path: str = "data/trading_agent.sqlite3"


class PositionManagerConfig(BaseModel):
    enabled: bool = True
    stop_loss_pct: float = Field(default=6.0, gt=0, le=100)
    take_profit_pct: float = Field(default=12.0, gt=0, le=500)
    trailing_stop_pct: float = Field(default=8.0, gt=0, le=100)
    max_holding_days: int = Field(default=20, ge=0)
    manage_options: bool = True
    option_stop_loss_pct: float = Field(default=40.0, gt=0, le=100)
    option_take_profit_pct: float = Field(default=80.0, gt=0, le=1000)


class RiskConfig(BaseModel):
    max_risk_per_trade_pct: float = Field(default=2.0, gt=0, le=10)
    max_daily_loss_pct: float = Field(default=3.0, gt=0, le=20)
    max_position_pct: float = Field(default=12.0, gt=0, le=100)
    max_crypto_position_pct: float = Field(default=10.0, gt=0, le=100)
    max_options_premium_pct: float = Field(default=2.0, gt=0, le=10)
    min_cash_buffer_pct: float = Field(default=5.0, ge=0, le=50)


class StrategyConfig(BaseModel):
    symbols: list[str] = Field(default_factory=lambda: ["SPY", "QQQ"])
    enabled: list[str] = Field(default_factory=lambda: ["equity_momentum"])
    allow_options: bool = True
    allow_crypto: bool = True
    allow_short: bool = False
    min_signal_score: float = Field(default=70.0, ge=0, le=100)


class ResearchConfig(BaseModel):
    news_headline_limit: int = Field(default=8, ge=0, le=50)
    sec_companyfacts_enabled: bool = True


class Settings(BaseModel):
    broker: BrokerConfig = Field(default_factory=BrokerConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    position_manager: PositionManagerConfig = Field(default_factory=PositionManagerConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    alpaca_api_key_id: SecretStr | None = None
    alpaca_api_secret_key: SecretStr | None = None
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
        "allow_live_trading": os.getenv("ALLOW_LIVE_TRADING", "false").lower()
        in {"1", "true", "yes", "on"},
        "sec_user_agent": os.getenv("SEC_USER_AGENT", data.get("sec_user_agent")),
    }
    clean_env_data = {key: value for key, value in env_data.items() if value is not None}
    return Settings.model_validate({**data, **clean_env_data})
