from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class AssetClass(StrEnum):
    EQUITY = "equity"
    ETF = "etf"
    OPTION = "option"
    CRYPTO = "crypto"


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"


class TimeInForce(StrEnum):
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"


class Position(BaseModel):
    symbol: str
    asset_class: AssetClass
    qty: float
    market_value: float
    avg_entry_price: float | None = None
    current_price: float | None = None
    unrealized_pl: float | None = None


class AccountSnapshot(BaseModel):
    equity: float
    cash: float
    buying_power: float
    last_equity: float | None = None

    @property
    def daily_pl_pct(self) -> float:
        if not self.last_equity:
            return 0.0
        return ((self.equity - self.last_equity) / self.last_equity) * 100


class MarketSnapshot(BaseModel):
    symbol: str
    asset_class: AssetClass
    price: float
    closes: list[float] = Field(default_factory=list)
    as_of: datetime = Field(default_factory=lambda: datetime.now(UTC))


class NewsItem(BaseModel):
    title: str
    url: str | None = None
    published: str | None = None


class ResearchSnapshot(BaseModel):
    symbol: str
    news: list[NewsItem] = Field(default_factory=list)
    sec_summary: dict[str, Any] = Field(default_factory=dict)
    crypto_summary: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class TradeCandidate(BaseModel):
    symbol: str
    asset_class: AssetClass
    side: OrderSide
    strategy: str
    score: float
    entry_price: float
    stop_price: float | None = None
    take_profit_price: float | None = None
    rationale: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class OrderIntent(BaseModel):
    symbol: str
    asset_class: AssetClass
    side: OrderSide
    qty: float | None = None
    notional: float | None = None
    order_type: OrderType = OrderType.MARKET
    time_in_force: TimeInForce = TimeInForce.DAY
    limit_price: float | None = None
    order_class: str | None = None
    legs: list[dict[str, str]] = Field(default_factory=list)
    stop_loss_price: float | None = None
    take_profit_price: float | None = None
    client_order_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RiskDecision(BaseModel):
    approved: bool
    reason: str
    intent: OrderIntent | None = None
    candidate: TradeCandidate
