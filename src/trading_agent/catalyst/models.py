from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


CatalystDirection = Literal["bullish", "bearish", "neutral"]
CatalystConfidence = Literal["low", "medium", "high"]


class CatalystPrediction(BaseModel):
    symbol: str
    asset_class: str
    direction: CatalystDirection
    horizon: str = "1-5 trading days"
    prediction_score: float = Field(ge=0, le=100)
    bullish_score: float = Field(ge=0, le=100)
    confidence: CatalystConfidence
    entry_allowed: bool
    block_reason: str | None = None
    market_regime: str
    components: dict[str, float] = Field(default_factory=dict)
    evidence: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
