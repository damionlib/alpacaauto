from __future__ import annotations

from trading_agent.config import Settings
from trading_agent.indicators import pct_change, realized_volatility, sma
from trading_agent.models import (
    AssetClass,
    MarketSnapshot,
    OrderSide,
    ResearchSnapshot,
    TradeCandidate,
)


NEGATIVE_WORDS = {
    "downgrade",
    "fraud",
    "investigation",
    "lawsuit",
    "misses",
    "plunge",
    "recall",
    "sec charges",
    "warning",
}


class MomentumStrategy:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def evaluate(
        self,
        market: MarketSnapshot,
        research: ResearchSnapshot,
    ) -> list[TradeCandidate]:
        enabled = set(self.settings.strategy.enabled)
        if market.asset_class == AssetClass.CRYPTO and "crypto_momentum" not in enabled:
            return []
        if market.asset_class in {AssetClass.EQUITY, AssetClass.ETF} and "equity_momentum" not in enabled:
            return []

        closes = market.closes
        sma20 = sma(closes, 20)
        sma50 = sma(closes, 50)
        change20 = pct_change(closes, 20)
        vol20 = realized_volatility(closes, 20)
        if sma20 is None or sma50 is None or change20 is None:
            return []

        headline_text = " ".join(item.title.lower() for item in research.news)
        negative_penalty = 20 if any(word in headline_text for word in NEGATIVE_WORDS) else 0
        trend_score = 40 if market.price > sma20 > sma50 else 0
        momentum_score = min(max(change20, 0) * 2, 30)
        quality_score = self._quality_score(market, research)
        vol_penalty = min((vol20 or 0) / 4, 20)
        crypto_adjustment = self._crypto_adjustment(market, research)
        score = max(
            0.0,
            min(
                100.0,
                30 + trend_score + momentum_score + quality_score + crypto_adjustment - vol_penalty - negative_penalty,
            ),
        )

        if score < self.settings.strategy.min_signal_score:
            return []

        stop_distance_pct = max(0.04, min(0.15, ((vol20 or 30) / 100) * 0.75))
        stop_price = market.price * (1 - stop_distance_pct)
        take_profit = market.price * (1 + stop_distance_pct * 2)

        return [
            TradeCandidate(
                symbol=market.symbol,
                asset_class=market.asset_class,
                side=OrderSide.BUY,
                strategy=f"{market.asset_class.value}_momentum",
                score=round(score, 2),
                entry_price=market.price,
                stop_price=round(stop_price, 2),
                take_profit_price=round(take_profit, 2),
                rationale=[
                    f"Price above trend filters: {market.price:.2f} vs SMA20 {sma20:.2f} and SMA50 {sma50:.2f}.",
                    f"20-day change is {change20:.2f}%; annualized realized volatility is {(vol20 or 0):.2f}%.",
                    f"Negative headline penalty applied: {negative_penalty > 0}.",
                    f"Crypto regime adjustment: {crypto_adjustment:.2f}." if market.asset_class == AssetClass.CRYPTO else "SEC quality check included.",
                ],
                metadata={
                    "sma20": sma20,
                    "sma50": sma50,
                    "change20": change20,
                    "vol20": vol20,
                    "crypto_regime": research.crypto_summary.get("regime"),
                    "crypto_risk_flags": research.crypto_summary.get("risk_flags", []),
                },
            )
        ]

    def _quality_score(self, market: MarketSnapshot, research: ResearchSnapshot) -> float:
        if market.asset_class == AssetClass.CRYPTO:
            regime = research.crypto_summary.get("regime") or {}
            label = regime.get("label")
            if label == "risk_on":
                return 8
            if label == "risk_off":
                return -12
            return 0
        return 10 if research.sec_summary.get("latest_net_income") else 0

    def _crypto_adjustment(self, market: MarketSnapshot, research: ResearchSnapshot) -> float:
        if market.asset_class != AssetClass.CRYPTO:
            return 0
        summary = research.crypto_summary
        regime = summary.get("regime") or {}
        score = float(regime.get("score") or 50)
        adjustment = (score - 50) / 2
        risk_flags = set(summary.get("risk_flags", []))
        if "extreme_greed" in risk_flags:
            adjustment -= 8
        if "perp_funding_crowded_long" in risk_flags:
            adjustment -= 6
        if "crypto_market_risk_off" in risk_flags:
            adjustment -= 10
        return max(-25.0, min(15.0, adjustment))
