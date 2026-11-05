from __future__ import annotations

from trading_agent.config import Settings
from trading_agent.indicators import atr, pct_change, realized_volatility, sma
from trading_agent.models import (
    AssetClass,
    MarketSnapshot,
    OrderSide,
    ResearchSnapshot,
    TradeCandidate,
)
from trading_agent.text_terms import count_terms


NEGATIVE_WORDS = {
    "downgrade",
    "fraud",
    "investigation",
    "lawsuit",
    "misses",
    "missed",
    "plunge",
    "plunges",
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
        negative_penalty = 20 if count_terms(headline_text, NEGATIVE_WORDS) else 0
        quality_score = self._quality_score(market, research)
        vol_penalty = min((vol20 or 0) / 4, 20)
        crypto_adjustment = self._crypto_adjustment(market, research)

        if market.asset_class == AssetClass.CRYPTO:
            trend_score = 40 if market.price > sma20 > sma50 else 0
            pullback_score = 0.0
            momentum_score = min(max(change20, 0) * 2, 30)
            extension_pct = 0.0
        else:
            # Buy pullbacks within an uptrend, NOT breakouts at the highs. An
            # uptrend requires SMA20 > SMA50 and price above SMA50; entries are
            # rewarded near/below the 20-SMA and penalized when price is stretched
            # far above it (chasing the top — the prior behaviour that bled).
            uptrend = sma20 > sma50 and market.price > sma50
            extension_pct = ((market.price - sma20) / sma20 * 100) if sma20 else 0.0
            trend_score = 35 if uptrend else 0
            if not uptrend:
                pullback_score = 0.0
            elif extension_pct <= 1.5:
                pullback_score = 18.0
            elif extension_pct <= 4.0:
                pullback_score = 8.0
            elif extension_pct <= 5.0:
                pullback_score = -5.0
            elif extension_pct <= 8.0:
                pullback_score = -15.0
            else:
                pullback_score = -20.0
            momentum_score = min(max(change20, 0) * 1.5, 22)

        score = max(
            0.0,
            min(
                100.0,
                30 + trend_score + pullback_score + momentum_score + quality_score
                + crypto_adjustment - vol_penalty - negative_penalty,
            ),
        )

        if score < self.settings.strategy.min_signal_score:
            return []

        stop_price, take_profit, atr_value, stop_basis = self._stops(market, vol20)
        extra_metadata = {}
        if atr_value is not None:
            extra_metadata = {
                "atr": round(atr_value, 4),
                "atr_period": self.settings.strategy.atr_period,
                "initial_stop": round(stop_price, 2),
            }

        return [
            TradeCandidate(
                symbol=market.symbol,
                asset_class=market.asset_class,
                side=OrderSide.BUY,
                strategy=f"{market.asset_class.value}_momentum",
                score=round(score, 2),
                entry_price=market.price,
                stop_price=round(stop_price, 2),
                take_profit_price=round(take_profit, 2) if take_profit is not None else None,
                rationale=[
                    f"Trend/pullback: price {market.price:.2f} vs SMA20 {sma20:.2f}, SMA50 {sma50:.2f} "
                    f"({extension_pct:.2f}% above SMA20).",
                    f"20-day change is {change20:.2f}%; annualized realized volatility is {(vol20 or 0):.2f}%.",
                    f"Negative headline penalty applied: {negative_penalty > 0}.",
                    stop_basis,
                    f"Crypto regime adjustment: {crypto_adjustment:.2f}." if market.asset_class == AssetClass.CRYPTO else "SEC quality check included.",
                ],
                metadata={
                    "sma20": sma20,
                    "sma50": sma50,
                    "change20": change20,
                    "vol20": vol20,
                    "extension_pct": extension_pct,
                    "crypto_regime": research.crypto_summary.get("regime"),
                    "crypto_risk_flags": research.crypto_summary.get("risk_flags", []),
                    **extra_metadata,
                },
            )
        ]

    def _stops(
        self, market: MarketSnapshot, vol20: float | None,
    ) -> tuple[float, float | None, float | None, str]:
        """Initial stop (and legacy target) for a long entry.

        Equities with high/low history get an ATR stop and no fixed target: the
        exit is a ratcheting ATR/chandelier stop managed by the position manager
        and held at the broker. The old annualized-vol formula put nearly every
        stock at its 15% clamp, so it is only a fallback when bars lack highs/lows.
        """
        cfg = self.settings.strategy
        metadata = market.metadata or {}
        highs = metadata.get("highs") or []
        lows = metadata.get("lows") or []
        atr_value = None
        if market.asset_class != AssetClass.CRYPTO and highs and lows:
            atr_value = atr(highs, lows, market.closes, cfg.atr_period)
        if atr_value and market.price > 0:
            raw_pct = cfg.atr_stop_multiple * atr_value / market.price * 100
            stop_pct = max(cfg.min_stop_pct, min(cfg.max_stop_pct, raw_pct))
            stop_price = market.price * (1 - stop_pct / 100)
            return (
                stop_price,
                None,
                atr_value,
                f"Initial stop {stop_pct:.2f}% below entry = {cfg.atr_stop_multiple:g} x ATR{cfg.atr_period} "
                f"({atr_value:.2f}); no fixed target, exit trails with the ATR stop.",
            )
        stop_distance_pct = max(0.04, min(0.15, ((vol20 or 30) / 100) * 0.75))
        return (
            market.price * (1 - stop_distance_pct),
            market.price * (1 + stop_distance_pct * 2),
            None,
            f"Volatility-based stop {stop_distance_pct * 100:.2f}% (no high/low history for ATR).",
        )

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
