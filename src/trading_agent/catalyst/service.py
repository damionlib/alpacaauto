from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from trading_agent.catalyst.models import CatalystPrediction
from trading_agent.config import Settings
from trading_agent.indicators import pct_change, realized_volatility, sma
from trading_agent.models import AssetClass, MarketSnapshot, OrderSide, ResearchSnapshot, TradeCandidate


POSITIVE_TERMS = {
    "accelerates",
    "acquisition",
    "approval",
    "award",
    "awarded",
    "beat",
    "beats",
    "buyback",
    "contract",
    "deal",
    "dividend",
    "earnings beat",
    "guidance raised",
    "launch",
    "partnership",
    "profit",
    "raises",
    "record",
    "secures",
    "upgrade",
}

NEGATIVE_TERMS = {
    "bankruptcy",
    "charges",
    "cut",
    "downgrade",
    "fraud",
    "guidance cut",
    "investigation",
    "lawsuit",
    "miss",
    "misses",
    "probe",
    "recall",
    "sanction",
    "slumps",
    "warning",
}

RUMOR_TERMS = {"could", "reportedly", "rumor", "speculation", "talks", "unconfirmed"}
MACRO_TERMS = {
    "cpi",
    "fed",
    "inflation",
    "jobs",
    "president",
    "rate",
    "tariff",
    "treasury",
}
GEOPOLITICAL_TERMS = {"china", "iran", "israel", "russia", "sanction", "ukraine", "war"}
OFFICIAL_TERMS = {"10-k", "10-q", "8-k", "announces", "company", "filing", "results"}


@dataclass
class _LayerResult:
    score: float
    evidence: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    metadata: dict[str, float | int | str | bool | list[str]] = field(default_factory=dict)


class CatalystEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def evaluate(self, market: MarketSnapshot, research: ResearchSnapshot) -> CatalystPrediction:
        if not self.settings.catalyst.enabled:
            return self._disabled_prediction(market)

        market_layer = self._market_regime(market)
        news_layer = self._news_catalyst(research)
        quality_layer = self._quality_layer(market, research)
        source_layer = self._source_quality(research)
        event_layer = self._event_risk(research, market_layer)

        bullish_score = self._clamp(
            market_layer.score * 0.30
            + news_layer.score * 0.25
            + quality_layer.score * 0.20
            + source_layer.score * 0.15
            + event_layer.score * 0.10
        )
        direction = self._direction(bullish_score)
        prediction_score = round(50 + abs(bullish_score - 50), 2)
        risks = [*market_layer.risks, *news_layer.risks, *quality_layer.risks, *source_layer.risks, *event_layer.risks]
        evidence = [
            *market_layer.evidence,
            *news_layer.evidence,
            *quality_layer.evidence,
            *source_layer.evidence,
            *event_layer.evidence,
        ]
        confidence = self._confidence(source_layer.score, market_layer, news_layer, quality_layer)
        block_reason = self._block_reason(direction, prediction_score, confidence, risks)

        return CatalystPrediction(
            symbol=market.symbol,
            asset_class=market.asset_class.value,
            direction=direction,
            prediction_score=prediction_score,
            bullish_score=round(bullish_score, 2),
            confidence=confidence,
            entry_allowed=block_reason is None,
            block_reason=block_reason,
            market_regime=str(market_layer.metadata.get("regime", "neutral")),
            components={
                "market_regime": round(market_layer.score, 2),
                "news_catalyst": round(news_layer.score, 2),
                "quality": round(quality_layer.score, 2),
                "source_quality": round(source_layer.score, 2),
                "event_risk": round(event_layer.score, 2),
            },
            evidence=evidence[:8],
            risks=risks[:8],
            metadata={
                "market": market_layer.metadata,
                "news": news_layer.metadata,
                "quality": quality_layer.metadata,
                "source": source_layer.metadata,
                "event": event_layer.metadata,
            },
        )

    def apply_to_candidates(
        self,
        candidates: Iterable[TradeCandidate],
        prediction: CatalystPrediction,
    ) -> tuple[list[TradeCandidate], list[tuple[TradeCandidate, str]]]:
        if not self.settings.catalyst.enabled:
            return list(candidates), []

        accepted: list[TradeCandidate] = []
        blocked: list[tuple[TradeCandidate, str]] = []
        for candidate in candidates:
            if candidate.metadata.get("exit"):
                accepted.append(candidate)
                continue
            reason = self._candidate_block_reason(candidate, prediction)
            if reason:
                blocked.append((self._attach_prediction(candidate, prediction), reason))
                continue
            accepted.append(self._boost_candidate(candidate, prediction))
        return accepted, blocked

    def entry_candidate(
        self,
        market: MarketSnapshot,
        prediction: CatalystPrediction,
        existing_candidates: Iterable[TradeCandidate],
    ) -> TradeCandidate | None:
        if not self.settings.catalyst.enabled or not self.settings.catalyst.generate_entry_candidates:
            return None
        if market.asset_class not in {AssetClass.EQUITY, AssetClass.ETF, AssetClass.CRYPTO}:
            return None
        if prediction.direction != "bullish" or prediction.prediction_score < self.settings.catalyst.min_entry_score:
            return None
        if not prediction.entry_allowed:
            return None
        for candidate in existing_candidates:
            if candidate.symbol == market.symbol and candidate.side == OrderSide.BUY:
                return None

        vol20 = float(prediction.metadata.get("market", {}).get("vol20") or 30)
        stop_distance_pct = max(0.05, min(0.14, (vol20 / 100) * 0.7))
        return TradeCandidate(
            symbol=market.symbol,
            asset_class=market.asset_class,
            side=OrderSide.BUY,
            strategy=f"{market.asset_class.value}_catalyst",
            score=round(prediction.prediction_score, 2),
            entry_price=market.price,
            stop_price=round(market.price * (1 - stop_distance_pct), 2),
            take_profit_price=round(market.price * (1 + stop_distance_pct * 2), 2),
            rationale=[
                f"Catalyst Engine predicts {prediction.direction} direction over {prediction.horizon}.",
                *prediction.evidence[:3],
            ],
            metadata={"catalyst": prediction.model_dump(mode="json")},
        )

    def _candidate_block_reason(self, candidate: TradeCandidate, prediction: CatalystPrediction) -> str | None:
        stance = self._strategy_stance(candidate.strategy)
        if prediction.block_reason and stance != "neutral_income":
            return f"Catalyst blocked entry: {prediction.block_reason}"
        if candidate.asset_class == AssetClass.OPTION and self.settings.catalyst.require_medium_confidence_for_options:
            if prediction.confidence == "low":
                return "Catalyst confidence is low; options are blocked."
        if stance == "bullish":
            if prediction.direction != "bullish":
                return f"Catalyst direction is {prediction.direction}, not bullish."
            if prediction.prediction_score < self.settings.catalyst.min_trade_score:
                return f"Catalyst prediction score {prediction.prediction_score:.2f} is below trade threshold."
        elif stance == "bearish":
            if prediction.direction != "bearish":
                return f"Catalyst direction is {prediction.direction}, not bearish."
            if prediction.prediction_score < self.settings.catalyst.min_trade_score:
                return f"Catalyst prediction score {prediction.prediction_score:.2f} is below trade threshold."
        elif stance == "neutral_income":
            if prediction.risks and "rumor_only_catalyst" in prediction.risks:
                return "Catalyst risk is rumor-only; income option entry blocked."
            if prediction.direction == "bullish" and prediction.prediction_score >= self.settings.catalyst.min_entry_score:
                return "Catalyst is strongly bullish; covered-call income would cap upside."
        return None

    def _boost_candidate(self, candidate: TradeCandidate, prediction: CatalystPrediction) -> TradeCandidate:
        weight = self.settings.catalyst.score_weight
        score = candidate.score * (1 - weight) + prediction.prediction_score * weight
        if prediction.confidence == "high":
            score += 2
        if prediction.confidence == "low":
            score -= 5
        score = self._clamp(score)
        return self._attach_prediction(candidate, prediction).model_copy(update={"score": round(score, 2)})

    def _attach_prediction(self, candidate: TradeCandidate, prediction: CatalystPrediction) -> TradeCandidate:
        metadata = dict(candidate.metadata)
        metadata["catalyst"] = prediction.model_dump(mode="json")
        rationale = [
            *candidate.rationale,
            f"Catalyst Engine: {prediction.direction}, score {prediction.prediction_score:.2f}, confidence {prediction.confidence}.",
        ]
        if prediction.block_reason:
            rationale.append(f"Catalyst block reason: {prediction.block_reason}")
        return candidate.model_copy(update={"metadata": metadata, "rationale": rationale})

    def _market_regime(self, market: MarketSnapshot) -> _LayerResult:
        closes = market.closes
        sma20 = sma(closes, 20)
        sma50 = sma(closes, 50)
        change20 = pct_change(closes, 20) or 0.0
        vol20 = realized_volatility(closes, 20) or 0.0
        score = 50.0
        evidence: list[str] = []
        risks: list[str] = []
        regime = "neutral"

        if sma20 is not None and sma50 is not None:
            if market.price > sma20 > sma50:
                score += 24
                regime = "risk_on"
                evidence.append(f"Price is above SMA20 and SMA50 for {market.symbol}.")
            elif market.price < sma20 < sma50:
                score -= 24
                regime = "risk_off"
                risks.append(f"Price is below SMA20 and SMA50 for {market.symbol}.")
        score += self._bounded(change20 * 1.6, -18, 18)
        if change20 > 3:
            evidence.append(f"20-day price change is {change20:.2f}%.")
        elif change20 < -3:
            risks.append(f"20-day price change is {change20:.2f}%.")
        if vol20 > self.settings.catalyst.max_volatility_pct:
            score -= 14
            risks.append(f"Realized volatility is elevated at {vol20:.2f}%.")
        elif 0 < vol20 < 35:
            score += 5
            evidence.append(f"Realized volatility is controlled at {vol20:.2f}%.")

        return _LayerResult(
            score=self._clamp(score),
            evidence=evidence,
            risks=risks,
            metadata={
                "regime": regime,
                "sma20": sma20 or 0.0,
                "sma50": sma50 or 0.0,
                "change20": change20,
                "vol20": vol20,
            },
        )

    def _news_catalyst(self, research: ResearchSnapshot) -> _LayerResult:
        text = self._news_text(research)
        positive = self._term_hits(text, POSITIVE_TERMS)
        negative = self._term_hits(text, NEGATIVE_TERMS)
        rumor = self._term_hits(text, RUMOR_TERMS)
        macro = self._term_hits(text, MACRO_TERMS)
        geo = self._term_hits(text, GEOPOLITICAL_TERMS)
        official = self._term_hits(text, OFFICIAL_TERMS)
        sentiment = self._average_news_sentiment(research)

        score = 50 + min(positive * 8, 28) - min(negative * 12, 45)
        evidence: list[str] = []
        risks: list[str] = []
        if sentiment is not None:
            score += self._bounded(sentiment * 20, -15, 15)
            if sentiment > 0.1:
                evidence.append(f"Entity news sentiment is positive at {sentiment:.2f}.")
            elif sentiment < -0.1:
                risks.append(f"Entity news sentiment is negative at {sentiment:.2f}.")
        if positive:
            evidence.append(f"{positive} positive catalyst keyword(s) found in headlines.")
        if negative:
            risks.append(f"{negative} negative catalyst keyword(s) found in headlines.")
        if rumor:
            risks.append("rumor_only_catalyst")
        if macro or geo:
            risks.append("macro_or_geopolitical_headline_risk")
            score -= min((macro + geo) * 3, 12)
        if official:
            score += min(official * 3, 9)
            evidence.append("Headlines include official/company or filing-style language.")
        if not research.news:
            risks.append("news_unavailable")
            score -= 6

        return _LayerResult(
            score=self._clamp(score),
            evidence=evidence,
            risks=risks,
            metadata={
                "positive_hits": positive,
                "negative_hits": negative,
                "rumor_hits": rumor,
                "macro_hits": macro,
                "geopolitical_hits": geo,
                "official_hits": official,
                "sentiment_avg": sentiment if sentiment is not None else 0.0,
            },
        )

    def _quality_layer(self, market: MarketSnapshot, research: ResearchSnapshot) -> _LayerResult:
        if market.asset_class == AssetClass.CRYPTO:
            return self._crypto_quality(research)

        sec = research.sec_summary or {}
        net_income = sec.get("latest_net_income") or {}
        revenue = sec.get("latest_revenue") or {}
        filings = sec.get("recent_filings") or []
        score = 50.0
        evidence: list[str] = []
        risks: list[str] = []
        if revenue.get("value") is not None:
            score += 8
            evidence.append("Latest revenue fact is available from SEC company facts.")
        else:
            risks.append("revenue_fact_unavailable")
        income_value = net_income.get("value")
        if income_value is not None:
            income_float = self._coerce_float(income_value)
            if income_float is None:
                risks.append("net_income_fact_unparseable")
            elif income_float > 0:
                score += 14
                evidence.append("Latest net income fact is positive.")
            else:
                score -= 22
                risks.append("latest_net_income_not_positive")
        else:
            risks.append("net_income_fact_unavailable")
        if filings:
            forms = {str(row.get("form") or "") for row in filings}
            score += 5
            evidence.append(f"Recent SEC filings available: {', '.join(sorted(forms)[:3])}.")
        return _LayerResult(
            score=self._clamp(score),
            evidence=evidence,
            risks=risks,
            metadata={"has_revenue": bool(revenue), "has_net_income": bool(net_income), "filing_count": len(filings)},
        )

    def _crypto_quality(self, research: ResearchSnapshot) -> _LayerResult:
        summary = research.crypto_summary or {}
        regime = summary.get("regime") or {}
        score = self._coerce_float(regime.get("score")) or 50.0
        risk_flags = list(summary.get("risk_flags") or [])
        evidence: list[str] = []
        risks: list[str] = []
        label = regime.get("label")
        if label:
            evidence.append(f"Crypto market regime is {label} with score {score:.2f}.")
        if risk_flags:
            risks.extend(str(flag) for flag in risk_flags)
        return _LayerResult(
            score=self._clamp(score),
            evidence=evidence,
            risks=risks,
            metadata={"crypto_regime": str(label or ""), "risk_flags": risk_flags},
        )

    def _source_quality(self, research: ResearchSnapshot) -> _LayerResult:
        news_count = len(research.news)
        sec = research.sec_summary or {}
        crypto = research.crypto_summary or {}
        score = 35.0
        evidence: list[str] = []
        risks: list[str] = []
        if news_count >= 3:
            score += 18
            evidence.append(f"{news_count} headline(s) available for catalyst analysis.")
        elif news_count:
            score += 8
        else:
            risks.append("no_headlines_for_catalyst")
        if sec.get("latest_revenue") or sec.get("latest_net_income"):
            score += 25
            evidence.append("SEC company-fact data is available.")
        if crypto.get("regime"):
            score += 22
            evidence.append("Crypto-specific regime data is available.")
        if research.notes:
            score -= min(len(research.notes) * 4, 12)
            risks.extend(note for note in research.notes[:2] if "failed" in note.lower())
        return _LayerResult(
            score=self._clamp(score),
            evidence=evidence,
            risks=risks,
            metadata={"news_count": news_count, "note_count": len(research.notes)},
        )

    def _event_risk(self, research: ResearchSnapshot, market_layer: _LayerResult) -> _LayerResult:
        text = self._news_text(research)
        earnings_hits = self._term_hits(text, {"earnings", "guidance", "quarter", "results"})
        macro_hits = self._term_hits(text, MACRO_TERMS | GEOPOLITICAL_TERMS)
        score = 72.0
        evidence: list[str] = []
        risks: list[str] = []
        if earnings_hits:
            evidence.append("Upcoming/recent earnings language is present in headlines.")
            score -= min(earnings_hits * 4, 10)
        if macro_hits:
            risks.append("event_risk_macro_or_geopolitical")
            score -= min(macro_hits * 6, 24)
        vol20 = float(market_layer.metadata.get("vol20") or 0)
        if vol20 > self.settings.catalyst.max_volatility_pct:
            risks.append("event_risk_high_volatility")
            score -= 10
        if not risks:
            evidence.append("No major macro/geopolitical risk keywords found in headlines.")
        return _LayerResult(
            score=self._clamp(score),
            evidence=evidence,
            risks=risks,
            metadata={"earnings_hits": earnings_hits, "macro_event_hits": macro_hits},
        )

    def _block_reason(
        self,
        direction: str,
        prediction_score: float,
        confidence: str,
        risks: list[str],
    ) -> str | None:
        if prediction_score < self.settings.catalyst.min_trade_score:
            return f"Prediction score {prediction_score:.2f} is below {self.settings.catalyst.min_trade_score:.2f}."
        if direction == "neutral":
            return "Prediction direction is neutral."
        if confidence == "low" and self.settings.catalyst.block_low_confidence:
            return "Prediction confidence is low."
        if self.settings.catalyst.block_rumor_only and "rumor_only_catalyst" in risks:
            return "Catalyst is rumor-only or unverified."
        if self.settings.catalyst.block_high_event_risk and (
            "event_risk_macro_or_geopolitical" in risks or "event_risk_high_volatility" in risks
        ):
            return "Macro/geopolitical or volatility event risk is elevated."
        return None

    def _confidence(
        self,
        source_quality_score: float,
        market_layer: _LayerResult,
        news_layer: _LayerResult,
        quality_layer: _LayerResult,
    ) -> str:
        source_count = 0
        if market_layer.metadata.get("sma20"):
            source_count += 1
        if news_layer.metadata.get("positive_hits") or news_layer.metadata.get("negative_hits"):
            source_count += 1
        if source_quality_score >= 70:
            source_count += 1
        if quality_layer.score >= 60:
            source_count += 1
        if source_quality_score >= 70 and source_count >= 3:
            return "high"
        if source_quality_score >= 45 and source_count >= 2:
            return "medium"
        return "low"

    def _direction(self, bullish_score: float) -> str:
        if bullish_score >= 62:
            return "bullish"
        if bullish_score <= 38:
            return "bearish"
        return "neutral"

    def _strategy_stance(self, strategy: str) -> str:
        if strategy in {"long_put", "put_debit_spread"}:
            return "bearish"
        if strategy == "covered_call":
            return "neutral_income"
        return "bullish"

    def _disabled_prediction(self, market: MarketSnapshot) -> CatalystPrediction:
        return CatalystPrediction(
            symbol=market.symbol,
            asset_class=market.asset_class.value,
            direction="neutral",
            prediction_score=0,
            bullish_score=50,
            confidence="low",
            entry_allowed=False,
            block_reason="Catalyst Engine is disabled.",
            market_regime="disabled",
        )

    def _term_hits(self, text: str, terms: set[str]) -> int:
        return sum(1 for term in terms if term in text)

    def _news_text(self, research: ResearchSnapshot) -> str:
        parts: list[str] = []
        for item in research.news:
            parts.append(item.title)
            if item.summary:
                parts.append(item.summary)
        return " ".join(parts).lower()

    def _average_news_sentiment(self, research: ResearchSnapshot) -> float | None:
        values = [
            item.sentiment_score
            for item in research.news
            if item.sentiment_score is not None
        ]
        if not values:
            return None
        return sum(values) / len(values)

    def _bounded(self, value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    def _clamp(self, value: float) -> float:
        return max(0.0, min(100.0, value))

    def _coerce_float(self, value) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
