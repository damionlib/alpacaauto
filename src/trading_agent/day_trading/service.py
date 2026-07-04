from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

from trading_agent.catalyst.models import CatalystPrediction
from trading_agent.config import Settings
from trading_agent.indicators import pct_change, sma
from trading_agent.models import AssetClass, MarketSnapshot, OrderSide, Position, ResearchSnapshot, TradeCandidate


class DayTradingEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._peak_pnl: dict[str, float] = {}

    def enabled(self) -> bool:
        if not self.settings.day_trading.enabled:
            return False
        if self.settings.day_trading.paper_only and self.settings.is_live:
            return False
        return True

    def evaluate_entry(
        self,
        market: MarketSnapshot,
        research: ResearchSnapshot,
        prediction: CatalystPrediction,
        positions: list[Position],
        *,
        trades_used_today: int = 0,
    ) -> tuple[TradeCandidate | None, dict]:
        signal = self._base_signal(market, research, prediction)
        signal["kind"] = "entry"
        if not self.enabled():
            signal.update({"status": "disabled", "reason": "Day trading is disabled."})
            return None, signal
        if market.asset_class not in {AssetClass.EQUITY, AssetClass.ETF}:
            signal.update({"status": "blocked", "reason": "Day trading is limited to stocks and ETFs."})
            return None, signal
        if self._position_for(market.symbol, positions):
            signal.update({"status": "blocked", "reason": "A position already exists for this symbol."})
            return None, signal
        if trades_used_today >= self.settings.day_trading.max_trades_per_day:
            signal.update({"status": "blocked", "reason": "Daily day-trade entry cap reached."})
            return None, signal
        if prediction.direction != "bullish":
            signal.update({"status": "blocked", "reason": f"Catalyst direction is {prediction.direction}, not bullish."})
            return None, signal
        if prediction.prediction_score < self.settings.day_trading.min_catalyst_score:
            signal.update({"status": "blocked", "reason": "Catalyst score is below day-trading threshold."})
            return None, signal
        if signal["intraday_score"] < self.settings.day_trading.min_intraday_score:
            signal.update({"status": "blocked", "reason": "Intraday score is below day-trading threshold."})
            return None, signal
        if signal["combined_score"] < self.settings.day_trading.min_combined_score:
            signal.update({"status": "blocked", "reason": "Combined day-trade score is below threshold."})
            return None, signal
        execution_reason = self._execution_block_reason(market)
        if execution_reason:
            signal.update({"status": "blocked", "reason": execution_reason})
            return None, signal

        cfg = self.settings.day_trading
        adr_pct = self._float((market.metadata or {}).get("adr_pct"))
        if adr_pct and adr_pct > 0:
            ceiling_pct = round(min(adr_pct * cfg.ceiling_adr_fraction, cfg.take_profit_pct), 4)
            floor_pct = round(min(adr_pct * cfg.floor_adr_fraction, cfg.stop_loss_pct), 4)
            partial_pct = round(min(adr_pct * cfg.partial_exit_adr_fraction, cfg.partial_exit_max_pct), 4)
        else:
            ceiling_pct = cfg.take_profit_pct
            floor_pct = cfg.stop_loss_pct
            partial_pct = cfg.partial_exit_max_pct
        stop_price = market.price * (1 - floor_pct / 100)
        take_profit = market.price * (1 + ceiling_pct / 100)
        signal.update({
            "status": "generated",
            "reason": "Day-trade entry setup passed all signal gates.",
            "adr_pct": adr_pct,
            "ceiling_pct": ceiling_pct,
            "floor_pct": floor_pct,
            "partial_pct": partial_pct,
        })
        candidate = TradeCandidate(
            symbol=market.symbol,
            asset_class=market.asset_class,
            side=OrderSide.BUY,
            strategy="day_trade_entry",
            score=round(signal["combined_score"], 2),
            entry_price=market.price,
            stop_price=round(stop_price, 2),
            take_profit_price=round(take_profit, 2),
            rationale=[
                f"Day trade setup {signal['setup']} scored {signal['combined_score']:.2f}.",
                f"Catalyst {prediction.direction} score {prediction.prediction_score:.2f}.",
                f"ADR {adr_pct:.2f}% → ceiling +{ceiling_pct:.2f}%, floor -{floor_pct:.2f}%."
                if adr_pct else f"No ADR; using defaults +{ceiling_pct:.2f}%/-{floor_pct:.2f}%.",
                *signal["evidence"][:3],
            ],
            metadata={
                "day_trade": True,
                "day_trade_entry": True,
                "setup": signal["setup"],
                "signal": signal,
                "adr_pct": adr_pct,
                "ceiling_pct": ceiling_pct,
                "floor_pct": floor_pct,
                "partial_pct": partial_pct,
                "catalyst": prediction.model_dump(mode="json"),
            },
        )
        return candidate, signal

    def evaluate_exit(
        self,
        position: Position,
        market: MarketSnapshot,
        research: ResearchSnapshot,
        prediction: CatalystPrediction,
        *,
        entry_event: dict | None,
        now: datetime | None = None,
    ) -> tuple[TradeCandidate | None, dict]:
        signal = self._base_signal(market, research, prediction)
        signal["kind"] = "exit"
        signal["status"] = "held"
        signal["reason"] = "No day-trade exit trigger."
        if not self.enabled():
            signal.update({"status": "disabled", "reason": "Day trading is disabled."})
            return None, signal
        if position.asset_class not in {AssetClass.EQUITY, AssetClass.ETF} or position.qty <= 0:
            signal.update({"status": "blocked", "reason": "Only long stock/ETF day-trade positions are managed."})
            return None, signal

        current_price = market.price or position.current_price or position.avg_entry_price
        if not current_price or not position.avg_entry_price:
            signal.update({"status": "blocked", "reason": "Position lacks price data for day-trade exit."})
            return None, signal

        pnl_pct = ((current_price - position.avg_entry_price) / position.avg_entry_price) * 100
        held_minutes = self._held_minutes(entry_event, now=now)
        prev_peak = self._peak_pnl.get(position.symbol, 0.0)
        peak_pnl_pct = max(prev_peak, pnl_pct)
        self._peak_pnl[position.symbol] = peak_pnl_pct
        entry_meta = self._entry_metadata(entry_event)
        cfg = self.settings.day_trading
        ceiling_pct = self._float(entry_meta.get("ceiling_pct")) or cfg.take_profit_pct
        floor_pct = self._float(entry_meta.get("floor_pct")) or cfg.stop_loss_pct
        partial_pct = self._float(entry_meta.get("partial_pct")) or cfg.partial_exit_max_pct
        # A position smaller than the entry order means the partial scale-out already
        # banked its half; the remainder runs to the ceiling behind a breakeven stop.
        entry_qty = self._entry_qty(entry_event)
        partial_done = entry_qty is not None and position.qty < entry_qty - 1e-9
        exit_reason, exit_score = self._exit_reason(
            pnl_pct=pnl_pct,
            held_minutes=held_minutes,
            prediction=prediction,
            intraday_score=float(signal["intraday_score"]),
            now=now,
            peak_pnl_pct=peak_pnl_pct,
            ceiling_pct=ceiling_pct,
            floor_pct=floor_pct,
            breakeven_floor=partial_done,
        )
        signal["pnl_pct"] = round(pnl_pct, 4)
        signal["peak_pnl_pct"] = round(peak_pnl_pct, 4)
        signal["held_minutes"] = held_minutes
        signal["partial_done"] = partial_done
        if not exit_reason:
            partial_candidate = self._partial_exit_candidate(
                position=position,
                current_price=current_price,
                pnl_pct=pnl_pct,
                partial_pct=partial_pct,
                ceiling_pct=ceiling_pct,
                partial_done=partial_done,
                prediction=prediction,
                signal=signal,
                entry_event=entry_event,
            )
            if partial_candidate is not None:
                return partial_candidate, signal
            return None, signal

        signal.update({"status": "generated", "reason": exit_reason})
        self._peak_pnl.pop(position.symbol, None)
        candidate = TradeCandidate(
            symbol=position.symbol,
            asset_class=position.asset_class,
            side=OrderSide.SELL,
            strategy="day_trade_exit",
            score=exit_score,
            entry_price=current_price,
            rationale=[
                exit_reason,
                f"Day-trade P/L is {pnl_pct:.2f}%; held for {held_minutes if held_minutes is not None else 'unknown'} minutes.",
                f"Intraday score is {signal['intraday_score']:.2f}; catalyst direction is {prediction.direction}.",
            ],
            metadata={
                "exit": True,
                "exit_qty": abs(position.qty),
                "day_trade": True,
                "day_trade_exit": True,
                "signal": signal,
                "entry_event_id": entry_event.get("id") if entry_event else None,
                "position": position.model_dump(mode="json"),
                "catalyst": prediction.model_dump(mode="json"),
            },
        )
        return candidate, signal

    def _base_signal(
        self,
        market: MarketSnapshot,
        research: ResearchSnapshot,
        prediction: CatalystPrediction,
    ) -> dict:
        intraday = self._intraday_score(market)
        research_score = self._research_score(research)
        catalyst_score = prediction.prediction_score if prediction.direction == "bullish" else 100 - prediction.prediction_score
        execution_score = self._execution_score(market)
        combined = (
            catalyst_score * 0.30
            + research_score * 0.15
            + intraday["score"] * 0.40
            + execution_score * 0.15
        )
        return {
            "symbol": market.symbol,
            "asset_class": market.asset_class.value,
            "setup": intraday["setup"],
            "combined_score": round(max(0.0, min(100.0, combined)), 2),
            "catalyst_score": round(catalyst_score, 2),
            "research_score": round(research_score, 2),
            "intraday_score": round(intraday["score"], 2),
            "execution_score": round(execution_score, 2),
            "components": {
                "catalyst": round(catalyst_score, 2),
                "research": round(research_score, 2),
                "intraday": round(intraday["score"], 2),
                "execution": round(execution_score, 2),
            },
            "evidence": intraday["evidence"],
            "risks": [*intraday["risks"], *self._news_risks(research), *prediction.risks[:3]],
            "market_metadata": market.metadata,
        }

    def _intraday_score(self, market: MarketSnapshot) -> dict:
        metadata = market.metadata or {}
        closes = market.closes
        score = 45.0
        evidence: list[str] = []
        risks: list[str] = []
        setup = "intraday_trend"

        vwap = self._float(metadata.get("vwap"))
        if vwap:
            if market.price >= vwap:
                score += 12
                evidence.append("Price is above VWAP.")
            else:
                score -= 14
                risks.append("Price is below VWAP.")

        relative_volume = self._float(metadata.get("relative_volume"))
        if relative_volume is not None:
            if relative_volume >= self.settings.day_trading.min_relative_volume:
                score += min(12, relative_volume * 4)
                evidence.append(f"Relative volume is {relative_volume:.2f}x.")
            else:
                score -= 8
                risks.append(f"Relative volume is only {relative_volume:.2f}x.")

        minute_trend = self._float(metadata.get("minute_trend_pct"))
        if minute_trend is None:
            minute_trend = pct_change(closes, min(5, max(len(closes) - 1, 1))) if len(closes) >= 6 else None
        if minute_trend is not None:
            score += max(-18, min(18, minute_trend * 4))
            if minute_trend > 0:
                evidence.append(f"Short-term trend is positive at {minute_trend:.2f}%.")
            else:
                risks.append(f"Short-term trend is weak at {minute_trend:.2f}%.")

        sma20 = sma(closes, 20)
        sma50 = sma(closes, 50)
        if sma20 and sma50:
            if market.price > sma20 > sma50:
                score += 18
                setup = "trend_continuation"
                evidence.append("Price is above SMA20 and SMA50.")
            elif market.price < sma20:
                score -= 12
                risks.append("Price is below SMA20.")

        opening_range_break = metadata.get("opening_range_break")
        if opening_range_break is True:
            score += 14
            setup = "opening_range_break"
            evidence.append("Opening range breakout flag is present.")

        return {
            "score": max(0.0, min(100.0, score)),
            "setup": setup,
            "evidence": evidence,
            "risks": risks,
        }

    def _research_score(self, research: ResearchSnapshot) -> float:
        score = 50.0
        if research.news:
            score += min(15, len(research.news) * 3)
        if research.sec_summary.get("latest_net_income"):
            score += 10
        if research.sec_summary.get("recent_filings"):
            score += 5
        negative = len(self._news_risks(research))
        score -= min(25, negative * 8)
        return max(0.0, min(100.0, score))

    def _execution_score(self, market: MarketSnapshot) -> float:
        spread_pct = self._spread_pct(market)
        if spread_pct is None:
            return 70.0
        if spread_pct <= self.settings.day_trading.max_spread_pct / 2:
            return 95.0
        if spread_pct <= self.settings.day_trading.max_spread_pct:
            return 80.0
        return 35.0

    def _execution_block_reason(self, market: MarketSnapshot) -> str | None:
        spread_pct = self._spread_pct(market)
        if spread_pct is not None and spread_pct > self.settings.day_trading.max_spread_pct:
            return f"Spread {spread_pct:.3f}% exceeds day-trading max spread."
        relative_volume = self._float((market.metadata or {}).get("relative_volume"))
        if relative_volume is not None and relative_volume < self.settings.day_trading.min_relative_volume:
            return "Relative volume is below day-trading threshold."
        return None

    def _exit_reason(
        self,
        *,
        pnl_pct: float,
        held_minutes: int | None,
        prediction: CatalystPrediction,
        intraday_score: float,
        now: datetime | None,
        peak_pnl_pct: float = 0.0,
        ceiling_pct: float | None = None,
        floor_pct: float | None = None,
        breakeven_floor: bool = False,
    ) -> tuple[str | None, float]:
        cfg = self.settings.day_trading
        effective_floor = floor_pct or cfg.stop_loss_pct
        effective_ceiling = ceiling_pct or cfg.take_profit_pct
        if breakeven_floor and pnl_pct <= 0:
            return (
                f"Breakeven stop on the remainder after partial profit-take ({pnl_pct:.2f}%).",
                100,
            )
        if pnl_pct <= -effective_floor:
            return (
                f"ADR floor stop hit at {pnl_pct:.2f}% (floor -{effective_floor:.2f}%).",
                100,
            )
        if pnl_pct <= -cfg.stop_loss_pct:
            return f"Hard stop loss hit at {pnl_pct:.2f}%.", 100
        if self._force_exit_window(now=now):
            return "Approaching market close; day-trade positions must not be held overnight.", 98
        if pnl_pct >= effective_ceiling:
            return (
                f"ADR ceiling target hit at +{pnl_pct:.2f}% (target +{effective_ceiling:.2f}%).",
                96,
            )
        if prediction.direction == "bearish" and prediction.prediction_score >= cfg.min_catalyst_score:
            return "Catalyst flipped bearish for the day-trade position.", 94
        if cfg.profit_protect_pct and peak_pnl_pct >= cfg.profit_protect_pct:
            trail_floor = peak_pnl_pct - cfg.profit_trail_pct
            if cfg.profit_trail_pct and pnl_pct <= trail_floor:
                return (
                    f"Trailing stop: P/L fell to {pnl_pct:.2f}% from peak "
                    f"{peak_pnl_pct:.2f}% (floor {trail_floor:.2f}%).",
                    95,
                )
        if peak_pnl_pct >= 0.3 and pnl_pct <= 0.0 and cfg.profit_protect_pct:
            return (
                f"Profit reversal: peaked at +{peak_pnl_pct:.2f}% but now {pnl_pct:.2f}%.",
                92,
            )
        if intraday_score <= cfg.exit_intraday_score:
            return "Intraday trend score fell below exit threshold.", 88
        if (
            held_minutes is not None
            and cfg.stale_negative_minutes > 0
            and held_minutes >= cfg.stale_negative_minutes
            and peak_pnl_pct < 0.2
        ):
            return (
                f"Stale negative: held {held_minutes} min, peak P/L only "
                f"{peak_pnl_pct:.2f}%.",
                86,
            )
        if held_minutes is not None and cfg.time_decay_minutes > 0:
            if held_minutes >= 90 and pnl_pct < 0.5:
                return (
                    f"Time decay: {pnl_pct:.2f}% P/L after {held_minutes} min "
                    f"(need >= 0.5%).",
                    85,
                )
            if held_minutes >= cfg.time_decay_minutes and pnl_pct < 0.3:
                return (
                    f"Time decay: {pnl_pct:.2f}% P/L after {held_minutes} min "
                    f"(need >= 0.3%).",
                    83,
                )
        if held_minutes is not None and held_minutes >= cfg.max_position_minutes:
            return "Day-trade maximum holding time reached.", 84
        return None, 0

    def _news_risks(self, research: ResearchSnapshot) -> list[str]:
        negative_terms = ("downgrade", "fraud", "investigation", "lawsuit", "miss", "recall", "warning")
        text = " ".join(f"{item.title} {item.summary or ''}".lower() for item in research.news)
        return [f"negative_news:{term}" for term in negative_terms if term in text]

    def _spread_pct(self, market: MarketSnapshot) -> float | None:
        metadata = market.metadata or {}
        explicit = self._float(metadata.get("spread_pct"))
        if explicit is not None:
            return explicit
        bid = self._float(metadata.get("bid"))
        ask = self._float(metadata.get("ask"))
        if bid is None or ask is None or bid <= 0 or ask <= 0:
            return None
        mid = (bid + ask) / 2
        return ((ask - bid) / mid) * 100 if mid > 0 else None

    def _held_minutes(self, entry_event: dict | None, *, now: datetime | None) -> int | None:
        if not entry_event or not entry_event.get("created_at"):
            return None
        try:
            started = datetime.fromisoformat(str(entry_event["created_at"]))
        except ValueError:
            return None
        current = now or datetime.now(UTC)
        if started.tzinfo is None:
            started = started.replace(tzinfo=UTC)
        if current.tzinfo is None:
            current = current.replace(tzinfo=UTC)
        return max(int((current.astimezone(UTC) - started.astimezone(UTC)).total_seconds() // 60), 0)

    def _force_exit_window(self, *, now: datetime | None) -> bool:
        minutes = self.settings.day_trading.force_exit_before_close_minutes
        if minutes <= 0:
            return False
        local_now = (now or datetime.now(UTC)).astimezone(ZoneInfo("America/Chicago"))
        if local_now.weekday() >= 5:
            return False
        close_time = datetime.combine(local_now.date(), time(15, 0), tzinfo=local_now.tzinfo)
        return close_time - timedelta(minutes=minutes) <= local_now <= close_time + timedelta(minutes=30)

    def _position_for(self, symbol: str, positions: list[Position]) -> Position | None:
        for position in positions:
            if position.symbol == symbol and position.qty != 0:
                return position
        return None

    def _partial_exit_candidate(
        self,
        *,
        position: Position,
        current_price: float,
        pnl_pct: float,
        partial_pct: float,
        ceiling_pct: float,
        partial_done: bool,
        prediction: CatalystPrediction,
        signal: dict,
        entry_event: dict | None,
    ) -> TradeCandidate | None:
        cfg = self.settings.day_trading
        if partial_done or cfg.partial_exit_size <= 0 or partial_pct <= 0:
            return None
        if pnl_pct < partial_pct or position.qty < 2:
            return None
        sell_qty = max(1, int(position.qty * cfg.partial_exit_size))
        remainder_qty = int(position.qty) - sell_qty
        if remainder_qty < 1:
            return None
        entry_price = position.avg_entry_price or current_price
        reason = (
            f"Partial profit-take: +{pnl_pct:.2f}% reached partial target "
            f"+{partial_pct:.2f}%; selling {sell_qty} of {int(position.qty)}, "
            f"remainder runs to +{ceiling_pct:.2f}% behind a breakeven stop."
        )
        signal.update({"status": "generated", "reason": reason})
        return TradeCandidate(
            symbol=position.symbol,
            asset_class=position.asset_class,
            side=OrderSide.SELL,
            strategy="day_trade_exit",
            score=90,
            entry_price=current_price,
            rationale=[
                reason,
                f"Intraday score is {signal['intraday_score']:.2f}; catalyst direction is {prediction.direction}.",
            ],
            metadata={
                "exit": True,
                "exit_qty": sell_qty,
                "partial_exit": True,
                "remainder_qty": remainder_qty,
                "remainder_stop_price": round(entry_price, 2),
                "remainder_take_profit_price": round(entry_price * (1 + ceiling_pct / 100), 2),
                "day_trade": True,
                "day_trade_exit": True,
                "signal": signal,
                "entry_event_id": entry_event.get("id") if entry_event else None,
                "position": position.model_dump(mode="json"),
                "catalyst": prediction.model_dump(mode="json"),
            },
        )

    @staticmethod
    def _entry_qty(entry_event: dict | None) -> float | None:
        if not entry_event:
            return None
        payload = entry_event.get("payload") or {}
        intent = payload.get("intent") or {}
        try:
            qty = float(intent.get("qty"))
        except (TypeError, ValueError):
            return None
        return qty if qty > 0 else None

    @staticmethod
    def _entry_metadata(entry_event: dict | None) -> dict:
        if not entry_event:
            return {}
        payload = entry_event.get("payload") or {}
        intent = payload.get("intent") or {}
        return intent.get("metadata") or {}

    def _float(self, value) -> float | None:
        try:
            return None if value is None else float(value)
        except (TypeError, ValueError):
            return None
