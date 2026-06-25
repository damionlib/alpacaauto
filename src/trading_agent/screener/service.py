from __future__ import annotations

import asyncio
from dataclasses import dataclass

from trading_agent.brokers.base import Broker
from trading_agent.config import Settings
from trading_agent.indicators import pct_change, realized_volatility, sma
from trading_agent.models import AssetClass, MarketSnapshot
from trading_agent.screener.universe import symbols_for_universes


@dataclass(frozen=True)
class ScreenedSymbol:
    symbol: str
    asset_class: AssetClass
    score: float
    reasons: list[str]
    snapshot: MarketSnapshot


class MarketScreener:
    def __init__(self, settings: Settings, broker: Broker) -> None:
        self.settings = settings
        self.broker = broker

    async def day_trade_symbols(self) -> list[str]:
        getter = getattr(self.broker, "get_most_actives", None)
        if not getter:
            return []
        try:
            actives = await getter(20)
        except Exception:
            actives = []
        mover_getter = getattr(self.broker, "get_movers", None)
        movers: list[dict] = []
        if mover_getter:
            try:
                data = await mover_getter(10)
                movers = data.get("gainers", [])
            except Exception:
                pass
        seen: set[str] = set()
        symbols: list[str] = []
        for item in actives:
            sym = item.get("symbol", "")
            if sym and sym not in seen:
                seen.add(sym)
                symbols.append(sym)
        for item in movers:
            sym = item.get("symbol", "")
            if sym and sym not in seen and item.get("price", 0) >= self.settings.screener.min_price:
                seen.add(sym)
                symbols.append(sym)
        return symbols

    async def top_symbols(self) -> list[ScreenedSymbol]:
        if not self.settings.screener.enabled:
            return []

        raw_symbols = symbols_for_universes(self.settings.screener.universes)
        if self.settings.day_trading.enabled:
            live_symbols = await self.day_trade_symbols()
            for sym in live_symbols:
                if sym not in raw_symbols:
                    raw_symbols.append(sym)
        if not self.settings.strategy.allow_crypto:
            raw_symbols = [symbol for symbol in raw_symbols if "/" not in symbol]
        equity_symbols = [s for s in raw_symbols if "/" not in s]
        crypto_symbols = [s for s in raw_symbols if "/" in s]

        batch_getter = getattr(self.broker, "get_snapshots_batch", None)
        candidates: list[ScreenedSymbol] = []

        if batch_getter and equity_symbols:
            for chunk_start in range(0, len(equity_symbols), 50):
                chunk = equity_symbols[chunk_start : chunk_start + 50]
                try:
                    snapshots = await batch_getter(chunk)
                except Exception:
                    snapshots = {}
                bar_data = await self._get_bars_for_symbols(chunk)
                for sym in chunk:
                    snap_data = snapshots.get(sym)
                    bars = bar_data.get(sym)
                    if not bars:
                        continue
                    closes, volumes = bars
                    if not closes:
                        continue
                    price = closes[-1]
                    if snap_data:
                        trade = snap_data.get("latestTrade", {})
                        if trade.get("p"):
                            price = float(trade["p"])
                    snapshot = MarketSnapshot(
                        symbol=sym, asset_class=AssetClass.EQUITY,
                        price=price, closes=closes,
                        metadata={"volumes": volumes},
                    )
                    scored = self.score_snapshot(snapshot)
                    if scored:
                        candidates.append(scored)
        else:
            tasks = [self._screen_symbol(symbol) for symbol in equity_symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, ScreenedSymbol):
                    candidates.append(result)

        crypto_tasks = [self._screen_symbol(symbol) for symbol in crypto_symbols]
        crypto_results = await asyncio.gather(*crypto_tasks, return_exceptions=True)
        for result in crypto_results:
            if isinstance(result, ScreenedSymbol):
                candidates.append(result)

        equity_candidates = [
            candidate
            for candidate in candidates
            if candidate.asset_class != AssetClass.CRYPTO
        ]
        crypto_candidates = [
            candidate
            for candidate in candidates
            if candidate.asset_class == AssetClass.CRYPTO
        ]
        equity_limit = max(self.settings.screener.max_candidates - self.settings.screener.max_crypto_candidates, 0)
        selected = [
            *sorted(equity_candidates, key=lambda item: item.score, reverse=True)[:equity_limit],
            *sorted(crypto_candidates, key=lambda item: item.score, reverse=True)[: self.settings.screener.max_crypto_candidates],
        ]
        return sorted(selected, key=lambda item: item.score, reverse=True)[: self.settings.screener.max_candidates]

    async def _get_bars_for_symbols(self, symbols: list[str]) -> dict[str, tuple[list[float], list[float]]]:
        getter = getattr(self.broker, "_get_stock_bars", None)
        if not getter:
            return {}
        try:
            return await getter(symbols)
        except Exception:
            return {}

    async def _screen_symbol(self, symbol: str) -> ScreenedSymbol | None:
        try:
            snapshot = await self.broker.get_market_snapshot(symbol)
        except Exception:
            return None
        return self.score_snapshot(snapshot)

    def score_snapshot(self, snapshot: MarketSnapshot) -> ScreenedSymbol | None:
        closes = snapshot.closes
        if len(closes) < 50:
            return None
        if snapshot.price < self.settings.screener.min_price and snapshot.asset_class != AssetClass.CRYPTO:
            return None

        sma20 = sma(closes, 20)
        sma50 = sma(closes, 50)
        change20 = pct_change(closes, 20)
        vol20 = realized_volatility(closes, 20)
        if sma20 is None or sma50 is None or change20 is None or vol20 is None:
            return None
        if vol20 > self.settings.screener.max_realized_volatility_pct:
            return None

        trend_points = 35 if snapshot.price > sma20 > sma50 else 0
        momentum_points = max(min(change20 * 2, 30), -20)
        volatility_points = max(20 - (vol20 / 5), 0)
        liquidity_points = self._liquidity_points(snapshot)
        crypto_bonus = 5 if snapshot.asset_class == AssetClass.CRYPTO else 0
        score = max(0.0, min(100.0, 20 + trend_points + momentum_points + volatility_points + liquidity_points + crypto_bonus))

        if score < self.settings.screener.min_trend_score:
            return None
        if snapshot.asset_class != AssetClass.CRYPTO and self._avg_dollar_volume(snapshot) < self.settings.screener.min_avg_dollar_volume:
            return None

        reasons = [
            f"price={snapshot.price:.2f}",
            f"sma20={sma20:.2f}",
            f"sma50={sma50:.2f}",
            f"change20={change20:.2f}%",
            f"vol20={vol20:.2f}%",
        ]
        return ScreenedSymbol(
            symbol=snapshot.symbol,
            asset_class=snapshot.asset_class,
            score=round(score, 2),
            reasons=reasons,
            snapshot=snapshot,
        )

    def _avg_dollar_volume(self, snapshot: MarketSnapshot) -> float:
        volumes = snapshot.metadata.get("volumes", [])
        closes = snapshot.closes
        if not volumes or not closes:
            return self.settings.screener.min_avg_dollar_volume
        paired = list(zip(closes[-20:], volumes[-20:], strict=False))
        if not paired:
            return 0.0
        return sum(close * volume for close, volume in paired) / len(paired)

    def _liquidity_points(self, snapshot: MarketSnapshot) -> float:
        if snapshot.asset_class == AssetClass.CRYPTO:
            return 8
        avg_dollar_volume = self._avg_dollar_volume(snapshot)
        if avg_dollar_volume >= self.settings.screener.min_avg_dollar_volume * 5:
            return 10
        if avg_dollar_volume >= self.settings.screener.min_avg_dollar_volume:
            return 5
        return 0
