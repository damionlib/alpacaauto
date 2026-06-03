from __future__ import annotations

import asyncio

from trading_agent.config import Settings
from trading_agent.models import ResearchSnapshot
from trading_agent.research.crypto import CryptoResearchService
from trading_agent.research.news import MarketNews, NewsCacheStore
from trading_agent.research.sec import SecClient


class ResearchService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        api_key_id = settings.alpaca_api_key_id.get_secret_value() if settings.alpaca_api_key_id else None
        api_secret_key = settings.alpaca_api_secret_key.get_secret_value() if settings.alpaca_api_secret_key else None
        marketaux_api_token = (
            settings.marketaux_api_token.get_secret_value() if settings.marketaux_api_token else None
        )
        news_cache = (
            NewsCacheStore(settings.research.news_cache_database_path)
            if settings.research.news_cache_enabled
            else None
        )
        self.news = MarketNews(
            api_key_id,
            api_secret_key,
            marketaux_api_token,
            cache_store=news_cache,
            cache_ttl_seconds=settings.research.news_cache_ttl_seconds,
            provider_daily_limits={
                "marketaux": settings.research.marketaux_daily_call_limit,
                "alpaca": settings.research.alpaca_news_daily_call_limit,
                "yahoo": settings.research.yahoo_news_daily_call_limit,
            },
        )
        self.sec = SecClient(settings.sec_user_agent)
        self.crypto = CryptoResearchService(settings)

    async def research_symbol(self, symbol: str) -> ResearchSnapshot:
        if "/" in symbol:
            crypto_task = self.crypto.research(symbol)
            try:
                news = await self.news.headlines(
                    symbol,
                    self.settings.research.news_headline_limit,
                )
                notes = ["Crypto asset; SEC company facts skipped."]
            except Exception as exc:
                news = []
                notes = [
                    "Crypto asset; SEC company facts skipped.",
                    f"News lookup failed: {exc}",
                ]
            crypto_summary, crypto_notes = await crypto_task
            notes.extend(crypto_notes)
            return ResearchSnapshot(
                symbol=symbol,
                news=news,
                crypto_summary=crypto_summary,
                notes=notes,
            )

        news_task = self.news.headlines(symbol, self.settings.research.news_headline_limit)
        sec_task = (
            self.sec.get_company_summary(symbol)
            if self.settings.research.sec_companyfacts_enabled
            else self._empty_sec()
        )
        news, sec_summary = await asyncio.gather(news_task, sec_task, return_exceptions=True)
        notes: list[str] = []
        if isinstance(news, Exception):
            notes.append(f"News lookup failed: {news}")
            news = []
        if isinstance(sec_summary, Exception):
            notes.append(f"SEC lookup failed: {sec_summary}")
            sec_summary = {}
        return ResearchSnapshot(
            symbol=symbol,
            news=news,
            sec_summary=sec_summary,
            notes=notes,
        )

    async def _empty_sec(self) -> dict:
        return {}
