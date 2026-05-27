from __future__ import annotations

import asyncio

from trading_agent.config import Settings
from trading_agent.models import ResearchSnapshot
from trading_agent.research.news import YahooFinanceNews
from trading_agent.research.sec import SecClient


class ResearchService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.news = YahooFinanceNews()
        self.sec = SecClient(settings.sec_user_agent)

    async def research_symbol(self, symbol: str) -> ResearchSnapshot:
        if "/" in symbol:
            try:
                news = await self.news.headlines(
                    symbol.replace("/", "-"),
                    self.settings.research.news_headline_limit,
                )
                notes = ["Crypto asset; SEC company facts skipped."]
            except Exception as exc:
                news = []
                notes = [
                    "Crypto asset; SEC company facts skipped.",
                    f"News lookup failed: {exc}",
                ]
            return ResearchSnapshot(symbol=symbol, news=news, notes=notes)

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
