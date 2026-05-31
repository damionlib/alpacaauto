import asyncio

import httpx
import pytest

from trading_agent.config import Settings
from trading_agent.research.news import YahooFinanceNews
from trading_agent.research.sec import SecClient
from trading_agent.research.service import ResearchService


class FailingNews:
    async def headlines(self, symbol: str, limit: int = 8):
        raise httpx.HTTPStatusError(
            "not found",
            request=httpx.Request("GET", "https://example.test"),
            response=httpx.Response(404),
        )


class FakeCryptoResearch:
    async def research(self, symbol: str):
        return {"regime": {"label": "neutral", "score": 50}, "risk_flags": []}, []


@pytest.mark.anyio
async def test_yahoo_news_404_returns_empty_list(respx_mock) -> None:
    respx_mock.get("https://feeds.finance.yahoo.com/rss/2.0/headline").respond(404)

    news = await YahooFinanceNews().headlines("BTC-USD")

    assert news == []


@pytest.mark.anyio
async def test_crypto_research_continues_when_news_fails() -> None:
    service = ResearchService(Settings())
    service.news = FailingNews()
    service.crypto = FakeCryptoResearch()

    snapshot = await service.research_symbol("BTC/USD")

    assert snapshot.symbol == "BTC/USD"
    assert snapshot.news == []
    assert snapshot.crypto_summary["regime"]["label"] == "neutral"
    assert any("News lookup failed" in note for note in snapshot.notes)


@pytest.mark.anyio
async def test_sec_ticker_map_loads_async_and_caches_once(respx_mock) -> None:
    route = respx_mock.get("https://www.sec.gov/files/company_tickers.json").respond(
        json={"0": {"ticker": "AAPL", "cik_str": 320193, "title": "Apple Inc."}}
    )
    client = SecClient("trading-agent test@example.com")

    # Concurrent first-time lookups must trigger exactly one download (lock dedup),
    # and the call must be awaitable (no blocking sync httpx.get on the loop).
    ciks = await asyncio.gather(client.get_cik("AAPL"), client.get_cik("AAPL"))

    assert ciks == ["0000320193", "0000320193"]
    assert await client.get_cik("MSFT") is None  # served from cache, still 1 fetch
    assert route.call_count == 1
