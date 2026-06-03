import asyncio

import httpx
import pytest

from trading_agent.config import Settings
from trading_agent.research.news import (
    AlpacaNews,
    MarketNews,
    MarketauxNews,
    NewsCacheStore,
    YahooFinanceNews,
)
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
async def test_yahoo_news_sends_user_agent_and_parses_summary(respx_mock) -> None:
    route = respx_mock.get("https://feeds.finance.yahoo.com/rss/2.0/headline").respond(
        text="""
        <rss><channel>
          <item>
            <title>AAPL announces new product launch</title>
            <description>Apple shared product updates.</description>
            <link>https://example.test/aapl</link>
            <pubDate>Wed, 03 Jun 2026 14:00:00 GMT</pubDate>
          </item>
        </channel></rss>
        """,
        headers={"content-type": "application/rss+xml"},
    )

    news = await YahooFinanceNews().headlines("AAPL")

    assert route.calls.last.request.headers["user-agent"] == "trading-agent/1.0"
    assert news[0].source == "yahoo"
    assert news[0].summary == "Apple shared product updates."


@pytest.mark.anyio
async def test_alpaca_news_parses_headlines(respx_mock) -> None:
    route = respx_mock.get("https://data.alpaca.markets/v1beta1/news").respond(
        json={
            "news": [
                {
                    "headline": "HON wins new automation contract",
                    "url": "https://example.test/hon",
                    "created_at": "2026-06-03T14:00:00Z",
                }
            ]
        }
    )

    news = await AlpacaNews("key", "secret").headlines("HON")

    assert news[0].title == "HON wins new automation contract"
    assert route.calls.last.request.url.params["symbols"] == "HON"


@pytest.mark.anyio
async def test_marketaux_news_parses_entity_sentiment(respx_mock) -> None:
    route = respx_mock.get("https://api.marketaux.com/v1/news/all").respond(
        json={
            "data": [
                {
                    "title": "HON secures automation contract",
                    "description": "Honeywell announces a new industrial partnership.",
                    "url": "https://example.test/hon",
                    "published_at": "2026-06-03T14:00:00Z",
                    "entities": [{"symbol": "HON", "sentiment_score": 0.42}],
                }
            ]
        }
    )

    news = await MarketauxNews("token").headlines("HON")

    assert news[0].source == "marketaux"
    assert news[0].summary == "Honeywell announces a new industrial partnership."
    assert news[0].sentiment_score == 0.42
    assert route.calls.last.request.url.params["symbols"] == "HON"


@pytest.mark.anyio
async def test_market_news_uses_marketaux_first(respx_mock) -> None:
    respx_mock.get("https://api.marketaux.com/v1/news/all").respond(
        json={
            "data": [
                {
                    "title": "AAPL raises guidance",
                    "entities": [{"symbol": "AAPL", "sentiment_score": 0.3}],
                }
            ]
        }
    )

    news = await MarketNews(None, None, "token").headlines("AAPL")

    assert news[0].source == "marketaux"
    assert news[0].sentiment_score == 0.3


@pytest.mark.anyio
async def test_market_news_reuses_cached_marketaux_response(respx_mock, tmp_path) -> None:
    route = respx_mock.get("https://api.marketaux.com/v1/news/all").respond(
        json={
            "data": [
                {
                    "title": "AAPL raises guidance",
                    "entities": [{"symbol": "AAPL", "sentiment_score": 0.3}],
                }
            ]
        }
    )
    cache = NewsCacheStore(tmp_path / "news_cache.sqlite3")
    market_news = MarketNews(
        None,
        None,
        "token",
        cache_store=cache,
        cache_ttl_seconds=1800,
        provider_daily_limits={"marketaux": 90},
    )

    first = await market_news.headlines("AAPL")
    second = await market_news.headlines("AAPL")

    assert first[0].title == "AAPL raises guidance"
    assert second[0].title == "AAPL raises guidance"
    assert route.call_count == 1
    assert cache.calls_today("marketaux") == 1


@pytest.mark.anyio
async def test_market_news_caches_empty_provider_response_and_falls_back(respx_mock, tmp_path) -> None:
    marketaux_route = respx_mock.get("https://api.marketaux.com/v1/news/all").respond(json={"data": []})
    yahoo_route = respx_mock.get("https://feeds.finance.yahoo.com/rss/2.0/headline").respond(
        text="""
        <rss><channel>
          <item>
            <title>AAPL services revenue expands</title>
            <description>Apple services revenue expands.</description>
          </item>
        </channel></rss>
        """,
        headers={"content-type": "application/rss+xml"},
    )
    cache = NewsCacheStore(tmp_path / "news_cache.sqlite3")
    market_news = MarketNews(
        "alpaca-key",
        "alpaca-secret",
        "marketaux-token",
        cache_store=cache,
        cache_ttl_seconds=1800,
        provider_daily_limits={"marketaux": 90},
    )

    first = await market_news.headlines("AAPL")
    second = await market_news.headlines("AAPL")

    assert first[0].source == "yahoo"
    assert second[0].source == "yahoo"
    assert marketaux_route.call_count == 1
    assert yahoo_route.call_count == 1


@pytest.mark.anyio
async def test_market_news_falls_back_to_yahoo_when_marketaux_is_exhausted(respx_mock) -> None:
    respx_mock.get("https://api.marketaux.com/v1/news/all").respond(
        402,
        json={"error": {"code": "usage_limit_reached", "message": "Usage limit reached."}},
    )
    respx_mock.get("https://feeds.finance.yahoo.com/rss/2.0/headline").respond(
        text="""
        <rss><channel>
          <item><title>AAPL announces services growth</title></item>
        </channel></rss>
        """,
        headers={"content-type": "application/rss+xml"},
    )

    news = await MarketNews("alpaca-key", "alpaca-secret", "marketaux-token").headlines("AAPL")

    assert news[0].source == "yahoo"


@pytest.mark.anyio
async def test_market_news_falls_back_to_alpaca_when_yahoo_has_no_news(respx_mock) -> None:
    respx_mock.get("https://feeds.finance.yahoo.com/rss/2.0/headline").respond(
        text="<rss><channel></channel></rss>",
        headers={"content-type": "application/rss+xml"},
    )
    respx_mock.get("https://data.alpaca.markets/v1beta1/news").respond(
        json={"news": [{"headline": "AAPL announces services growth"}]}
    )

    news = await MarketNews("alpaca-key", "alpaca-secret").headlines("AAPL")

    assert news[0].source == "alpaca"


@pytest.mark.anyio
async def test_market_news_skips_provider_after_internal_daily_quota(respx_mock, tmp_path) -> None:
    marketaux_route = respx_mock.get("https://api.marketaux.com/v1/news/all").respond(
        402,
        json={"error": {"code": "usage_limit_reached", "message": "Usage limit reached."}},
    )
    yahoo_route = respx_mock.get("https://feeds.finance.yahoo.com/rss/2.0/headline").respond(
        text="""
        <rss><channel>
          <item><title>AAPL announces services growth</title></item>
        </channel></rss>
        """,
        headers={"content-type": "application/rss+xml"},
    )
    cache = NewsCacheStore(tmp_path / "news_cache.sqlite3")
    market_news = MarketNews(
        "alpaca-key",
        "alpaca-secret",
        "marketaux-token",
        cache_store=cache,
        cache_ttl_seconds=0,
        provider_daily_limits={"marketaux": 90},
    )

    first = await market_news.headlines("AAPL")
    second = await market_news.headlines("MSFT")

    assert first[0].source == "yahoo"
    assert second[0].source == "yahoo"
    assert marketaux_route.call_count == 1
    assert yahoo_route.call_count == 2
    assert cache.calls_today("marketaux") == 90


@pytest.mark.anyio
async def test_market_news_falls_back_to_yahoo_when_alpaca_has_no_credentials(respx_mock) -> None:
    respx_mock.get("https://feeds.finance.yahoo.com/rss/2.0/headline").respond(
        text="""
        <rss><channel>
          <item>
            <title>AAPL announces new product launch</title>
            <link>https://example.test/aapl</link>
            <pubDate>Wed, 03 Jun 2026 14:00:00 GMT</pubDate>
          </item>
        </channel></rss>
        """,
        headers={"content-type": "application/rss+xml"},
    )

    news = await MarketNews(None, None).headlines("AAPL")

    assert [item.title for item in news] == ["AAPL announces new product launch"]


@pytest.mark.anyio
async def test_alpaca_news_normalizes_crypto_symbol(respx_mock) -> None:
    route = respx_mock.get("https://data.alpaca.markets/v1beta1/news").respond(json={"news": []})

    await AlpacaNews("key", "secret").headlines("BTC/USD")

    assert route.calls.last.request.url.params["symbols"] == "BTCUSD"


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
