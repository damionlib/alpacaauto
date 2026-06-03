from __future__ import annotations

import json
import sqlite3
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import quote

import httpx

from trading_agent.models import NewsItem

DATA_API_BASE = "https://data.alpaca.markets"
MARKETAUX_API_BASE = "https://api.marketaux.com"
YAHOO_USER_AGENT = "trading-agent/1.0"


@dataclass(frozen=True)
class CachedNewsResult:
    items: list[NewsItem]
    provider: str
    fetched_at: str


class NewsCacheStore:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("pragma journal_mode = wal")
        connection.execute("pragma synchronous = normal")
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                create table if not exists news_cache (
                    cache_key text primary key,
                    symbol text not null,
                    provider text not null,
                    limit_value integer not null,
                    fetched_at text not null,
                    expires_at text not null,
                    items_json text not null
                );

                create table if not exists news_provider_usage (
                    provider text not null,
                    usage_date text not null,
                    calls integer not null,
                    last_called_at text not null,
                    primary key(provider, usage_date)
                );

                create index if not exists idx_news_cache_expires_at on news_cache(expires_at);
                create index if not exists idx_news_usage_provider_date
                    on news_provider_usage(provider, usage_date);
                """
            )

    def get(self, *, provider: str, symbol: str, limit: int) -> CachedNewsResult | None:
        cache_key = self._cache_key(provider, symbol, limit)
        now = self._now()
        with self._connect() as connection:
            row = connection.execute(
                """
                select *
                from news_cache
                where cache_key = ?
                  and expires_at > ?
                """,
                (cache_key, now.isoformat()),
            ).fetchone()
        if not row:
            return None
        data = json.loads(row["items_json"])
        return CachedNewsResult(
            items=[NewsItem.model_validate(item) for item in data],
            provider=str(row["provider"]),
            fetched_at=str(row["fetched_at"]),
        )

    def put(self, *, provider: str, symbol: str, limit: int, items: list[NewsItem], ttl_seconds: int) -> None:
        now = self._now()
        expires_at = now + timedelta(seconds=ttl_seconds)
        cache_key = self._cache_key(provider, symbol, limit)
        payload = json.dumps([item.model_dump(mode="json") for item in items], sort_keys=True, default=str)
        with self._connect() as connection:
            connection.execute(
                """
                insert into news_cache
                    (cache_key, symbol, provider, limit_value, fetched_at, expires_at, items_json)
                values (?, ?, ?, ?, ?, ?, ?)
                on conflict(cache_key) do update set
                    fetched_at = excluded.fetched_at,
                    expires_at = excluded.expires_at,
                    items_json = excluded.items_json
                """,
                (
                    cache_key,
                    symbol.upper(),
                    provider,
                    limit,
                    now.isoformat(),
                    expires_at.isoformat(),
                    payload,
                ),
            )

    def is_exhausted(self, *, provider: str, daily_limit: int) -> bool:
        if daily_limit <= 0:
            return False
        return self.calls_today(provider) >= daily_limit

    def record_provider_call(self, provider: str) -> None:
        now = self._now()
        usage_date = self._usage_date(now)
        with self._connect() as connection:
            connection.execute(
                """
                insert into news_provider_usage
                    (provider, usage_date, calls, last_called_at)
                values (?, ?, 1, ?)
                on conflict(provider, usage_date) do update set
                    calls = calls + 1,
                    last_called_at = excluded.last_called_at
                """,
                (provider, usage_date, now.isoformat()),
            )

    def mark_provider_exhausted(self, *, provider: str, daily_limit: int) -> None:
        if daily_limit <= 0:
            return
        now = self._now()
        usage_date = self._usage_date(now)
        with self._connect() as connection:
            connection.execute(
                """
                insert into news_provider_usage
                    (provider, usage_date, calls, last_called_at)
                values (?, ?, ?, ?)
                on conflict(provider, usage_date) do update set
                    calls = max(calls, excluded.calls),
                    last_called_at = excluded.last_called_at
                """,
                (provider, usage_date, daily_limit, now.isoformat()),
            )

    def calls_today(self, provider: str) -> int:
        usage_date = self._usage_date(self._now())
        with self._connect() as connection:
            row = connection.execute(
                """
                select calls
                from news_provider_usage
                where provider = ?
                  and usage_date = ?
                """,
                (provider, usage_date),
            ).fetchone()
        return int(row["calls"]) if row else 0

    def _cache_key(self, provider: str, symbol: str, limit: int) -> str:
        return f"{provider}:{symbol.upper()}:{limit}"

    def _now(self) -> datetime:
        return datetime.now(UTC)

    def _usage_date(self, now: datetime) -> str:
        return now.date().isoformat()


class MarketauxNews:
    def __init__(self, api_token: str | None) -> None:
        self.api_token = api_token

    async def headlines(self, symbol: str, limit: int = 8) -> list[NewsItem]:
        if not self.api_token:
            raise RuntimeError("Missing MARKETAUX_API_TOKEN for news lookup.")
        params = {
            "api_token": self.api_token,
            "symbols": self._normalize_symbol(symbol),
            "filter_entities": "true",
            "language": "en",
            "limit": limit,
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(f"{MARKETAUX_API_BASE}/v1/news/all", params=params)
            response.raise_for_status()
        data = response.json()
        items: list[NewsItem] = []
        normalized_symbol = self._normalize_symbol(symbol)
        for item in data.get("data", [])[:limit]:
            title = item.get("title") or item.get("headline") or ""
            if not title:
                continue
            items.append(
                NewsItem(
                    title=title,
                    url=item.get("url"),
                    published=item.get("published_at") or item.get("published"),
                    source="marketaux",
                    summary=item.get("description") or item.get("snippet"),
                    sentiment_score=self._entity_sentiment(item, normalized_symbol),
                )
            )
        return items

    def _normalize_symbol(self, symbol: str) -> str:
        if "/" in symbol:
            return symbol.split("/", 1)[0].upper()
        if symbol.endswith("-USD"):
            return symbol[:-4].upper()
        return symbol.upper()

    def _entity_sentiment(self, item: dict, symbol: str) -> float | None:
        for entity in item.get("entities") or []:
            entity_symbol = str(entity.get("symbol") or "").upper()
            if entity_symbol != symbol:
                continue
            try:
                return float(entity.get("sentiment_score"))
            except (TypeError, ValueError):
                return None
        return None


class AlpacaNews:
    def __init__(self, api_key_id: str | None, api_secret_key: str | None) -> None:
        self.api_key_id = api_key_id
        self.api_secret_key = api_secret_key

    async def headlines(self, symbol: str, limit: int = 8) -> list[NewsItem]:
        if not self.api_key_id or not self.api_secret_key:
            raise RuntimeError("Missing Alpaca API credentials for news lookup.")
        params = {
            "symbols": self._normalize_symbol(symbol),
            "limit": limit,
            "sort": "desc",
            "include_content": "false",
            "exclude_contentless": "false",
        }
        headers = {
            "APCA-API-KEY-ID": self.api_key_id,
            "APCA-API-SECRET-KEY": self.api_secret_key,
        }
        async with httpx.AsyncClient(timeout=20.0, headers=headers) as client:
            response = await client.get(f"{DATA_API_BASE}/v1beta1/news", params=params)
            response.raise_for_status()
        data = response.json()
        items: list[NewsItem] = []
        for item in data.get("news", [])[:limit]:
            title = item.get("headline") or item.get("title") or ""
            if not title:
                continue
            items.append(
                NewsItem(
                    title=title,
                    url=item.get("url"),
                    published=item.get("created_at") or item.get("updated_at"),
                    source="alpaca",
                    summary=item.get("summary"),
                )
            )
        return items

    def _normalize_symbol(self, symbol: str) -> str:
        if "/" in symbol:
            return symbol.replace("/", "")
        if symbol.endswith("-USD"):
            return symbol.replace("-USD", "USD")
        return symbol


class YahooFinanceNews:
    async def headlines(self, symbol: str, limit: int = 8) -> list[NewsItem]:
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={quote(symbol)}&region=US&lang=en-US"
        async with httpx.AsyncClient(timeout=20.0, headers={"User-Agent": YAHOO_USER_AGENT}) as client:
            response = await client.get(url)
            if response.status_code == 404:
                return []
            response.raise_for_status()

        root = ET.fromstring(response.text)
        items: list[NewsItem] = []
        for item in root.findall("./channel/item")[:limit]:
            title = item.findtext("title") or ""
            link = item.findtext("link")
            published = item.findtext("pubDate")
            summary = item.findtext("description")
            if title:
                items.append(
                    NewsItem(
                        title=title,
                        url=link,
                        published=published,
                        source="yahoo",
                        summary=summary,
                    )
                )
        return items


class MarketNews:
    def __init__(
        self,
        api_key_id: str | None,
        api_secret_key: str | None,
        marketaux_api_token: str | None = None,
        *,
        cache_store: NewsCacheStore | None = None,
        cache_ttl_seconds: int = 1800,
        provider_daily_limits: dict[str, int] | None = None,
    ) -> None:
        self.marketaux_api_token = marketaux_api_token
        self.api_key_id = api_key_id
        self.api_secret_key = api_secret_key
        self.marketaux = MarketauxNews(marketaux_api_token)
        self.alpaca = AlpacaNews(api_key_id, api_secret_key)
        self.yahoo = YahooFinanceNews()
        self.cache_store = cache_store
        self.cache_ttl_seconds = cache_ttl_seconds
        self.provider_daily_limits = provider_daily_limits or {}

    async def headlines(self, symbol: str, limit: int = 8) -> list[NewsItem]:
        for provider, client, provider_symbol in self._providers(symbol):
            cached = self._cached(provider, provider_symbol, limit)
            if cached is not None:
                if cached.items:
                    return cached.items
                continue
            if self._quota_exhausted(provider):
                continue
            try:
                headlines = await client.headlines(provider_symbol, limit)
            except Exception as exc:
                self._record_provider_call(provider)
                if self._is_quota_error(exc):
                    self._mark_provider_exhausted(provider)
                continue
            self._record_provider_call(provider)
            self._cache(provider, provider_symbol, limit, headlines)
            if headlines:
                return headlines
        return []

    def _yahoo_symbol(self, symbol: str) -> str:
        if "/" in symbol:
            return symbol.replace("/", "-")
        return symbol

    def _providers(self, symbol: str) -> list[tuple[str, object, str]]:
        providers: list[tuple[str, object, str]] = []
        if self.marketaux_api_token:
            providers.append(("marketaux", self.marketaux, symbol))
        providers.append(("yahoo", self.yahoo, self._yahoo_symbol(symbol)))
        if self.api_key_id and self.api_secret_key:
            providers.append(("alpaca", self.alpaca, symbol))
        return providers

    def _cached(self, provider: str, symbol: str, limit: int) -> CachedNewsResult | None:
        if not self.cache_store or self.cache_ttl_seconds <= 0:
            return None
        return self.cache_store.get(provider=provider, symbol=symbol, limit=limit)

    def _cache(self, provider: str, symbol: str, limit: int, headlines: list[NewsItem]) -> None:
        if not self.cache_store or self.cache_ttl_seconds <= 0:
            return
        self.cache_store.put(
            provider=provider,
            symbol=symbol,
            limit=limit,
            items=headlines,
            ttl_seconds=self.cache_ttl_seconds,
        )

    def _quota_exhausted(self, provider: str) -> bool:
        if not self.cache_store:
            return False
        return self.cache_store.is_exhausted(
            provider=provider,
            daily_limit=self.provider_daily_limits.get(provider, 0),
        )

    def _record_provider_call(self, provider: str) -> None:
        if self.cache_store:
            self.cache_store.record_provider_call(provider)

    def _mark_provider_exhausted(self, provider: str) -> None:
        if self.cache_store:
            self.cache_store.mark_provider_exhausted(
                provider=provider,
                daily_limit=self.provider_daily_limits.get(provider, 0),
            )

    def _is_quota_error(self, exc: Exception) -> bool:
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code in {402, 429}
        message = str(exc).lower()
        return any(
            token in message
            for token in (
                "api call frequency",
                "exhausted",
                "frequency",
                "limit",
                "quota",
                "rate",
                "usage",
            )
        )
