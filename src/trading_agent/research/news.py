from __future__ import annotations

import xml.etree.ElementTree as ET
from urllib.parse import quote

import httpx

from trading_agent.models import NewsItem


class YahooFinanceNews:
    async def headlines(self, symbol: str, limit: int = 8) -> list[NewsItem]:
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={quote(symbol)}&region=US&lang=en-US"
        async with httpx.AsyncClient(timeout=20.0) as client:
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
            if title:
                items.append(NewsItem(title=title, url=link, published=published))
        return items
