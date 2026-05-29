from __future__ import annotations

import asyncio
from typing import Any

import httpx

from trading_agent.config import Settings


SYMBOL_TO_COINGECKO_ID = {
    "BTC/USD": "bitcoin",
    "ETH/USD": "ethereum",
}

SYMBOL_TO_BINANCE_PERP = {
    "BTC/USD": "BTCUSDT",
    "ETH/USD": "ETHUSDT",
}


class CryptoResearchService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def research(self, symbol: str) -> tuple[dict[str, Any], list[str]]:
        if not self.settings.research.crypto_research_enabled:
            return {}, ["Crypto research disabled."]

        notes: list[str] = []
        tasks = {
            "global": self._global_market(),
            "asset": self._asset_market(symbol),
            "fear_greed": self._fear_greed(),
            "stablecoins": self._stablecoin_liquidity(),
            "funding": self._funding_rate(symbol),
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        sections = dict(zip(tasks.keys(), results, strict=False))

        summary: dict[str, Any] = {}
        for key, result in sections.items():
            if isinstance(result, Exception):
                notes.append(f"Crypto {key} lookup failed: {result}")
                summary[key] = {}
            else:
                summary[key] = result

        summary["regime"] = self._regime(summary)
        summary["risk_flags"] = self._risk_flags(summary)
        summary["exchange_flows"] = self._exchange_flows_stub()
        summary["onchain"] = self._onchain_stub()
        return summary, notes

    async def _global_market(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get("https://api.coingecko.com/api/v3/global")
            response.raise_for_status()
            data = response.json().get("data", {})

        market_cap_change = data.get("market_cap_change_percentage_24h_usd")
        return {
            "total_market_cap_usd": data.get("total_market_cap", {}).get("usd"),
            "total_volume_usd": data.get("total_volume", {}).get("usd"),
            "btc_dominance_pct": data.get("market_cap_percentage", {}).get("btc"),
            "eth_dominance_pct": data.get("market_cap_percentage", {}).get("eth"),
            "market_cap_change_24h_pct": market_cap_change,
            "active_cryptocurrencies": data.get("active_cryptocurrencies"),
            "markets": data.get("markets"),
        }

    async def _asset_market(self, symbol: str) -> dict[str, Any]:
        coin_id = SYMBOL_TO_COINGECKO_ID.get(symbol)
        if not coin_id:
            return {"status": "unsupported_symbol"}
        params = {
            "vs_currency": "usd",
            "ids": coin_id,
            "price_change_percentage": "24h,7d,30d",
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get("https://api.coingecko.com/api/v3/coins/markets", params=params)
            response.raise_for_status()
            rows = response.json()
        if not rows:
            return {}
        row = rows[0]
        return {
            "coin_id": coin_id,
            "market_cap_rank": row.get("market_cap_rank"),
            "market_cap_usd": row.get("market_cap"),
            "total_volume_usd": row.get("total_volume"),
            "price_change_24h_pct": row.get("price_change_percentage_24h_in_currency"),
            "price_change_7d_pct": row.get("price_change_percentage_7d_in_currency"),
            "price_change_30d_pct": row.get("price_change_percentage_30d_in_currency"),
        }

    async def _fear_greed(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get("https://api.alternative.me/fng/", params={"limit": 1})
            response.raise_for_status()
            rows = response.json().get("data", [])
        if not rows:
            return {}
        row = rows[0]
        return {
            "value": int(row["value"]) if str(row.get("value", "")).isdigit() else row.get("value"),
            "classification": row.get("value_classification"),
            "timestamp": row.get("timestamp"),
        }

    async def _stablecoin_liquidity(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get("https://stablecoins.llama.fi/stablecoins", params={"includePrices": "true"})
            response.raise_for_status()
            data = response.json()

        assets = data.get("peggedAssets", [])
        total_circulating_usd = 0.0
        top_assets: list[dict[str, Any]] = []
        for asset in assets:
            circulating = asset.get("circulating") or {}
            pegged_usd = circulating.get("peggedUSD") or 0
            try:
                pegged_usd_float = float(pegged_usd)
            except (TypeError, ValueError):
                pegged_usd_float = 0.0
            total_circulating_usd += pegged_usd_float
            top_assets.append(
                {
                    "symbol": asset.get("symbol"),
                    "name": asset.get("name"),
                    "circulating_usd": pegged_usd_float,
                }
            )
        top_assets.sort(key=lambda row: row["circulating_usd"], reverse=True)
        return {
            "total_circulating_usd": total_circulating_usd,
            "asset_count": len(assets),
            "top_assets": top_assets[:5],
        }

    async def _funding_rate(self, symbol: str) -> dict[str, Any]:
        perp_symbol = SYMBOL_TO_BINANCE_PERP.get(symbol)
        if not perp_symbol:
            return {"status": "unsupported_symbol"}
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(
                "https://fapi.binance.com/fapi/v1/premiumIndex",
                params={"symbol": perp_symbol},
            )
            if response.status_code == 451:
                return {
                    "status": "unavailable",
                    "venue": "binance_usdt_perp",
                    "symbol": perp_symbol,
                    "reason": "venue_restricted_or_unavailable_from_current_location",
                }
            response.raise_for_status()
            data = response.json()
        funding_rate = data.get("lastFundingRate")
        return {
            "venue": "binance_usdt_perp",
            "symbol": perp_symbol,
            "last_funding_rate": float(funding_rate) if funding_rate is not None else None,
            "next_funding_time": data.get("nextFundingTime"),
            "mark_price": float(data["markPrice"]) if data.get("markPrice") else None,
            "index_price": float(data["indexPrice"]) if data.get("indexPrice") else None,
        }

    def _regime(self, summary: dict[str, Any]) -> dict[str, Any]:
        global_market = summary.get("global") or {}
        asset = summary.get("asset") or {}
        fear_greed = summary.get("fear_greed") or {}
        funding = summary.get("funding") or {}

        market_change = float(global_market.get("market_cap_change_24h_pct") or 0)
        asset_change_7d = float(asset.get("price_change_7d_pct") or 0)
        btc_dominance = float(global_market.get("btc_dominance_pct") or 0)
        fear_value = fear_greed.get("value")
        fear_value = float(fear_value) if isinstance(fear_value, int | float) else None
        funding_rate = funding.get("last_funding_rate")
        funding_rate = float(funding_rate) if funding_rate is not None else 0.0

        score = 50.0
        score += max(min(market_change * 2, 15), -15)
        score += max(min(asset_change_7d, 15), -15)
        if fear_value is not None:
            if fear_value < 20:
                score -= 10
            elif fear_value > 80:
                score -= 8
            elif 35 <= fear_value <= 70:
                score += 5
        if funding_rate > 0.0005:
            score -= 8
        elif funding_rate < -0.0002:
            score += 3
        if btc_dominance > 55:
            score -= 3

        score = round(max(0.0, min(100.0, score)), 2)
        if score >= 65:
            label = "risk_on"
        elif score <= 40:
            label = "risk_off"
        else:
            label = "neutral"
        return {
            "label": label,
            "score": score,
            "inputs": {
                "market_cap_change_24h_pct": market_change,
                "asset_change_7d_pct": asset_change_7d,
                "btc_dominance_pct": btc_dominance,
                "fear_greed_value": fear_value,
                "funding_rate": funding_rate,
            },
        }

    def _risk_flags(self, summary: dict[str, Any]) -> list[str]:
        flags: list[str] = []
        fear_greed = summary.get("fear_greed") or {}
        funding = summary.get("funding") or {}
        stablecoins = summary.get("stablecoins") or {}
        regime = summary.get("regime") or {}

        value = fear_greed.get("value")
        if isinstance(value, int | float):
            if value >= 80:
                flags.append("extreme_greed")
            elif value <= 20:
                flags.append("extreme_fear")
        funding_rate = funding.get("last_funding_rate")
        if funding_rate is not None and funding_rate > 0.0005:
            flags.append("perp_funding_crowded_long")
        if regime.get("label") == "risk_off":
            flags.append("crypto_market_risk_off")
        if not stablecoins.get("total_circulating_usd"):
            flags.append("stablecoin_liquidity_unavailable")
        return flags

    def _exchange_flows_stub(self) -> dict[str, Any]:
        if not self.settings.research.crypto_exchange_flows_enabled:
            return {
                "status": "not_configured",
                "note": "Exchange-flow data usually requires a paid/on-chain provider; disabled in config.",
            }
        return {
            "status": "provider_required",
            "note": "Configure a supported exchange-flow provider before using this signal live.",
        }

    def _onchain_stub(self) -> dict[str, Any]:
        if not self.settings.research.crypto_onchain_enabled:
            return {
                "status": "not_configured",
                "note": "On-chain data usually requires a provider key; disabled in config.",
            }
        return {
            "status": "provider_required",
            "provider": self.settings.research.crypto_onchain_provider,
            "note": "Provider integration is not configured yet.",
        }
