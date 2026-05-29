from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from trading_agent.config import Settings
from trading_agent.models import (
    AccountSnapshot,
    AssetClass,
    MarketSnapshot,
    OrderIntent,
    Position,
)


TRADING_API_BASE = {
    "paper": "https://paper-api.alpaca.markets",
    "live": "https://api.alpaca.markets",
}
DATA_API_BASE = "https://data.alpaca.markets"


class AlpacaBroker:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.require_live_confirmation()
        if not settings.alpaca_api_key_id or not settings.alpaca_api_secret_key:
            raise RuntimeError("Missing ALPACA_API_KEY_ID or ALPACA_API_SECRET_KEY.")
        self.trading_base_url = TRADING_API_BASE[settings.broker.mode]
        self.headers = {
            "APCA-API-KEY-ID": settings.alpaca_api_key_id.get_secret_value(),
            "APCA-API-SECRET-KEY": settings.alpaca_api_secret_key.get_secret_value(),
        }

    async def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> Any:
        async with httpx.AsyncClient(timeout=30.0, headers=self.headers) as client:
            response = await client.request(method, url, params=params, json=json)
            if response.is_error:
                details = self._error_details(response)
                raise RuntimeError(f"{method} {url} failed with {response.status_code}: {details}")
            response.raise_for_status()
            return response.json()

    def _error_details(self, response: httpx.Response) -> str:
        try:
            data = response.json()
        except ValueError:
            return response.text[:500]
        if isinstance(data, dict):
            message = data.get("message") or data.get("error") or data
            code = data.get("code")
            return f"{code}: {message}" if code else str(message)
        return str(data)

    @retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(3))
    async def get_account(self) -> AccountSnapshot:
        data = await self._request("GET", f"{self.trading_base_url}/v2/account")
        return AccountSnapshot(
            equity=float(data["equity"]),
            cash=float(data["cash"]),
            buying_power=float(data["buying_power"]),
            last_equity=float(data["last_equity"]) if data.get("last_equity") else None,
        )

    @retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(3))
    async def get_positions(self) -> list[Position]:
        rows = await self._request("GET", f"{self.trading_base_url}/v2/positions")
        positions: list[Position] = []
        for row in rows:
            positions.append(
                Position(
                    symbol=row["symbol"],
                    asset_class=self._asset_class_from_alpaca(row.get("asset_class"), row["symbol"]),
                    qty=float(row["qty"]),
                    market_value=float(row["market_value"]),
                    avg_entry_price=float(row["avg_entry_price"])
                    if row.get("avg_entry_price")
                    else None,
                    current_price=float(row["current_price"])
                    if row.get("current_price")
                    else None,
                    unrealized_pl=float(row["unrealized_pl"])
                    if row.get("unrealized_pl")
                    else None,
                )
            )
        return positions

    async def get_market_snapshot(self, symbol: str) -> MarketSnapshot:
        if "/" in symbol:
            return await self._get_crypto_snapshot(symbol)
        if self._looks_like_option_symbol(symbol):
            return await self._get_option_snapshot(symbol)
        return await self._get_equity_snapshot(symbol)

    async def get_option_contracts(
        self,
        underlying_symbol: str,
        *,
        option_type: str | None = None,
        expiration_after_days: int = 14,
        expiration_before_days: int = 60,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        today = datetime.now(UTC).date()
        params: dict[str, Any] = {
            "underlying_symbols": underlying_symbol,
            "expiration_date_gte": str(today + timedelta(days=expiration_after_days)),
            "expiration_date_lte": str(today + timedelta(days=expiration_before_days)),
            "status": "active",
            "limit": limit,
        }
        if option_type:
            params["type"] = option_type
        data = await self._request(
            "GET",
            f"{self.trading_base_url}/v2/options/contracts",
            params=params,
        )
        return data.get("option_contracts", [])

    async def submit_order(self, intent: OrderIntent) -> dict:
        if intent.order_class == "mleg":
            payload: dict[str, Any] = {
                "order_class": "mleg",
                "qty": self._format_qty(intent.qty or 1),
                "type": intent.order_type.value,
                "time_in_force": intent.time_in_force.value,
                "legs": intent.legs,
            }
        else:
            payload = {
                "symbol": intent.symbol,
                "side": intent.side.value,
                "type": intent.order_type.value,
                "time_in_force": intent.time_in_force.value,
            }
        if intent.qty is not None:
            payload["qty"] = self._format_qty(intent.qty)
        if intent.notional is not None:
            payload["notional"] = round(intent.notional, 2)
        if intent.limit_price is not None:
            payload["limit_price"] = round(intent.limit_price, 2)
        if intent.client_order_id:
            payload["client_order_id"] = intent.client_order_id

        if intent.asset_class in {AssetClass.EQUITY, AssetClass.ETF}:
            if intent.stop_loss_price and intent.take_profit_price:
                payload["order_class"] = "bracket"
                payload["stop_loss"] = {"stop_price": round(intent.stop_loss_price, 2)}
                payload["take_profit"] = {"limit_price": round(intent.take_profit_price, 2)}

        return await self._request("POST", f"{self.trading_base_url}/v2/orders", json=payload)

    async def cancel_all_orders(self) -> list[dict]:
        return await self._request("DELETE", f"{self.trading_base_url}/v2/orders")

    async def get_open_orders(self) -> list[dict]:
        return await self._request(
            "GET",
            f"{self.trading_base_url}/v2/orders",
            params={"status": "open", "limit": 100, "direction": "desc"},
        )

    async def _get_equity_snapshot(self, symbol: str) -> MarketSnapshot:
        latest = await self._request(
            "GET",
            f"{DATA_API_BASE}/v2/stocks/{symbol}/trades/latest",
            params={"feed": "iex"},
        )
        price = float(latest["trade"]["p"])
        bars = await self._get_stock_bars([symbol])
        closes, volumes = bars.get(symbol, ([], []))
        return MarketSnapshot(
            symbol=symbol,
            asset_class=AssetClass.EQUITY,
            price=price,
            closes=closes,
            metadata={"volumes": volumes},
        )

    async def _get_crypto_snapshot(self, symbol: str) -> MarketSnapshot:
        latest = await self._request(
            "GET",
            f"{DATA_API_BASE}/v1beta3/crypto/us/latest/trades",
            params={"symbols": symbol},
        )
        price = float(latest["trades"][symbol]["p"])
        bars = await self._get_crypto_bars([symbol])
        closes, volumes = bars.get(symbol, ([], []))
        return MarketSnapshot(
            symbol=symbol,
            asset_class=AssetClass.CRYPTO,
            price=price,
            closes=closes,
            metadata={"volumes": volumes},
        )

    async def _get_option_snapshot(self, symbol: str) -> MarketSnapshot:
        latest_trade = await self._request(
            "GET",
            f"{DATA_API_BASE}/v1beta1/options/trades/latest",
            params={"symbols": symbol},
        )
        trade = latest_trade.get("trades", {}).get(symbol)
        if trade:
            price = float(trade["p"])
        else:
            latest_quote = await self.get_option_latest_quote(symbol)
            price = self.option_quote_midpoint(latest_quote)
            if price is None:
                raise RuntimeError(f"No latest option trade or usable quote returned for {symbol}.")
        return MarketSnapshot(
            symbol=symbol,
            asset_class=AssetClass.OPTION,
            price=price,
            closes=[],
        )

    async def get_option_latest_quote(self, symbol: str) -> dict[str, Any] | None:
        data = await self._request(
            "GET",
            f"{DATA_API_BASE}/v1beta1/options/quotes/latest",
            params={"symbols": symbol},
        )
        return data.get("quotes", {}).get(symbol)

    def option_quote_bid_ask(self, quote: dict[str, Any] | None) -> tuple[float | None, float | None]:
        if not quote:
            return None, None
        bid = quote.get("bp", quote.get("bid_price"))
        ask = quote.get("ap", quote.get("ask_price"))
        return (
            float(bid) if bid not in {None, ""} else None,
            float(ask) if ask not in {None, ""} else None,
        )

    def option_quote_midpoint(self, quote: dict[str, Any] | None) -> float | None:
        bid, ask = self.option_quote_bid_ask(quote)
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            return round((bid + ask) / 2, 2)
        if ask is not None and ask > 0:
            return float(ask)
        if bid is not None and bid > 0:
            return float(bid)
        return None

    async def _get_stock_bars(self, symbols: list[str]) -> dict[str, tuple[list[float], list[float]]]:
        start = (datetime.now(UTC) - timedelta(days=100)).isoformat()
        data = await self._request(
            "GET",
            f"{DATA_API_BASE}/v2/stocks/bars",
            params={
                "symbols": ",".join(symbols),
                "timeframe": "1Day",
                "start": start,
                "feed": "iex",
                "limit": 1000,
            },
        )
        return self._bars_to_closes_and_volumes(data.get("bars", {}))

    async def _get_crypto_bars(self, symbols: list[str]) -> dict[str, tuple[list[float], list[float]]]:
        start = (datetime.now(UTC) - timedelta(days=100)).isoformat()
        data = await self._request(
            "GET",
            f"{DATA_API_BASE}/v1beta3/crypto/us/bars",
            params={
                "symbols": ",".join(symbols),
                "timeframe": "1Day",
                "start": start,
                "limit": 1000,
            },
        )
        return self._bars_to_closes_and_volumes(data.get("bars", {}))

    def _bars_to_closes_and_volumes(self, rows_by_symbol: dict) -> dict[str, tuple[list[float], list[float]]]:
        parsed: dict[str, tuple[list[float], list[float]]] = {}
        for symbol, bars in rows_by_symbol.items():
            closes = [float(bar["c"]) for bar in bars]
            volumes = [float(bar.get("v") or 0) for bar in bars]
            parsed[symbol] = (closes, volumes)
        return parsed

    def _asset_class_from_alpaca(self, value: str | None, symbol: str = "") -> AssetClass:
        if self._looks_like_option_symbol(symbol):
            return AssetClass.OPTION
        if value == "crypto":
            return AssetClass.CRYPTO
        if value == "option":
            return AssetClass.OPTION
        return AssetClass.EQUITY

    def _looks_like_option_symbol(self, symbol: str) -> bool:
        return len(symbol) >= 15 and symbol[-15:-9].isdigit()

    def _format_qty(self, qty: float) -> str:
        if math.isclose(qty, round(qty)):
            return str(int(round(qty)))
        return f"{qty:.8f}".rstrip("0").rstrip(".")
