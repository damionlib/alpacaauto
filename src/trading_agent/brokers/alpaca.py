from __future__ import annotations

import math
from datetime import UTC, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from trading_agent.config import Settings
from trading_agent.day_trading.features import (
    EASTERN,
    cumulative_volume_profile,
    intraday_features,
    regular_session_bars,
)
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
        self._volume_profiles: dict[tuple[str, Any], list[float]] = {}

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
            # Alpaca returns 204 No Content for DELETE (cancel order); parsing
            # the empty body as JSON would raise even though the call succeeded.
            if response.status_code == 204 or not response.content:
                return None
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
        if intent.stop_price is not None:
            payload["stop_price"] = round(intent.stop_price, 2)
        if intent.client_order_id:
            payload["client_order_id"] = intent.client_order_id

        if intent.order_class == "oco":
            payload["order_class"] = "oco"
            if intent.take_profit_price is not None:
                payload["take_profit"] = {"limit_price": round(intent.take_profit_price, 2)}
            if intent.stop_loss_price is not None:
                payload["stop_loss"] = {"stop_price": round(intent.stop_loss_price, 2)}
        elif intent.asset_class in {AssetClass.EQUITY, AssetClass.ETF}:
            if intent.stop_loss_price and intent.take_profit_price:
                payload["order_class"] = "bracket"
                payload["stop_loss"] = {"stop_price": round(intent.stop_loss_price, 2)}
                payload["take_profit"] = {"limit_price": round(intent.take_profit_price, 2)}

        return await self._request("POST", f"{self.trading_base_url}/v2/orders", json=payload)

    async def get_clock(self) -> dict[str, Any]:
        return await self._request("GET", f"{self.trading_base_url}/v2/clock")

    async def cancel_all_orders(self) -> list[dict]:
        return await self._request("DELETE", f"{self.trading_base_url}/v2/orders")

    async def close_all_positions(self, *, cancel_orders: bool = True) -> list[dict]:
        return await self._request(
            "DELETE",
            f"{self.trading_base_url}/v2/positions",
            params={"cancel_orders": str(cancel_orders).lower()},
        )

    async def close_position(self, symbol: str) -> dict:
        return await self._request("DELETE", f"{self.trading_base_url}/v2/positions/{symbol}")

    async def replace_order(
        self,
        order_id: str,
        *,
        qty: float | None = None,
        stop_price: float | None = None,
        limit_price: float | None = None,
    ) -> dict:
        payload: dict[str, Any] = {}
        if qty is not None:
            payload["qty"] = self._format_qty(qty)
        if stop_price is not None:
            payload["stop_price"] = round(stop_price, 2)
        if limit_price is not None:
            payload["limit_price"] = round(limit_price, 2)
        return await self._request("PATCH", f"{self.trading_base_url}/v2/orders/{order_id}", json=payload)

    async def cancel_order(self, order_id: str) -> None:
        await self._request("DELETE", f"{self.trading_base_url}/v2/orders/{order_id}")

    async def get_open_orders(self) -> list[dict]:
        return await self._request(
            "GET",
            f"{self.trading_base_url}/v2/orders",
            params={"status": "open", "limit": 100, "direction": "desc"},
        )

    async def get_recent_orders(
        self,
        *,
        status: str = "closed",
        limit: int = 100,
        after: str | None = None,
        until: str | None = None,
    ) -> list[dict]:
        params: dict[str, Any] = {
            "status": status,
            "limit": limit,
            "direction": "desc",
        }
        if after:
            params["after"] = after
        if until:
            params["until"] = until
        return await self._request("GET", f"{self.trading_base_url}/v2/orders", params=params)

    async def _get_equity_snapshot(self, symbol: str) -> MarketSnapshot:
        latest = await self._request(
            "GET",
            f"{DATA_API_BASE}/v2/stocks/{symbol}/trades/latest",
            params={"feed": "iex"},
        )
        price = float(latest["trade"]["p"])
        series = (await self.get_daily_bars([symbol])).get(symbol) or {}
        return MarketSnapshot(
            symbol=symbol,
            asset_class=AssetClass.EQUITY,
            price=price,
            closes=series.get("closes", []),
            metadata={
                "volumes": series.get("volumes", []),
                "highs": series.get("highs", []),
                "lows": series.get("lows", []),
            },
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

    async def get_intraday_features(
        self,
        symbol: str,
        daily_volumes: list[float] | None = None,  # unused; kept for call compatibility
    ) -> dict[str, Any]:
        """Intraday inputs the day-trade engine needs: bid/ask, VWAP, short-term
        trend, opening-range break and relative volume. Each piece is
        best-effort; a failure leaves the field absent (treated as neutral)."""
        features: dict[str, Any] = {}
        try:
            data = await self._request(
                "GET",
                f"{DATA_API_BASE}/v2/stocks/{symbol}/quotes/latest",
                params={"feed": "iex"},
            )
            quote = data.get("quote", {}) or {}
            bid = self._coerce_float(quote.get("bp"))
            ask = self._coerce_float(quote.get("ap"))
            if bid and bid > 0:
                features["bid"] = bid
            if ask and ask > 0:
                features["ask"] = ask
        except Exception:
            pass
        try:
            now = datetime.now(UTC)
            today = now.astimezone(EASTERN).date()
            bars = await self._get_intraday_minute_bars(symbol)
            profile = await self._intraday_volume_profile(symbol, today)
            features.update(
                intraday_features(
                    regular_session_bars(bars, today),
                    now=now,
                    volume_profile=profile,
                    orb_valid_minutes=self.settings.day_trading.orb_valid_minutes,
                )
            )
        except Exception:
            pass
        return features

    async def _get_intraday_minute_bars(self, symbol: str) -> list[dict[str, Any]]:
        # Real-time intraday bars must come from IEX on the free data plan (SIP
        # is only served 15+ minutes late), so relative volume is computed
        # IEX-vs-IEX history below to keep the venue consistent.
        start = (datetime.now(UTC) - timedelta(hours=12)).isoformat()
        return await self._paged_minute_bars(symbol, start=start)

    async def _intraday_volume_profile(self, symbol: str, today) -> list[float]:
        key = (symbol, today)
        cached = self._volume_profiles.get(key)
        if cached is not None:
            return cached
        start = (datetime.now(UTC) - timedelta(days=16)).isoformat()
        history = await self._paged_minute_bars(symbol, start=start)
        profile = cumulative_volume_profile(history, before_day=today, days=10)
        self._volume_profiles = {k: v for k, v in self._volume_profiles.items() if k[1] == today}
        self._volume_profiles[key] = profile
        return profile

    async def _paged_minute_bars(self, symbol: str, *, start: str) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "symbols": symbol,
            "timeframe": "1Min",
            "start": start,
            "feed": "iex",
            "limit": 10000,
        }
        rows: list[dict[str, Any]] = []
        for _page in range(20):
            data = await self._request("GET", f"{DATA_API_BASE}/v2/stocks/bars", params=params)
            rows.extend((data.get("bars") or {}).get(symbol, []) or [])
            token = data.get("next_page_token")
            if not token:
                break
            params["page_token"] = token
        return rows

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        try:
            return float(value) if value not in (None, "") else None
        except (TypeError, ValueError):
            return None

    async def get_daily_adr(self, symbol: str, days: int = 20) -> float | None:
        series = (await self.get_daily_bars([symbol], lookback_days=days * 3)).get(symbol)
        if not series or not series["closes"]:
            return None
        ranges = [
            ((high - low) / open_) * 100
            for open_, high, low in zip(
                series["opens"][-days:], series["highs"][-days:], series["lows"][-days:], strict=False
            )
            if open_ > 0
        ]
        return sum(ranges) / len(ranges) if ranges else None

    async def get_most_actives(self, top: int = 20) -> list[dict[str, Any]]:
        data = await self._request(
            "GET",
            f"{DATA_API_BASE}/v1beta1/screener/stocks/most-actives",
            params={"by": "volume", "top": top},
        )
        return data.get("most_actives", [])

    async def get_movers(self, top: int = 10) -> dict[str, list[dict[str, Any]]]:
        data = await self._request(
            "GET",
            f"{DATA_API_BASE}/v1beta1/screener/stocks/movers",
            params={"top": top},
        )
        return {
            "gainers": data.get("gainers", []),
            "losers": data.get("losers", []),
        }

    async def get_snapshots_batch(
        self, symbols: list[str], *, feed: str = "iex",
    ) -> dict[str, dict[str, Any]]:
        data = await self._request(
            "GET",
            f"{DATA_API_BASE}/v2/stocks/snapshots",
            params={"symbols": ",".join(symbols), "feed": feed},
        )
        return data if isinstance(data, dict) else {}

    async def get_daily_bars(
        self, symbols: list[str], *, lookback_days: int | None = None,
    ) -> dict[str, dict[str, list[float]]]:
        """Daily OHLCV per symbol, following pagination across all symbols.

        A multi-symbol request is paginated by total bar count, not per symbol;
        without following next_page_token only the first ~1000 bars (the first
        dozen or so symbols alphabetically) come back and the rest silently vanish.
        """
        cfg = self.settings.market_data
        now = datetime.now(UTC)
        start = (now - timedelta(days=lookback_days or cfg.daily_lookback_days)).isoformat()
        params: dict[str, Any] = {
            "symbols": ",".join(symbols),
            "timeframe": "1Day",
            "start": start,
            "feed": cfg.historical_feed,
            "adjustment": cfg.adjustment,
            "limit": 10000,
        }
        if cfg.historical_feed == "sip":
            params["end"] = (now - timedelta(minutes=cfg.sip_delay_minutes)).isoformat()
        rows_by_symbol: dict[str, list[dict]] = {}
        for _page in range(100):
            data = await self._request("GET", f"{DATA_API_BASE}/v2/stocks/bars", params=params)
            for symbol, rows in (data.get("bars") or {}).items():
                rows_by_symbol.setdefault(symbol, []).extend(rows or [])
            token = data.get("next_page_token")
            if not token:
                break
            params["page_token"] = token
        return {symbol: self._bars_to_series(rows) for symbol, rows in rows_by_symbol.items()}

    async def _get_stock_bars(self, symbols: list[str]) -> dict[str, tuple[list[float], list[float]]]:
        series = await self.get_daily_bars(symbols)
        return {symbol: (bars["closes"], bars["volumes"]) for symbol, bars in series.items()}

    @staticmethod
    def _bars_to_series(rows: list[dict]) -> dict[str, list[float]]:
        return {
            "opens": [float(bar["o"]) for bar in rows],
            "highs": [float(bar["h"]) for bar in rows],
            "lows": [float(bar["l"]) for bar in rows],
            "closes": [float(bar["c"]) for bar in rows],
            "volumes": [float(bar.get("v") or 0) for bar in rows],
        }

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
