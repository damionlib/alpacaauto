from __future__ import annotations

from datetime import datetime
from typing import Any

from trading_agent.config import Settings
from trading_agent.models import AssetClass, MarketSnapshot, OrderSide, TradeCandidate


class OptionsStrategy:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def covered_call_candidates(
        self,
        underlying: MarketSnapshot,
        contracts: list[dict[str, Any]],
        owned_shares: float,
    ) -> list[TradeCandidate]:
        if "covered_call" not in self.settings.strategy.enabled or owned_shares < 100:
            return []
        calls = self._rank_contracts(contracts, option_type="call", min_strike=underlying.price * 1.03)
        if not calls:
            return []
        contract = calls[0]
        return [
            TradeCandidate(
                symbol=contract["symbol"],
                asset_class=AssetClass.OPTION,
                side=OrderSide.SELL,
                strategy="covered_call",
                score=72,
                entry_price=0.0,
                rationale=[
                    f"Underlying position has at least 100 shares of {underlying.symbol}.",
                    f"Selected out-of-the-money call strike {contract.get('strike_price')} expiring {contract.get('expiration_date')}.",
                ],
                metadata={"underlying": underlying.symbol, "contract": contract, "contracts_per_100_shares": int(owned_shares // 100)},
            )
        ]

    def cash_secured_put_candidates(
        self,
        underlying: MarketSnapshot,
        contracts: list[dict[str, Any]],
    ) -> list[TradeCandidate]:
        if "cash_secured_put" not in self.settings.strategy.enabled:
            return []
        puts = self._rank_contracts(contracts, option_type="put", max_strike=underlying.price * 0.95)
        if not puts:
            return []
        contract = puts[0]
        return [
            TradeCandidate(
                symbol=contract["symbol"],
                asset_class=AssetClass.OPTION,
                side=OrderSide.SELL,
                strategy="cash_secured_put",
                score=70,
                entry_price=0.0,
                rationale=[
                    f"Selected out-of-the-money put strike {contract.get('strike_price')} expiring {contract.get('expiration_date')}.",
                    "Risk engine will require cash coverage before execution.",
                ],
                metadata={"underlying": underlying.symbol, "contract": contract},
            )
        ]

    def long_option_candidates(
        self,
        underlying: MarketSnapshot,
        contracts: list[dict[str, Any]],
        bullish_score: float,
    ) -> list[TradeCandidate]:
        candidates: list[TradeCandidate] = []
        if bullish_score >= self.settings.strategy.min_signal_score and "long_call" in self.settings.strategy.enabled:
            calls = self._rank_contracts(
                contracts,
                option_type="call",
                min_strike=underlying.price * 0.98,
                max_strike=underlying.price * 1.05,
            )
            if calls:
                candidates.append(self._long_option_candidate(underlying, calls[0], "long_call"))
        if bullish_score < 35 and "long_put" in self.settings.strategy.enabled:
            puts = self._rank_contracts(
                contracts,
                option_type="put",
                min_strike=underlying.price * 0.95,
                max_strike=underlying.price * 1.02,
            )
            if puts:
                candidates.append(self._long_option_candidate(underlying, puts[0], "long_put"))
        return candidates

    def debit_spread_candidates(
        self,
        underlying: MarketSnapshot,
        contracts: list[dict[str, Any]],
        bullish_score: float,
    ) -> list[TradeCandidate]:
        candidates: list[TradeCandidate] = []
        if bullish_score >= self.settings.strategy.min_signal_score and "call_debit_spread" in self.settings.strategy.enabled:
            spread = self._vertical_spread(
                underlying,
                contracts,
                option_type="call",
                buy_min=underlying.price * 0.98,
                sell_min=underlying.price * 1.03,
            )
            if spread:
                buy_leg, sell_leg = spread
                candidates.append(self._spread_candidate(underlying, "call_debit_spread", buy_leg, sell_leg))
        if bullish_score < 35 and "put_debit_spread" in self.settings.strategy.enabled:
            spread = self._vertical_spread(
                underlying,
                contracts,
                option_type="put",
                buy_min=underlying.price * 0.97,
                sell_max=underlying.price * 0.92,
            )
            if spread:
                buy_leg, sell_leg = spread
                candidates.append(self._spread_candidate(underlying, "put_debit_spread", buy_leg, sell_leg))
        return candidates

    def _long_option_candidate(
        self,
        underlying: MarketSnapshot,
        contract: dict[str, Any],
        strategy: str,
    ) -> TradeCandidate:
        return TradeCandidate(
            symbol=contract["symbol"],
            asset_class=AssetClass.OPTION,
            side=OrderSide.BUY,
            strategy=strategy,
            score=71,
            entry_price=0.0,
            rationale=[
                f"Selected {contract.get('type')} near underlying price {underlying.price:.2f}.",
                "Risk is limited to premium paid when buying a single-leg option.",
            ],
            metadata={"underlying": underlying.symbol, "contract": contract},
        )

    def _rank_contracts(
        self,
        contracts: list[dict[str, Any]],
        *,
        option_type: str,
        min_strike: float | None = None,
        max_strike: float | None = None,
    ) -> list[dict[str, Any]]:
        filtered = []
        for contract in contracts:
            if contract.get("type") != option_type:
                continue
            strike = float(contract["strike_price"])
            if min_strike is not None and strike < min_strike:
                continue
            if max_strike is not None and strike > max_strike:
                continue
            filtered.append(contract)

        def sort_key(contract: dict[str, Any]) -> tuple[str, float]:
            expiry = contract.get("expiration_date") or "9999-12-31"
            try:
                expiry_key = datetime.fromisoformat(expiry).date().isoformat()
            except ValueError:
                expiry_key = expiry
            return expiry_key, abs(float(contract["strike_price"]))

        return sorted(filtered, key=sort_key)

    def _vertical_spread(
        self,
        underlying: MarketSnapshot,
        contracts: list[dict[str, Any]],
        *,
        option_type: str,
        buy_min: float,
        sell_min: float | None = None,
        sell_max: float | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        buys = self._rank_contracts(contracts, option_type=option_type, min_strike=buy_min)
        for buy_leg in buys:
            expiry = buy_leg.get("expiration_date")
            buy_strike = float(buy_leg["strike_price"])
            for sell_leg in contracts:
                if sell_leg.get("type") != option_type or sell_leg.get("expiration_date") != expiry:
                    continue
                sell_strike = float(sell_leg["strike_price"])
                if option_type == "call" and sell_strike <= buy_strike:
                    continue
                if option_type == "put" and sell_strike >= buy_strike:
                    continue
                if sell_min is not None and sell_strike < sell_min:
                    continue
                if sell_max is not None and sell_strike > sell_max:
                    continue
                return buy_leg, sell_leg
        return None

    def _spread_candidate(
        self,
        underlying: MarketSnapshot,
        strategy: str,
        buy_leg: dict[str, Any],
        sell_leg: dict[str, Any],
    ) -> TradeCandidate:
        return TradeCandidate(
            symbol=f"{underlying.symbol}_{strategy}".replace("/", "_"),
            asset_class=AssetClass.OPTION,
            side=OrderSide.BUY,
            strategy=strategy,
            score=72,
            entry_price=0.0,
            rationale=[
                f"Selected same-expiration spread for {underlying.symbol}.",
                "Risk is capped to net debit paid plus fees if filled as intended.",
            ],
            metadata={
                "underlying": underlying.symbol,
                "legs": [
                    {
                        "symbol": buy_leg["symbol"],
                        "ratio_qty": "1",
                        "side": "buy",
                        "position_intent": "buy_to_open",
                    },
                    {
                        "symbol": sell_leg["symbol"],
                        "ratio_qty": "1",
                        "side": "sell",
                        "position_intent": "sell_to_open",
                    },
                ],
                "contracts": [buy_leg, sell_leg],
            },
        )
