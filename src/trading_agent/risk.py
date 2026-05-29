from __future__ import annotations

import uuid

from trading_agent.config import Settings
from trading_agent.models import (
    AccountSnapshot,
    AssetClass,
    OrderIntent,
    OrderSide,
    OrderType,
    Position,
    RiskDecision,
    TimeInForce,
    TradeCandidate,
)


class RiskEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def evaluate(
        self,
        candidate: TradeCandidate,
        account: AccountSnapshot,
        positions: list[Position],
    ) -> RiskDecision:
        if candidate.metadata.get("exit"):
            return self._evaluate_exit(candidate, positions)

        if account.daily_pl_pct <= -self.settings.risk.max_daily_loss_pct:
            return self._reject(candidate, f"Daily loss stop reached: {account.daily_pl_pct:.2f}%.")

        cash_buffer = account.equity * (self.settings.risk.min_cash_buffer_pct / 100)
        spendable_balance = min(account.cash, account.buying_power)
        available_cash = max(spendable_balance - cash_buffer, 0)
        if candidate.side == OrderSide.BUY and available_cash <= 0:
            return self._reject(candidate, "Cash/buying-power buffer would be breached.")

        if candidate.asset_class in {AssetClass.EQUITY, AssetClass.ETF, AssetClass.CRYPTO}:
            return self._evaluate_spot(candidate, account, positions, available_cash)
        if candidate.asset_class == AssetClass.OPTION:
            return self._evaluate_option(candidate, account, positions, available_cash)
        return self._reject(candidate, f"Unsupported asset class: {candidate.asset_class}.")

    def _evaluate_exit(
        self,
        candidate: TradeCandidate,
        positions: list[Position],
    ) -> RiskDecision:
        position = self._position(candidate.symbol, positions)
        if not position:
            return self._reject(candidate, "No existing position found to close.")
        requested_qty = float(candidate.metadata.get("exit_qty") or abs(position.qty))
        close_qty = min(abs(position.qty), requested_qty)
        if close_qty <= 0:
            return self._reject(candidate, "Calculated close quantity is zero.")

        if position.qty > 0 and candidate.side != OrderSide.SELL:
            return self._reject(candidate, "Long positions must be closed with a sell order.")
        if position.qty < 0 and candidate.side != OrderSide.BUY:
            return self._reject(candidate, "Short positions must be closed with a buy order.")

        qty = close_qty
        if candidate.asset_class in {AssetClass.EQUITY, AssetClass.ETF, AssetClass.OPTION}:
            qty = int(close_qty)
            if qty < 1:
                return self._reject(candidate, "Calculated close quantity is below 1.")

        intent = OrderIntent(
            symbol=candidate.symbol,
            asset_class=candidate.asset_class,
            side=candidate.side,
            qty=qty,
            order_type=OrderType.LIMIT if candidate.asset_class == AssetClass.OPTION else OrderType.MARKET,
            time_in_force=TimeInForce.GTC if candidate.asset_class == AssetClass.CRYPTO else TimeInForce.DAY,
            limit_price=round(candidate.entry_price, 2) if candidate.asset_class == AssetClass.OPTION else None,
            client_order_id=self._client_order_id(candidate),
            metadata=candidate.metadata,
        )
        return RiskDecision(approved=True, reason="Approved position exit.", intent=intent, candidate=candidate)

    def _evaluate_spot(
        self,
        candidate: TradeCandidate,
        account: AccountSnapshot,
        positions: list[Position],
        available_cash: float,
    ) -> RiskDecision:
        max_position_pct = (
            self.settings.risk.max_crypto_position_pct
            if candidate.asset_class == AssetClass.CRYPTO
            else self.settings.risk.max_position_pct
        )
        max_position_value = account.equity * (max_position_pct / 100)
        existing = self._position_value(candidate.symbol, positions)
        remaining_position_capacity = max(max_position_value - existing, 0)
        if remaining_position_capacity <= 0:
            return self._reject(candidate, "Position cap already reached.")

        risk_budget = account.equity * (self.settings.risk.max_risk_per_trade_pct / 100)
        if candidate.stop_price and candidate.stop_price < candidate.entry_price:
            per_unit_risk = candidate.entry_price - candidate.stop_price
            qty_by_risk = risk_budget / per_unit_risk
        else:
            qty_by_risk = risk_budget / candidate.entry_price

        notional = min(qty_by_risk * candidate.entry_price, remaining_position_capacity, available_cash)
        if notional < 1:
            return self._reject(candidate, "Calculated order notional is below $1.")

        qty = None
        if candidate.asset_class != AssetClass.CRYPTO:
            qty = int(notional / candidate.entry_price)
            if qty < 1:
                return self._reject(candidate, "Calculated equity quantity is below 1 share.")
            notional = qty * candidate.entry_price

        intent = OrderIntent(
            symbol=candidate.symbol,
            asset_class=candidate.asset_class,
            side=candidate.side,
            qty=qty,
            notional=round(notional, 2) if candidate.asset_class == AssetClass.CRYPTO else None,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.GTC if candidate.asset_class == AssetClass.CRYPTO else TimeInForce.DAY,
            stop_loss_price=candidate.stop_price if candidate.asset_class != AssetClass.CRYPTO else None,
            take_profit_price=candidate.take_profit_price if candidate.asset_class != AssetClass.CRYPTO else None,
            client_order_id=self._client_order_id(candidate),
            metadata=candidate.metadata,
        )
        return RiskDecision(approved=True, reason="Approved by risk engine.", intent=intent, candidate=candidate)

    def _evaluate_option(
        self,
        candidate: TradeCandidate,
        account: AccountSnapshot,
        positions: list[Position],
        available_cash: float,
    ) -> RiskDecision:
        contract = candidate.metadata.get("contract", {})
        strategy = candidate.strategy
        if strategy == "covered_call":
            underlying = candidate.metadata.get("underlying")
            owned = self._position_qty(str(underlying), positions)
            contracts = int(owned // 100)
            if contracts < 1:
                return self._reject(candidate, "Covered call requires 100 underlying shares per contract.")
            return self._approve_option(candidate, min(contracts, 1), OrderType.LIMIT)

        strike = float(contract.get("strike_price", 0) or 0)
        if strategy == "cash_secured_put":
            required_cash = strike * 100
            if required_cash > available_cash:
                return self._reject(candidate, f"Cash-secured put requires ${required_cash:,.2f}.")
            return self._approve_option(candidate, 1, OrderType.LIMIT)

        if strategy in {"long_call", "long_put"}:
            max_premium = account.equity * (self.settings.risk.max_options_premium_pct / 100)
            premium = candidate.entry_price * 100
            if premium > max_premium:
                return self._reject(candidate, f"Option premium ${premium:,.2f} exceeds premium cap.")
            if premium > available_cash:
                return self._reject(candidate, "Option premium exceeds available cash.")
            return self._approve_option(candidate, 1, OrderType.LIMIT)

        if strategy in {"call_debit_spread", "put_debit_spread"}:
            max_premium = account.equity * (self.settings.risk.max_options_premium_pct / 100)
            debit = candidate.entry_price * 100
            if debit <= 0:
                return self._reject(candidate, "Debit spread requires a positive net debit price.")
            if debit > max_premium:
                return self._reject(candidate, f"Spread debit ${debit:,.2f} exceeds premium cap.")
            if debit > available_cash:
                return self._reject(candidate, "Spread debit exceeds available cash.")
            return self._approve_mleg(candidate, 1)

        return self._reject(candidate, f"Unsupported option strategy: {strategy}.")

    def _approve_mleg(self, candidate: TradeCandidate, qty: int) -> RiskDecision:
        intent = OrderIntent(
            symbol=candidate.symbol,
            asset_class=AssetClass.OPTION,
            side=OrderSide.BUY,
            qty=qty,
            order_type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            limit_price=round(candidate.entry_price, 2),
            order_class="mleg",
            legs=candidate.metadata["legs"],
            client_order_id=self._client_order_id(candidate),
            metadata=candidate.metadata,
        )
        return RiskDecision(approved=True, reason="Approved multi-leg option trade.", intent=intent, candidate=candidate)

    def _approve_option(
        self,
        candidate: TradeCandidate,
        qty: int,
        order_type: OrderType,
    ) -> RiskDecision:
        intent = OrderIntent(
            symbol=candidate.symbol,
            asset_class=AssetClass.OPTION,
            side=candidate.side,
            qty=qty,
            order_type=order_type,
            time_in_force=TimeInForce.DAY,
            limit_price=round(candidate.entry_price, 2),
            client_order_id=self._client_order_id(candidate),
            metadata=candidate.metadata,
        )
        return RiskDecision(approved=True, reason="Approved option trade.", intent=intent, candidate=candidate)

    def _position_value(self, symbol: str, positions: list[Position]) -> float:
        for position in positions:
            if position.symbol == symbol:
                return abs(position.market_value)
        return 0.0

    def _position(self, symbol: str, positions: list[Position]) -> Position | None:
        for position in positions:
            if position.symbol == symbol:
                return position
        return None

    def _position_qty(self, symbol: str, positions: list[Position]) -> float:
        for position in positions:
            if position.symbol == symbol:
                return position.qty
        return 0.0

    def _client_order_id(self, candidate: TradeCandidate) -> str:
        return f"ta-{candidate.strategy}-{uuid.uuid4().hex[:16]}"

    def _reject(self, candidate: TradeCandidate, reason: str) -> RiskDecision:
        return RiskDecision(approved=False, reason=reason, candidate=candidate)
