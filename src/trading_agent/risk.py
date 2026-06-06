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

        if candidate.metadata.get("day_trade"):
            # Gate day trades on the day-trade book's OWN daily P/L (stamped by the
            # agent), not the shared account P/L — so a swing drawdown does not shut
            # off day trading. Falls back to account P/L when not stamped.
            max_daily_loss_pct = self.settings.day_trading.max_daily_loss_pct
            daily_pl_pct = float(candidate.metadata.get("day_trade_daily_pl_pct", account.daily_pl_pct))
        else:
            max_daily_loss_pct = self.settings.risk.max_daily_loss_pct
            daily_pl_pct = account.daily_pl_pct
        if daily_pl_pct <= -max_daily_loss_pct:
            return self._reject(candidate, f"Daily loss stop reached: {daily_pl_pct:.2f}%.")

        cash_buffer = account.equity * (self.settings.risk.min_cash_buffer_pct / 100)
        spendable_balance = min(account.cash, account.buying_power)
        available_cash = max(spendable_balance - cash_buffer, 0)
        if candidate.side == OrderSide.BUY and available_cash <= 0:
            return self._reject(candidate, "Cash/buying-power buffer would be breached.")

        if candidate.side == OrderSide.BUY:
            ok, exposure, cap = self._correlated_exposure_ok(candidate, account, positions)
            if not ok:
                return self._reject(
                    candidate,
                    f"Correlated-group exposure cap reached: ${exposure:,.0f} of ${cap:,.0f} already deployed.",
                )

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
        if candidate.metadata.get("spread_exit"):
            return self._evaluate_spread_exit(candidate, positions)

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

    def _evaluate_spread_exit(
        self,
        candidate: TradeCandidate,
        positions: list[Position],
    ) -> RiskDecision:
        legs = candidate.metadata.get("legs") or []
        if not legs:
            return self._reject(candidate, "Spread exit requires close legs.")
        requested_qty = int(float(candidate.metadata.get("exit_qty") or 1))
        if requested_qty < 1:
            return self._reject(candidate, "Calculated spread close quantity is below 1.")

        for leg in legs:
            symbol = str(leg.get("symbol") or "")
            side = str(leg.get("side") or "")
            position = self._position(symbol, positions)
            if not position:
                return self._reject(candidate, f"No existing spread leg found to close: {symbol}.")
            if side == "sell" and position.qty <= 0:
                return self._reject(candidate, f"Spread leg {symbol} is not a long leg.")
            if side == "buy" and position.qty >= 0:
                return self._reject(candidate, f"Spread leg {symbol} is not a short leg.")
            if abs(position.qty) < requested_qty:
                return self._reject(candidate, f"Spread leg {symbol} has insufficient quantity to close.")

        if candidate.entry_price <= 0:
            return self._reject(candidate, "Spread close requires a positive net credit price.")

        intent = OrderIntent(
            symbol=candidate.symbol,
            asset_class=AssetClass.OPTION,
            side=candidate.side,
            qty=requested_qty,
            order_type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            limit_price=round(candidate.entry_price, 2),
            order_class="mleg",
            legs=legs,
            client_order_id=self._client_order_id(candidate),
            metadata=candidate.metadata,
        )
        return RiskDecision(approved=True, reason="Approved multi-leg position exit.", intent=intent, candidate=candidate)

    def _evaluate_spot(
        self,
        candidate: TradeCandidate,
        account: AccountSnapshot,
        positions: list[Position],
        available_cash: float,
    ) -> RiskDecision:
        if candidate.metadata.get("day_trade"):
            max_position_pct = self.settings.day_trading.max_position_pct
        else:
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

        risk_per_trade_pct = (
            self.settings.day_trading.risk_per_trade_pct
            if candidate.metadata.get("day_trade")
            else self.settings.risk.max_risk_per_trade_pct
        )
        risk_budget = account.equity * (risk_per_trade_pct / 100)
        if candidate.stop_price and candidate.stop_price < candidate.entry_price:
            per_unit_risk = candidate.entry_price - candidate.stop_price
            qty_by_risk = risk_budget / per_unit_risk
        else:
            qty_by_risk = risk_budget / candidate.entry_price

        notional = min(qty_by_risk * candidate.entry_price, remaining_position_capacity, available_cash)
        if notional < 1:
            return self._reject(candidate, "Calculated order notional is below $1.")

        # Price entries as marketable limit orders so a gap or thin quote cannot
        # fill far away from the price the sizing/stop math assumed.
        slippage = self.settings.risk.max_entry_slippage_pct / 100
        if candidate.side == OrderSide.BUY:
            limit_price = round(candidate.entry_price * (1 + slippage), 2)
        else:
            limit_price = round(candidate.entry_price * (1 - slippage), 2)

        if candidate.asset_class == AssetClass.CRYPTO:
            qty = round(notional / candidate.entry_price, 8)
            if qty <= 0:
                return self._reject(candidate, "Calculated crypto quantity is zero.")
            intent = OrderIntent(
                symbol=candidate.symbol,
                asset_class=candidate.asset_class,
                side=candidate.side,
                qty=qty,
                order_type=OrderType.LIMIT,
                time_in_force=TimeInForce.GTC,
                limit_price=limit_price,
                stop_loss_price=candidate.stop_price,
                client_order_id=self._client_order_id(candidate),
                metadata=candidate.metadata,
            )
            return RiskDecision(approved=True, reason="Approved by risk engine.", intent=intent, candidate=candidate)

        qty = int(notional / candidate.entry_price)
        if qty < 1:
            return self._reject(candidate, "Calculated equity quantity is below 1 share.")

        intent = OrderIntent(
            symbol=candidate.symbol,
            asset_class=candidate.asset_class,
            side=candidate.side,
            qty=qty,
            order_type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            limit_price=limit_price,
            stop_loss_price=candidate.stop_price,
            take_profit_price=candidate.take_profit_price,
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
            underlying = str(candidate.metadata.get("underlying") or "")
            owned = self._position_qty(underlying, positions)
            coverable_contracts = int(owned // 100)
            if coverable_contracts < 1:
                return self._reject(candidate, "Covered call requires 100 underlying shares per contract.")
            # Shares already committed to calls we have written cannot back another
            # contract. Writing past that coverage would create a (partially) naked
            # short call, which the broker rejects as "not eligible".
            already_written = self._short_call_contracts(underlying, positions)
            if coverable_contracts - already_written < 1:
                return self._reject(
                    candidate,
                    f"Covered-call coverage already used: {already_written} call(s) written against "
                    f"{coverable_contracts * 100} coverable {underlying} shares.",
                )
            return self._approve_option(candidate, 1, OrderType.LIMIT)

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

    def _short_call_contracts(self, underlying: str, positions: list[Position]) -> int:
        total = 0
        for position in positions:
            if position.asset_class != AssetClass.OPTION or position.qty >= 0:
                continue
            parsed = _occ_underlying_and_type(position.symbol)
            if parsed and parsed[0] == underlying and parsed[1] == "C":
                total += int(abs(position.qty))
        return total

    def _client_order_id(self, candidate: TradeCandidate) -> str:
        return f"ta-{candidate.strategy}-{uuid.uuid4().hex[:16]}"

    def _group_for(self, symbol: str) -> set[str] | None:
        symbol = (symbol or "").upper()
        for group in self.settings.risk.correlated_groups:
            members = {str(s).upper() for s in group}
            if symbol in members:
                return members
        return None

    def _correlated_exposure_ok(
        self,
        candidate: TradeCandidate,
        account: AccountSnapshot,
        positions: list[Position],
    ) -> tuple[bool, float, float]:
        cfg = self.settings.risk
        if cfg.max_correlated_exposure_pct <= 0 or not cfg.correlated_groups:
            return True, 0.0, 0.0
        underlying = str(candidate.metadata.get("underlying") or candidate.symbol)
        group = self._group_for(underlying)
        if not group:
            return True, 0.0, 0.0
        cap = account.equity * (cfg.max_correlated_exposure_pct / 100)
        exposure = 0.0
        for position in positions:
            sym = (position.symbol or "").upper()
            parsed = _occ_underlying_and_type(position.symbol)
            base = parsed[0].upper() if parsed else sym
            if base in group or sym in group:
                exposure += abs(position.market_value)
        return exposure < cap, exposure, cap

    def _reject(self, candidate: TradeCandidate, reason: str) -> RiskDecision:
        return RiskDecision(approved=False, reason=reason, candidate=candidate)


def _occ_underlying_and_type(symbol: str) -> tuple[str, str] | None:
    # OCC-style symbols look like AAPL260612C00322500: <root><yymmdd><C|P><strike>.
    for index, char in enumerate(symbol):
        if char.isdigit():
            if len(symbol) < index + 15:
                return None
            date_part = symbol[index : index + 6]
            option_type = symbol[index + 6 : index + 7]
            if not date_part.isdigit() or option_type not in {"C", "P"}:
                return None
            return symbol[:index], option_type
    return None
