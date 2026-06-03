from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime

from rich.console import Console
from rich.table import Table

from trading_agent.audit import AuditStore, start_of_trading_day
from trading_agent.broker_sync import BrokerOrderSync
from trading_agent.brokers.alpaca import AlpacaBroker
from trading_agent.catalyst.service import CatalystEngine
from trading_agent.config import Settings
from trading_agent.models import (
    AssetClass,
    OrderIntent,
    OrderSide,
    OrderType,
    Position,
    RiskDecision,
    TimeInForce,
    TradeCandidate,
)
from trading_agent.position_manager import PositionManager
from trading_agent.research.service import ResearchService
from trading_agent.risk import RiskEngine
from trading_agent.screener.service import MarketScreener
from trading_agent.strategies.momentum import MomentumStrategy
from trading_agent.strategies.options import OptionsStrategy


# Substrings that mark a broker rejection as a holding/quantity conflict — i.e. a
# resting order is reserving the shares/qty an exit needs. Only these justify
# canceling resting (protective) orders and retrying the exit. Anything else
# (transient network/5xx, validation, account-not-eligible) must NOT strip
# protection. We deliberately match on message text, not Alpaca numeric codes:
# code 40310000 is a generic 403 reused for unrelated cases like "account not
# eligible to trade options", so it is not a reliable conflict signal.
_CONFLICT_ERROR_SIGNALS = (
    "insufficient qty",
    "insufficient balance",
    "qty available",
    "held for orders",
    "wash trade",
    "potential wash",
    "not enough",
)


class TradingAgent:
    def __init__(self, settings: Settings, console: Console | None = None) -> None:
        self.settings = settings
        self.console = console or Console()
        self.broker = AlpacaBroker(settings)
        self.research = ResearchService(settings)
        self.momentum = MomentumStrategy(settings)
        self.options = OptionsStrategy(settings)
        self.catalyst = CatalystEngine(settings)
        self.audit = AuditStore(settings.audit.database_path) if settings.audit.enabled else None
        self.broker_sync = BrokerOrderSync(self.broker, self.audit, self.console) if self.audit else None
        self.position_manager = PositionManager(settings, self.audit)
        self.risk = RiskEngine(settings)
        self.screener = MarketScreener(settings, self.broker)

    async def run_once(self) -> list[RiskDecision]:
        analysis_time = datetime.now().astimezone()
        account = await self.broker.get_account()
        positions = await self.broker.get_positions()
        cycle_id = self.audit.start_cycle(account, positions) if self.audit else None
        # On a daily loss stop, halt NEW entries but keep managing exits. We no
        # longer cancel all open orders here: cancel_all_orders() also wipes the
        # protective stop/take-profit legs of bracket orders, which would leave
        # open positions unguarded on exactly the worst day. The position manager
        # still runs below to actively trim/exit losers.
        halt_entries = account.daily_pl_pct <= -self.settings.risk.max_daily_loss_pct
        if halt_entries:
            self.console.print(
                f"[red]daily loss stop reached[/red] {account.daily_pl_pct:.2f}%; "
                "halting new entries and managing exits only"
            )
            self._audit_event(
                cycle_id,
                "risk_stop",
                {"account": account, "positions": positions},
                status="triggered",
                reason=f"Daily loss stop reached: {account.daily_pl_pct:.2f}%.",
            )
            await self._cancel_opening_orders_for_halt(cycle_id)
        try:
            if self.audit:
                self.audit.reconcile_position_states({position.symbol for position in positions})
            exit_candidates = self.position_manager.evaluate(positions)
            for candidate in exit_candidates:
                self._audit_candidate(cycle_id, candidate)
            entry_candidates = (
                [] if halt_entries else await self._generate_candidates(positions, cycle_id)
            )
            candidates = [*exit_candidates, *entry_candidates]
            decisions = [self.risk.evaluate(candidate, account, positions) for candidate in candidates]
            for decision in decisions:
                self._audit_decision(cycle_id, decision)
            self._print_decisions(account, decisions, analysis_time)

            if self.settings.agent.execute_orders:
                await self._submit_decisions(decisions, account, cycle_id, positions)
                await self._sync_recent_order_updates(cycle_id)
            if self.audit and cycle_id:
                self.audit.finish_cycle(cycle_id, status="stopped" if halt_entries else "completed")
            return decisions
        except Exception as exc:
            if self.audit and cycle_id:
                self.audit.finish_cycle(cycle_id, status="failed", error=str(exc) or repr(exc))
            raise

    async def loop(self) -> None:
        while True:
            try:
                await self.run_once()
            except Exception as exc:
                self.console.print(f"[red]agent cycle failed[/red] {exc}")
            await asyncio.sleep(self.settings.agent.cycle_interval(live=self.settings.is_live))

    async def _generate_candidates(
        self,
        positions: list[Position],
        cycle_id: int | None = None,
    ) -> list[TradeCandidate]:
        candidates: list[TradeCandidate] = []
        screened = await self._screened_symbols(cycle_id)
        for symbol, market in screened:
            self._audit_event(
                cycle_id,
                "market_snapshot",
                market,
                symbol=market.symbol,
                status="captured",
            )
            research = await self.research.research_symbol(symbol)
            self._audit_event(
                cycle_id,
                "research_result",
                research,
                symbol=research.symbol,
                status="captured",
            )
            prediction = self.catalyst.evaluate(market, research)
            if self.settings.catalyst.enabled:
                self._audit_event(
                    cycle_id,
                    "catalyst_prediction",
                    prediction,
                    symbol=market.symbol,
                    score=prediction.prediction_score,
                    status=prediction.direction,
                    reason=prediction.block_reason,
                )

            raw_symbol_candidates = self.momentum.evaluate(market, research)
            catalyst_entry = self.catalyst.entry_candidate(market, prediction, raw_symbol_candidates)
            if catalyst_entry:
                raw_symbol_candidates.append(catalyst_entry)
            symbol_candidates, blocked_candidates = self.catalyst.apply_to_candidates(
                raw_symbol_candidates,
                prediction,
            )
            for candidate in symbol_candidates:
                self._audit_candidate(cycle_id, candidate)
            for candidate, reason in blocked_candidates:
                self._audit_candidate(cycle_id, candidate, status="blocked", reason=reason)
            candidates.extend(symbol_candidates)
            if (
                self.settings.strategy.allow_options
                and market.asset_class in {AssetClass.EQUITY, AssetClass.ETF}
            ):
                option_candidates = await self._option_candidates(market, positions, candidates)
                option_candidates, blocked_options = self.catalyst.apply_to_candidates(
                    option_candidates,
                    prediction,
                )
                for candidate in option_candidates:
                    self._audit_candidate(cycle_id, candidate)
                for candidate, reason in blocked_options:
                    self._audit_candidate(cycle_id, candidate, status="blocked", reason=reason)
                candidates.extend(option_candidates)
        return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)

    async def _screened_symbols(
        self,
        cycle_id: int | None,
    ) -> list[tuple[str, object]]:
        if self.settings.screener.enabled:
            screened = await self.screener.top_symbols()
            self._audit_event(
                cycle_id,
                "screener_result",
                {
                    "enabled": True,
                    "symbols": [
                        {
                            "symbol": item.symbol,
                            "asset_class": item.asset_class.value,
                            "score": item.score,
                            "reasons": item.reasons,
                        }
                        for item in screened
                    ],
                    "fallback_symbols": self.settings.strategy.symbols,
                },
                status="captured",
            )
            if screened:
                return [(item.symbol, item.snapshot) for item in screened]
            self.console.print("[yellow]screener returned no symbols; falling back to configured symbols[/yellow]")

        symbols: list[tuple[str, object]] = []
        for symbol in self.settings.strategy.symbols:
            market = await self.broker.get_market_snapshot(symbol)
            symbols.append((symbol, market))
        return symbols

    async def _submit_decisions(
        self,
        decisions: list[RiskDecision],
        account,
        cycle_id: int | None,
        positions: list[Position] | None = None,
    ) -> None:
        open_orders = await self._open_order_reservations()
        daily_counts = self._daily_order_counts()
        max_entry_orders = self.settings.agent.max_entry_orders_per_day(live=self.settings.is_live)
        max_total_orders = self.settings.agent.max_total_orders_per_day(live=self.settings.is_live)
        # Cycle-aggregate cash budget. The risk engine sizes each candidate
        # independently against the same account snapshot, so without this the
        # agent could approve several buys that each believe they have the full
        # cash buffer. We decrement a shared budget as cash-consuming orders are
        # submitted and skip anything that would overrun it.
        remaining_cash = self._cycle_cash_budget(account)
        exit_decisions = [
            decision
            for decision in decisions
            if decision.candidate.metadata.get("exit")
        ]
        entry_decisions = [
            decision
            for decision in decisions
            if not decision.candidate.metadata.get("exit")
        ]

        submitted_entries = 0
        submitted_total = 0
        for decision in [*exit_decisions, *entry_decisions]:
            is_exit = bool(decision.candidate.metadata.get("exit"))
            if not is_exit and submitted_entries >= self.settings.agent.max_orders_per_cycle:
                break
            if not decision.approved or not decision.intent:
                continue
            cap_reason = self._daily_cap_reason(
                decision,
                daily_counts=daily_counts,
                submitted_entries=submitted_entries,
                submitted_total=submitted_total,
                max_entry_orders=max_entry_orders,
                max_total_orders=max_total_orders,
            )
            if cap_reason:
                self.console.print(f"[yellow]order skipped[/yellow] {decision.intent.symbol}: {cap_reason}")
                self._audit_event(
                    cycle_id,
                    "order",
                    {"intent": decision.intent, "daily_counts": daily_counts},
                    symbol=decision.intent.symbol,
                    strategy=decision.candidate.strategy,
                    status="skipped",
                    reason=cap_reason,
                )
                continue
            if not is_exit:
                skip_reason = self._skip_due_to_open_orders(decision, open_orders, positions)
                if skip_reason:
                    self.console.print(f"[yellow]order skipped[/yellow] {decision.intent.symbol}: {skip_reason}")
                    self._audit_event(
                        cycle_id,
                        "order",
                        {"intent": decision.intent, "open_orders": open_orders},
                        symbol=decision.intent.symbol,
                        strategy=decision.candidate.strategy,
                        status="skipped",
                        reason=skip_reason,
                    )
                    continue

            cash_required = self._estimated_cash_requirement(decision)
            if cash_required > remaining_cash + 0.01:
                budget_reason = (
                    f"Cycle cash budget exhausted: needs ${cash_required:,.2f}, "
                    f"${remaining_cash:,.2f} remaining after this cycle's orders."
                )
                self.console.print(f"[yellow]order skipped[/yellow] {decision.intent.symbol}: {budget_reason}")
                self._audit_event(
                    cycle_id,
                    "order",
                    {"intent": decision.intent, "cash_required": cash_required, "remaining_cash": remaining_cash},
                    symbol=decision.intent.symbol,
                    strategy=decision.candidate.strategy,
                    status="skipped",
                    reason=budget_reason,
                )
                continue
            order = None
            submit_error: Exception | None = None
            try:
                order = await self.broker.submit_order(decision.intent)
            except Exception as exc:
                submit_error = exc
                # An exit may clear resting orders to get filled, but only when the
                # rejection is a genuine holding/qty conflict. For transient or
                # unrelated errors we leave protective orders untouched and let the
                # position manager retry the exit on the next cycle.
                if is_exit and self._is_conflict_error(exc):
                    order, submit_error = await self._retry_exit_after_clearing_conflicts(
                        decision, open_orders, cycle_id, exc
                    )
            if order is None:
                self.console.print(
                    f"[red]order rejected[/red] {decision.intent.symbol}: {submit_error}"
                )
                self._audit_event(
                    cycle_id,
                    "order",
                    {"intent": decision.intent, "error": str(submit_error)},
                    symbol=decision.intent.symbol,
                    strategy=decision.candidate.strategy,
                    status="rejected",
                    reason=str(submit_error),
                )
                continue
            if is_exit:
                await self._cancel_conflicting_open_orders(decision, open_orders, cycle_id)
            remaining_cash = max(remaining_cash - cash_required, 0.0)
            if not is_exit:
                submitted_entries += 1
            submitted_total += 1
            self._audit_event(
                cycle_id,
                "order",
                {"intent": decision.intent, "broker_order": order},
                symbol=decision.intent.symbol,
                strategy=decision.candidate.strategy,
                status="submitted",
                reason=order.get("status"),
            )
            self.console.print(f"[green]submitted[/green] {order.get('id')} {decision.intent.symbol}")
            if not is_exit:
                await self._place_protective_stop(decision, order, cycle_id)

    def _cycle_cash_budget(self, account) -> float:
        cash_buffer = account.equity * (self.settings.risk.min_cash_buffer_pct / 100)
        spendable_balance = min(account.cash, account.buying_power)
        return max(spendable_balance - cash_buffer, 0.0)

    def _estimated_cash_requirement(self, decision: RiskDecision) -> float:
        intent = decision.intent
        candidate = decision.candidate
        if intent is None or candidate.metadata.get("exit"):
            return 0.0
        if intent.asset_class == AssetClass.OPTION:
            qty = int(intent.qty or 1)
            if candidate.strategy == "covered_call":
                # Collateralized by shares already held; no new cash required.
                return 0.0
            if candidate.strategy == "cash_secured_put":
                contract = candidate.metadata.get("contract", {})
                strike = float(contract.get("strike_price", 0) or 0)
                return strike * 100 * qty
            price = float(intent.limit_price or candidate.entry_price or 0)
            return price * 100 * qty
        if intent.side != OrderSide.BUY:
            return 0.0
        if intent.notional is not None:
            return float(intent.notional)
        price = float(intent.limit_price or candidate.entry_price or 0)
        return price * float(intent.qty or 0)

    async def _cancel_conflicting_open_orders(
        self,
        decision: RiskDecision,
        open_orders: dict,
        cycle_id: int | None,
    ) -> list[dict]:
        if not decision.intent:
            return []
        symbol = decision.intent.symbol
        if symbol not in open_orders["symbols"]:
            return []
        canceled: list[dict] = []
        for order in open_orders["orders"]:
            if str(order.get("symbol") or "") != symbol:
                continue
            order_id = str(order.get("id") or "")
            if not order_id:
                continue
            try:
                await self.broker.cancel_order(order_id)
            except Exception as exc:
                self.console.print(f"[yellow]could not cancel resting order[/yellow] {order_id}: {exc}")
                continue
            canceled.append(order)
            self._audit_event(
                cycle_id,
                "order",
                {"canceled_order_id": order_id, "symbol": symbol},
                symbol=symbol,
                strategy=decision.candidate.strategy,
                status="canceled",
                reason="Canceled resting order so the position exit can be submitted.",
            )
        open_orders["symbols"].discard(symbol)
        return canceled

    def _is_conflict_error(self, exc: Exception) -> bool:
        text = str(exc).lower()
        return any(signal in text for signal in _CONFLICT_ERROR_SIGNALS)

    async def _retry_exit_after_clearing_conflicts(
        self,
        decision: RiskDecision,
        open_orders: dict,
        cycle_id: int | None,
        original_exc: Exception,
    ) -> tuple[dict | None, Exception | None]:
        canceled = await self._cancel_conflicting_open_orders(decision, open_orders, cycle_id)
        if not canceled:
            return None, original_exc
        try:
            order = await self.broker.submit_order(decision.intent)
        except Exception as retry_exc:
            # The exit still failed after we cleared protection. Put the canceled
            # orders back so the position is not left unguarded; the position
            # manager will attempt the exit again next cycle.
            await self._rearm_orders(canceled, cycle_id)
            return None, retry_exc
        return order, None

    async def _rearm_orders(self, canceled: list[dict], cycle_id: int | None) -> None:
        # Only restore protective (position-reducing) resting orders. Re-arming an
        # unfilled entry would re-add risk, so non-protective canceled orders are
        # left alone.
        protective = [order for order in canceled if self._is_protective_order(order)]
        restored: set[str] = set()
        for order in protective:
            symbol = str(order.get("symbol") or "")
            if "/" in symbol and await self._rearm_crypto_protective(order, cycle_id):
                restored.add(symbol)
        # Equity bracket leaves two protective legs (a stop and a take-profit).
        # Rebuild them as a single OCO so they cannot both fill and oversell.
        equity_legs: dict[str, list[dict]] = defaultdict(list)
        for order in protective:
            symbol = str(order.get("symbol") or "")
            if "/" not in symbol:
                equity_legs[symbol].append(order)
        for symbol, legs in equity_legs.items():
            if await self._rearm_equity_protection(symbol, legs, cycle_id):
                restored.add(symbol)
        for order in protective:
            if str(order.get("symbol") or "") not in restored:
                self._warn_unprotected(order, cycle_id)

    def _is_protective_order(self, order: dict) -> bool:
        if "protect" in str(order.get("client_order_id") or ""):
            return True
        position_intent = str(order.get("position_intent") or "")
        if position_intent.endswith("_to_open"):
            return False
        side = str(order.get("side") or "")
        order_type = str(order.get("type") or order.get("order_type") or "").lower()
        # A resting sell stop/limit on a held long is a bracket protective leg.
        return side == "sell" and order_type in {"stop", "stop_limit", "limit"}

    async def _rearm_equity_protection(
        self,
        symbol: str,
        legs: list[dict],
        cycle_id: int | None,
    ) -> bool:
        stop_price: float | None = None
        take_profit_price: float | None = None
        qty: float | None = None
        for leg in legs:
            leg_type = str(leg.get("type") or leg.get("order_type") or "").lower()
            leg_qty = self._coerce_float(leg.get("qty"))
            if leg_qty and leg_qty > 0:
                qty = leg_qty if qty is None else max(qty, leg_qty)
            leg_stop = self._coerce_float(leg.get("stop_price"))
            if leg_stop and leg_stop > 0:
                stop_price = leg_stop
            if leg_type == "limit":
                leg_limit = self._coerce_float(leg.get("limit_price"))
                if leg_limit and leg_limit > 0:
                    take_profit_price = leg_limit
        protective = self._equity_protection_intent(symbol, qty, stop_price, take_profit_price)
        if protective is None:
            return False
        try:
            protect_order = await self.broker.submit_order(protective)
        except Exception as exc:
            self.console.print(f"[yellow]equity protection not restored[/yellow] {symbol}: {exc}")
            self._audit_event(
                cycle_id,
                "order",
                {"intent": protective, "error": str(exc)},
                symbol=symbol,
                strategy="equity_protective_exit",
                status="unprotected",
                reason=f"Could not restore bracket protection after a failed exit retry: {exc}",
            )
            return False
        self._audit_event(
            cycle_id,
            "order",
            {"intent": protective, "broker_order": protect_order},
            symbol=symbol,
            strategy="equity_protective_exit",
            status="submitted",
            reason="Restored bracket protection (OCO) after the exit retry failed.",
        )
        self.console.print(f"[green]bracket protection restored[/green] {symbol}")
        return True

    def _equity_protection_intent(
        self,
        symbol: str,
        qty: float | None,
        stop_price: float | None,
        take_profit_price: float | None,
    ) -> OrderIntent | None:
        if not qty or qty < 1:
            return None
        qty_int = int(qty)
        if qty_int < 1:
            return None
        client_order_id = f"ta-equity-protect-{uuid.uuid4().hex[:12]}"
        metadata = {"protective_stop": True, "parent_symbol": symbol}
        if stop_price is not None and take_profit_price is not None:
            return OrderIntent(
                symbol=symbol,
                asset_class=AssetClass.EQUITY,
                side=OrderSide.SELL,
                qty=qty_int,
                order_type=OrderType.LIMIT,
                time_in_force=TimeInForce.GTC,
                order_class="oco",
                stop_loss_price=round(stop_price, 2),
                take_profit_price=round(take_profit_price, 2),
                client_order_id=client_order_id,
                metadata=metadata,
            )
        if stop_price is not None:
            return OrderIntent(
                symbol=symbol,
                asset_class=AssetClass.EQUITY,
                side=OrderSide.SELL,
                qty=qty_int,
                order_type=OrderType.STOP,
                time_in_force=TimeInForce.GTC,
                stop_price=round(stop_price, 2),
                client_order_id=client_order_id,
                metadata=metadata,
            )
        if take_profit_price is not None:
            return OrderIntent(
                symbol=symbol,
                asset_class=AssetClass.EQUITY,
                side=OrderSide.SELL,
                qty=qty_int,
                order_type=OrderType.LIMIT,
                time_in_force=TimeInForce.GTC,
                limit_price=round(take_profit_price, 2),
                client_order_id=client_order_id,
                metadata=metadata,
            )
        return None

    def _warn_unprotected(self, order: dict, cycle_id: int | None) -> None:
        symbol = str(order.get("symbol") or "")
        self.console.print(
            f"[red]protection not restored[/red] {symbol}: exit retry failed and the "
            "resting order could not be auto-replaced"
        )
        self._audit_event(
            cycle_id,
            "order",
            {"canceled_order": order},
            symbol=symbol,
            strategy=self._strategy_from_order(order),
            status="unprotected",
            reason=(
                "Exit retry failed after canceling this resting order and it could not be "
                "auto-restored; the position manager will retry the exit next cycle."
            ),
        )

    async def _rearm_crypto_protective(self, order: dict, cycle_id: int | None) -> bool:
        symbol = str(order.get("symbol") or "")
        qty = self._coerce_float(order.get("qty"))
        stop_price = self._coerce_float(order.get("stop_price"))
        if not symbol or not qty or qty <= 0 or not stop_price or stop_price <= 0:
            return False
        parent = None
        if self.audit:
            parent = self.audit.protective_parent_for(str(order.get("client_order_id") or ""))
        protective = self._crypto_protective_intent(symbol, qty, stop_price, parent)
        try:
            protect_order = await self.broker.submit_order(protective)
        except Exception as exc:
            self.console.print(f"[yellow]crypto protective stop not restored[/yellow] {symbol}: {exc}")
            self._audit_event(
                cycle_id,
                "order",
                {"intent": protective, "error": str(exc)},
                symbol=symbol,
                strategy="crypto_protective_stop",
                status="unprotected",
                reason=f"Could not restore protective stop after a failed exit retry: {exc}",
            )
            return False
        self._audit_event(
            cycle_id,
            "order",
            {"intent": protective, "broker_order": protect_order},
            symbol=symbol,
            strategy="crypto_protective_stop",
            status="submitted",
            reason="Restored protective stop after the exit retry failed.",
        )
        self.console.print(f"[green]protective stop restored[/green] {symbol} @ {stop_price:.2f}")
        return True

    def _coerce_float(self, value) -> float | None:
        try:
            return float(value) if value not in {None, ""} else None
        except (TypeError, ValueError):
            return None

    async def _cancel_opening_orders_for_halt(self, cycle_id: int | None) -> None:
        try:
            open_orders = await self.broker.get_open_orders()
        except Exception as exc:
            self.console.print(f"[yellow]daily-loss open-order check failed[/yellow] {exc}")
            self._audit_event(
                cycle_id,
                "order",
                {"error": str(exc)},
                status="failed",
                reason=f"Could not inspect open orders during daily-loss halt: {exc}",
            )
            return

        for order in open_orders:
            if not self._is_opening_order(order):
                continue
            order_id = str(order.get("id") or "")
            if not order_id:
                continue
            try:
                await self.broker.cancel_order(order_id)
            except Exception as exc:
                self.console.print(f"[yellow]could not cancel opening order[/yellow] {order_id}: {exc}")
                self._audit_event(
                    cycle_id,
                    "order",
                    {"open_order": order, "error": str(exc)},
                    symbol=order.get("symbol"),
                    strategy=self._strategy_from_order(order),
                    status="failed",
                    reason=f"Could not cancel opening order during daily-loss halt: {exc}",
                )
                continue
            self._audit_event(
                cycle_id,
                "order",
                {"canceled_order": order},
                symbol=order.get("symbol"),
                strategy=self._strategy_from_order(order),
                status="canceled",
                reason="Canceled opening order during daily-loss halt; protective orders preserved.",
            )

    def _is_opening_order(self, order: dict) -> bool:
        client_order_id = str(order.get("client_order_id") or "")
        if "protect" in client_order_id:
            return False
        position_intent = str(order.get("position_intent") or "")
        if position_intent.endswith("_to_open"):
            return True
        if position_intent.endswith("_to_close"):
            return False
        order_type = str(order.get("type") or order.get("order_type") or "")
        side = str(order.get("side") or "")
        if client_order_id.startswith("ta-") and side == "buy" and order_type not in {"stop", "stop_limit"}:
            return True
        return False

    def _strategy_from_order(self, order: dict) -> str | None:
        client_order_id = str(order.get("client_order_id") or "")
        if not client_order_id.startswith("ta-"):
            return None
        return client_order_id[3:].rsplit("-", 1)[0] or None

    async def _place_protective_stop(
        self,
        decision: RiskDecision,
        order: dict,
        cycle_id: int | None,
    ) -> None:
        intent = decision.intent
        candidate = decision.candidate
        # Crypto cannot use Alpaca bracket orders, so a fresh crypto entry has no
        # broker-side stop. Rest a GTC stop-limit sell as a hard floor between
        # position-manager polls. If the broker rejects it we fall back to the
        # position manager's software stop (which always runs each cycle).
        if not intent or intent.asset_class != AssetClass.CRYPTO or intent.side != OrderSide.BUY:
            return
        stop_price = candidate.stop_price
        if not stop_price or stop_price <= 0:
            return
        qty = self._protective_qty(intent, order)
        if not qty or qty <= 0:
            self._audit_event(
                cycle_id,
                "order",
                {"intent": intent, "broker_order": order},
                symbol=intent.symbol,
                strategy="crypto_protective_stop",
                status="deferred",
                reason="Crypto entry is not filled yet; protective stop will be placed after broker fill sync.",
            )
            return
        protective = self._crypto_protective_intent(intent.symbol, qty, stop_price, intent.client_order_id)
        try:
            protect_order = await self.broker.submit_order(protective)
        except Exception as exc:
            self.console.print(
                f"[yellow]crypto protective stop not placed[/yellow] {intent.symbol}: {exc}"
            )
            self._audit_event(
                cycle_id,
                "order",
                {"intent": protective, "error": str(exc)},
                symbol=intent.symbol,
                strategy="crypto_protective_stop",
                status="rejected",
                reason=f"Protective stop rejected; relying on position-manager exit: {exc}",
            )
            return
        self._audit_event(
            cycle_id,
            "order",
            {"intent": protective, "broker_order": protect_order},
            symbol=intent.symbol,
            strategy="crypto_protective_stop",
            status="submitted",
            reason=protect_order.get("status"),
        )
        self.console.print(f"[green]protective stop[/green] {intent.symbol} @ {stop_price:.2f}")

    def _protective_qty(self, intent: OrderIntent, order: dict) -> float | None:
        filled = order.get("filled_qty") if isinstance(order, dict) else None
        try:
            filled_qty = float(filled) if filled not in {None, ""} else 0.0
        except (TypeError, ValueError):
            filled_qty = 0.0
        if filled_qty > 0:
            return round(filled_qty, 8)
        return None

    def _crypto_protective_intent(
        self,
        symbol: str,
        qty: float,
        stop_price: float,
        parent_client_order_id: str | None,
    ) -> OrderIntent:
        return OrderIntent(
            symbol=symbol,
            asset_class=AssetClass.CRYPTO,
            side=OrderSide.SELL,
            qty=qty,
            order_type=OrderType.STOP_LIMIT,
            time_in_force=TimeInForce.GTC,
            stop_price=round(stop_price, 2),
            limit_price=round(stop_price * 0.99, 2),
            client_order_id=f"ta-crypto-protect-{uuid.uuid4().hex[:12]}",
            metadata={
                "protective_stop": True,
                "parent_symbol": symbol,
                "parent_client_order_id": parent_client_order_id,
            },
        )

    async def _sync_recent_order_updates(self, cycle_id: int | None) -> None:
        if not self.broker_sync:
            return
        try:
            result = await self.broker_sync.sync_closed_orders(
                cycle_id=cycle_id,
                after=start_of_trading_day().isoformat(),
                include_orders=True,
            )
        except Exception as exc:
            self.console.print(f"[yellow]order fill audit skipped[/yellow] {exc}")
            self._audit_event(
                cycle_id,
                "broker_order_update",
                {"error": str(exc)},
                status="failed",
                reason=str(exc),
            )
            return
        for order in result.get("orders", []):
            await self._place_crypto_protective_stop_from_fill(order, cycle_id)

    async def _place_crypto_protective_stop_from_fill(
        self,
        broker_order: dict,
        cycle_id: int | None,
    ) -> None:
        if not self.audit:
            return
        symbol = str(broker_order.get("symbol") or "")
        asset_class = str(broker_order.get("asset_class") or "")
        if "/" not in symbol and asset_class != "crypto":
            return
        if str(broker_order.get("side") or "") != "buy":
            return
        client_order_id = str(broker_order.get("client_order_id") or "")
        if not client_order_id or "protect" in client_order_id:
            return
        if self.audit.crypto_protective_stop_exists(client_order_id):
            return
        qty = self._filled_qty(broker_order)
        if not qty:
            return
        current_qty = await self._current_position_qty(symbol)
        if current_qty <= 0:
            return
        qty = min(qty, current_qty)
        entry_event = self.audit.order_event_by_client_order_id(client_order_id)
        if not entry_event:
            return
        intent = (entry_event.get("payload") or {}).get("intent") or {}
        stop_price = intent.get("stop_loss_price")
        if not stop_price:
            return
        protective = self._crypto_protective_intent(symbol, qty, float(stop_price), client_order_id)
        try:
            protect_order = await self.broker.submit_order(protective)
        except Exception as exc:
            self.console.print(f"[yellow]crypto protective stop not placed[/yellow] {symbol}: {exc}")
            self._audit_event(
                cycle_id,
                "order",
                {"intent": protective, "broker_order": broker_order, "error": str(exc)},
                symbol=symbol,
                strategy="crypto_protective_stop",
                status="rejected",
                reason=f"Protective stop rejected after fill sync; relying on position-manager exit: {exc}",
            )
            return
        self._audit_event(
            cycle_id,
            "order",
            {"intent": protective, "broker_order": protect_order, "parent_broker_order": broker_order},
            symbol=symbol,
            strategy="crypto_protective_stop",
            status="submitted",
            reason=protect_order.get("status"),
        )

    def _filled_qty(self, order: dict) -> float | None:
        value = order.get("filled_qty")
        try:
            qty = float(value) if value not in {None, ""} else 0.0
        except (TypeError, ValueError):
            return None
        return round(qty, 8) if qty > 0 else None

    async def _current_position_qty(self, symbol: str) -> float:
        try:
            positions = await self.broker.get_positions()
        except Exception as exc:
            self.console.print(f"[yellow]position check failed before crypto stop[/yellow] {symbol}: {exc}")
            return 0.0
        for position in positions:
            if position.symbol == symbol:
                return max(float(position.qty), 0.0)
        return 0.0

    def _daily_order_counts(self) -> dict[str, int]:
        if not self.audit:
            return {"total_orders": 0, "entry_orders": 0, "exit_orders": 0}
        return self.audit.order_counts_since(start_of_trading_day())

    def _daily_cap_reason(
        self,
        decision: RiskDecision,
        *,
        daily_counts: dict[str, int],
        submitted_entries: int,
        submitted_total: int,
        max_entry_orders: int,
        max_total_orders: int,
    ) -> str | None:
        projected_total = daily_counts["total_orders"] + submitted_total + 1
        if max_total_orders and projected_total > max_total_orders:
            return (
                f"Daily total order cap reached "
                f"({daily_counts['total_orders']}/{max_total_orders} already submitted today)."
            )
        if decision.candidate.metadata.get("exit"):
            return None

        projected_entries = daily_counts["entry_orders"] + submitted_entries + 1
        if max_entry_orders and projected_entries > max_entry_orders:
            return (
                f"Daily entry order cap reached "
                f"({daily_counts['entry_orders']}/{max_entry_orders} already submitted today)."
            )
        return None

    async def _open_order_reservations(self) -> dict:
        try:
            open_orders = await self.broker.get_open_orders()
        except Exception as exc:
            self.console.print(f"[yellow]open-order check failed[/yellow] {exc}")
            return {"symbols": set(), "covered_call_contracts_by_underlying": {}, "orders": []}

        symbols = set()
        covered_call_contracts_by_underlying: dict[str, int] = {}
        for order in open_orders:
            symbol = str(order.get("symbol") or "")
            side = str(order.get("side") or "")
            qty = int(float(order.get("qty") or 0))
            symbols.add(symbol)
            parsed = self._parse_option_symbol(symbol)
            if side == "sell" and parsed and parsed["type"] == "C":
                underlying = parsed["underlying"]
                covered_call_contracts_by_underlying[underlying] = (
                    covered_call_contracts_by_underlying.get(underlying, 0) + qty
                )
        return {
            "symbols": symbols,
            "covered_call_contracts_by_underlying": covered_call_contracts_by_underlying,
            "orders": open_orders,
        }

    def _skip_due_to_open_orders(
        self,
        decision: RiskDecision,
        open_orders: dict,
        positions: list[Position] | None = None,
    ) -> str | None:
        if not decision.intent:
            return None
        if decision.intent.symbol in open_orders["symbols"]:
            return "Open order already exists for this symbol."
        if decision.candidate.strategy != "covered_call":
            return None

        underlying = str(decision.candidate.metadata.get("underlying") or "")
        contracts_per_100_shares = int(decision.candidate.metadata.get("contracts_per_100_shares") or 0)
        already_reserved = open_orders["covered_call_contracts_by_underlying"].get(underlying, 0)
        requested = int(decision.intent.qty or 0)
        # Shares are committed both by covered calls already written (filled short
        # positions) and by covered-call orders still working. Count both so the
        # agent never submits an order that would overrun the underlying coverage.
        already_written = self._short_call_contracts(underlying, positions or [])
        if already_reserved + requested + already_written > contracts_per_100_shares:
            message = (
                f"Open covered-call orders already reserve {already_reserved * 100} "
                f"of {contracts_per_100_shares * 100} available {underlying} shares."
            )
            if already_written:
                message += f" {already_written} call(s) already written against this underlying."
            return message
        return None

    def _short_call_contracts(self, underlying: str, positions: list[Position]) -> int:
        underlying = str(underlying or "")
        if not underlying or not positions:
            return 0
        total = 0
        for position in positions:
            if position.asset_class != AssetClass.OPTION or position.qty >= 0:
                continue
            parsed = self._parse_option_symbol(position.symbol)
            if parsed and parsed["underlying"] == underlying and parsed["type"] == "C":
                total += int(abs(position.qty))
        return total

    def _parse_option_symbol(self, symbol: str) -> dict | None:
        # OCC-style symbols here look like AAPL260612C00322500.
        for index, char in enumerate(symbol):
            if char.isdigit():
                if len(symbol) < index + 15:
                    return None
                date_part = symbol[index : index + 6]
                option_type = symbol[index + 6 : index + 7]
                if not date_part.isdigit() or option_type not in {"C", "P"}:
                    return None
                try:
                    expiration = datetime.strptime(date_part, "%y%m%d").date().isoformat()
                except ValueError:
                    expiration = None
                return {
                    "underlying": symbol[:index],
                    "expiration": expiration,
                    "type": option_type,
                }
        return None

    async def _option_candidates(
        self,
        market,
        positions: list[Position],
        existing_candidates: Iterable[TradeCandidate],
    ) -> list[TradeCandidate]:
        try:
            contracts = await self.broker.get_option_contracts(market.symbol)
        except Exception as exc:
            self.console.print(f"[yellow]options skipped for {market.symbol}[/yellow] {exc}")
            return []
        owned_shares = next((position.qty for position in positions if position.symbol == market.symbol), 0)
        bullish_score = max(
            (candidate.score for candidate in existing_candidates if candidate.symbol == market.symbol),
            default=0,
        )
        candidates = [
            *self.options.covered_call_candidates(market, contracts, owned_shares),
            *self.options.cash_secured_put_candidates(market, contracts),
            *self.options.long_option_candidates(market, contracts, bullish_score),
            *self.options.debit_spread_candidates(market, contracts, bullish_score),
        ]
        return await self._hydrate_option_prices(candidates)

    async def _hydrate_option_prices(self, candidates: list[TradeCandidate]) -> list[TradeCandidate]:
        priced: list[TradeCandidate] = []
        skipped = 0
        for candidate in candidates:
            if candidate.metadata.get("legs"):
                spread_price = await self._spread_debit(candidate)
                if spread_price and spread_price > 0:
                    priced.append(candidate.model_copy(update={"entry_price": round(spread_price, 2)}))
                else:
                    skipped += 1
                continue
            price = await self._single_option_price(candidate)
            if not price or price <= 0:
                skipped += 1
                continue
            priced.append(candidate.model_copy(update={"entry_price": round(price, 2)}))
        if skipped:
            self.console.print(f"[yellow]skipped {skipped} option candidate(s) without usable pricing[/yellow]")
        return priced

    async def _single_option_price(self, candidate: TradeCandidate) -> float | None:
        quote_price = await self._option_quote_price(
            candidate.symbol,
            preferred_side="ask" if candidate.side.value == "buy" else "bid",
        )
        if quote_price:
            return quote_price
        price = self._contract_close_price(candidate)
        try:
            option_market = await self.broker.get_market_snapshot(candidate.symbol)
            price = option_market.price
        except Exception:
            pass
        return price

    async def _spread_debit(self, candidate: TradeCandidate) -> float | None:
        net = 0.0
        contracts = {contract["symbol"]: contract for contract in candidate.metadata.get("contracts", [])}
        for leg in candidate.metadata.get("legs", []):
            symbol = leg["symbol"]
            price = await self._option_quote_price(
                symbol,
                preferred_side="ask" if leg["side"] == "buy" else "bid",
            )
            if not price:
                price = self._close_price_from_contract(contracts.get(symbol, {}))
                try:
                    option_market = await self.broker.get_market_snapshot(symbol)
                    price = option_market.price
                except Exception:
                    pass
            if not price or price <= 0:
                return None
            net += price if leg["side"] == "buy" else -price
        return max(net, 0)

    async def _option_quote_price(self, symbol: str, *, preferred_side: str) -> float | None:
        try:
            quote = await self.broker.get_option_latest_quote(symbol)
        except Exception:
            return None
        bid, ask = self.broker.option_quote_bid_ask(quote)
        if preferred_side == "ask" and ask and ask > 0:
            return ask
        if preferred_side == "bid" and bid and bid > 0:
            return bid
        return self.broker.option_quote_midpoint(quote)

    def _contract_close_price(self, candidate: TradeCandidate) -> float | None:
        contract = candidate.metadata.get("contract", {})
        return self._close_price_from_contract(contract)

    def _close_price_from_contract(self, contract) -> float | None:
        close_price = contract.get("close_price")
        if close_price in {None, ""}:
            return None
        return float(close_price)

    def _print_decisions(
        self,
        account,
        decisions: list[RiskDecision],
        analysis_time: datetime | None = None,
    ) -> None:
        timestamp = (analysis_time or datetime.now().astimezone()).strftime("%Y-%m-%d %H:%M:%S %Z")
        self.console.print(
            f"Analysis Time {timestamp} | Equity ${account.equity:,.2f} | Cash ${account.cash:,.2f} | "
            f"Buying Power ${account.buying_power:,.2f} | Daily P/L {account.daily_pl_pct:.2f}%"
        )
        table = Table("Approved", "Symbol", "Strategy", "Score", "Reason")
        for decision in decisions:
            table.add_row(
                "yes" if decision.approved else "no",
                decision.candidate.symbol,
                decision.candidate.strategy,
                f"{decision.candidate.score:.2f}",
                decision.reason,
            )
        self.console.print(table)

    def _audit_candidate(
        self,
        cycle_id: int | None,
        candidate: TradeCandidate,
        *,
        status: str = "generated",
        reason: str | None = None,
    ) -> None:
        self._audit_event(
            cycle_id,
            "trade_candidate",
            candidate,
            symbol=candidate.symbol,
            strategy=candidate.strategy,
            score=candidate.score,
            status=status,
            reason=reason,
        )

    def _audit_decision(self, cycle_id: int | None, decision: RiskDecision) -> None:
        self._audit_event(
            cycle_id,
            "risk_decision",
            decision,
            symbol=decision.candidate.symbol,
            strategy=decision.candidate.strategy,
            approved=decision.approved,
            score=decision.candidate.score,
            status="approved" if decision.approved else "rejected",
            reason=decision.reason,
        )

    def _audit_event(
        self,
        cycle_id: int | None,
        event_type: str,
        payload,
        *,
        symbol: str | None = None,
        strategy: str | None = None,
        approved: bool | None = None,
        score: float | None = None,
        status: str | None = None,
        reason: str | None = None,
    ) -> None:
        if not self.audit:
            return
        self.audit.record_event(
            cycle_id=cycle_id,
            event_type=event_type,
            payload=payload,
            symbol=symbol,
            strategy=strategy,
            approved=approved,
            score=score,
            status=status,
            reason=reason,
        )
