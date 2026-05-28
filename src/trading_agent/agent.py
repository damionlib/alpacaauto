from __future__ import annotations

import asyncio
from collections.abc import Iterable
from datetime import datetime

from rich.console import Console
from rich.table import Table

from trading_agent.audit import AuditStore, start_of_trading_day
from trading_agent.brokers.alpaca import AlpacaBroker
from trading_agent.config import Settings
from trading_agent.models import AssetClass, Position, RiskDecision, TradeCandidate
from trading_agent.position_manager import PositionManager
from trading_agent.research.service import ResearchService
from trading_agent.risk import RiskEngine
from trading_agent.strategies.momentum import MomentumStrategy
from trading_agent.strategies.options import OptionsStrategy


class TradingAgent:
    def __init__(self, settings: Settings, console: Console | None = None) -> None:
        self.settings = settings
        self.console = console or Console()
        self.broker = AlpacaBroker(settings)
        self.research = ResearchService(settings)
        self.momentum = MomentumStrategy(settings)
        self.options = OptionsStrategy(settings)
        self.audit = AuditStore(settings.audit.database_path) if settings.audit.enabled else None
        self.position_manager = PositionManager(settings, self.audit)
        self.risk = RiskEngine(settings)

    async def run_once(self) -> list[RiskDecision]:
        account = await self.broker.get_account()
        positions = await self.broker.get_positions()
        cycle_id = self.audit.start_cycle(account, positions) if self.audit else None
        if account.daily_pl_pct <= -self.settings.risk.max_daily_loss_pct:
            self.console.print(
                f"[red]daily loss stop reached[/red] {account.daily_pl_pct:.2f}%; canceling open orders"
            )
            self._audit_event(
                cycle_id,
                "risk_stop",
                {"account": account, "positions": positions},
                status="triggered",
                reason=f"Daily loss stop reached: {account.daily_pl_pct:.2f}%.",
            )
            await self.broker.cancel_all_orders()
            if self.audit and cycle_id:
                self.audit.finish_cycle(cycle_id, status="stopped")
            return []
        try:
            if self.audit:
                self.audit.reconcile_position_states({position.symbol for position in positions})
            exit_candidates = self.position_manager.evaluate(positions)
            for candidate in exit_candidates:
                self._audit_candidate(cycle_id, candidate)
            entry_candidates = await self._generate_candidates(positions, cycle_id)
            candidates = [*exit_candidates, *entry_candidates]
            decisions = [self.risk.evaluate(candidate, account, positions) for candidate in candidates]
            for decision in decisions:
                self._audit_decision(cycle_id, decision)
            self._print_decisions(account, decisions)

            if self.settings.agent.execute_orders:
                await self._submit_decisions(decisions, cycle_id)
            if self.audit and cycle_id:
                self.audit.finish_cycle(cycle_id)
            return decisions
        except Exception as exc:
            if self.audit and cycle_id:
                self.audit.finish_cycle(cycle_id, status="failed", error=str(exc))
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
        for symbol in self.settings.strategy.symbols:
            market = await self.broker.get_market_snapshot(symbol)
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
            symbol_candidates = self.momentum.evaluate(market, research)
            for candidate in symbol_candidates:
                self._audit_candidate(cycle_id, candidate)
            candidates.extend(symbol_candidates)
            if (
                self.settings.strategy.allow_options
                and market.asset_class in {AssetClass.EQUITY, AssetClass.ETF}
            ):
                option_candidates = await self._option_candidates(market, positions, candidates)
                for candidate in option_candidates:
                    self._audit_candidate(cycle_id, candidate)
                candidates.extend(option_candidates)
        return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)

    async def _submit_decisions(
        self,
        decisions: list[RiskDecision],
        cycle_id: int | None,
    ) -> None:
        open_orders = await self._open_order_reservations()
        daily_counts = self._daily_order_counts()
        max_entry_orders = self.settings.agent.max_entry_orders_per_day(live=self.settings.is_live)
        max_total_orders = self.settings.agent.max_total_orders_per_day(live=self.settings.is_live)
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
            if (
                not decision.candidate.metadata.get("exit")
                and submitted_entries >= self.settings.agent.max_orders_per_cycle
            ):
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
            skip_reason = self._skip_due_to_open_orders(decision, open_orders)
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
            try:
                order = await self.broker.submit_order(decision.intent)
            except Exception as exc:
                self.console.print(f"[red]order rejected[/red] {decision.intent.symbol}: {exc}")
                self._audit_event(
                    cycle_id,
                    "order",
                    {"intent": decision.intent, "error": str(exc)},
                    symbol=decision.intent.symbol,
                    strategy=decision.candidate.strategy,
                    status="rejected",
                    reason=str(exc),
                )
                continue
            if not decision.candidate.metadata.get("exit"):
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

    def _skip_due_to_open_orders(self, decision: RiskDecision, open_orders: dict) -> str | None:
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
        if already_reserved + requested > contracts_per_100_shares:
            return (
                f"Open covered-call orders already reserve {already_reserved * 100} "
                f"of {contracts_per_100_shares * 100} available {underlying} shares."
            )
        return None

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

    def _print_decisions(self, account, decisions: list[RiskDecision]) -> None:
        self.console.print(
            f"Equity ${account.equity:,.2f} | Cash ${account.cash:,.2f} | "
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

    def _audit_candidate(self, cycle_id: int | None, candidate: TradeCandidate) -> None:
        self._audit_event(
            cycle_id,
            "trade_candidate",
            candidate,
            symbol=candidate.symbol,
            strategy=candidate.strategy,
            score=candidate.score,
            status="generated",
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
