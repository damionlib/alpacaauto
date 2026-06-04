from __future__ import annotations

from datetime import UTC, datetime

from trading_agent.audit import AuditStore
from trading_agent.config import Settings
from trading_agent.models import AssetClass, OrderSide, Position, TradeCandidate

OPTION_MULTIPLIER = 100


class PositionManager:
    def __init__(self, settings: Settings, audit: AuditStore | None = None) -> None:
        self.settings = settings
        self.audit = audit
        self._memory_state: dict[str, dict] = {}

    def evaluate(self, positions: list[Position]) -> list[TradeCandidate]:
        if not self.settings.position_manager.enabled:
            return []

        candidates: list[TradeCandidate] = []
        paired_option_symbols: set[str] = set()
        if self.settings.position_manager.manage_options:
            spread_candidates, paired_option_symbols = self._spread_exit_candidates(positions)
            candidates.extend(spread_candidates)

        for position in positions:
            if position.qty == 0:
                continue
            if position.asset_class == AssetClass.OPTION and not self.settings.position_manager.manage_options:
                continue
            if position.symbol in paired_option_symbols:
                continue

            current_price = self._current_price(position)
            if current_price is None or current_price <= 0:
                continue

            state = self._update_state(position, current_price)
            metrics = self._metrics(position, current_price, state)
            exit_reason = self._exit_reason(position, metrics)
            if not exit_reason:
                continue

            candidates.append(
                TradeCandidate(
                    symbol=position.symbol,
                    asset_class=position.asset_class,
                    side=OrderSide.SELL if position.qty > 0 else OrderSide.BUY,
                    strategy=exit_reason["strategy"],
                    score=exit_reason["score"],
                    entry_price=current_price,
                    rationale=exit_reason["rationale"],
                    metadata={
                        "exit": True,
                        "exit_qty": abs(position.qty),
                        "position": position.model_dump(mode="json"),
                        "position_state": state,
                        "metrics": metrics,
                    },
                )
            )
        return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)

    def _spread_exit_candidates(self, positions: list[Position]) -> tuple[list[TradeCandidate], set[str]]:
        option_positions = [
            position
            for position in positions
            if position.asset_class == AssetClass.OPTION and position.qty != 0
        ]
        grouped: dict[tuple[str, str | None, str], list[tuple[Position, dict]]] = {}
        for position in option_positions:
            parts = self._parse_option_symbol(position.symbol)
            if not parts:
                continue
            key = (parts["underlying"], parts["expiration"], parts["type"])
            grouped.setdefault(key, []).append((position, parts))

        candidates: list[TradeCandidate] = []
        paired_symbols: set[str] = set()
        for (underlying, _expiration, option_type), rows in grouped.items():
            long_rows = [(position, parts) for position, parts in rows if position.qty > 0]
            short_rows = [(position, parts) for position, parts in rows if position.qty < 0]
            for long_position, long_parts in long_rows:
                for short_position, short_parts in short_rows:
                    if long_position.symbol in paired_symbols or short_position.symbol in paired_symbols:
                        continue
                    strategy = self._debit_spread_strategy(option_type, long_parts["strike"], short_parts["strike"])
                    if not strategy:
                        continue
                    candidate = self._spread_exit_candidate(
                        underlying=underlying,
                        strategy=strategy,
                        long_position=long_position,
                        short_position=short_position,
                    )
                    paired_symbols.update({long_position.symbol, short_position.symbol})
                    if candidate:
                        candidates.append(candidate)
        return candidates, paired_symbols

    def _debit_spread_strategy(self, option_type: str, long_strike: float, short_strike: float) -> str | None:
        if option_type == "C" and long_strike < short_strike:
            return "call_debit_spread"
        if option_type == "P" and long_strike > short_strike:
            return "put_debit_spread"
        return None

    def _spread_exit_candidate(
        self,
        *,
        underlying: str,
        strategy: str,
        long_position: Position,
        short_position: Position,
    ) -> TradeCandidate | None:
        qty = int(min(abs(long_position.qty), abs(short_position.qty)))
        if qty < 1:
            return None

        long_current = self._current_price(long_position)
        short_current = self._current_price(short_position)
        if long_current is None or short_current is None or long_current <= 0 or short_current <= 0:
            return None
        if long_position.avg_entry_price is None or short_position.avg_entry_price is None:
            return None

        entry_debit = long_position.avg_entry_price - short_position.avg_entry_price
        current_credit = long_current - short_current
        if entry_debit <= 0 or current_credit <= 0:
            return None

        cost_basis = entry_debit * OPTION_MULTIPLIER * qty
        unrealized_pl = self._spread_unrealized_pl(
            long_position=long_position,
            short_position=short_position,
            entry_debit=entry_debit,
            current_credit=current_credit,
            qty=qty,
        )
        pnl_pct = (unrealized_pl / cost_basis) * 100
        spread_symbol = f"{underlying}_{strategy}"
        state = self._update_state(
            Position(
                symbol=spread_symbol,
                asset_class=AssetClass.OPTION,
                qty=qty,
                market_value=current_credit * OPTION_MULTIPLIER * qty,
                avg_entry_price=entry_debit,
                current_price=current_credit,
                unrealized_pl=unrealized_pl,
            ),
            current_credit,
        )
        metrics = self._spread_metrics(
            cost_basis=cost_basis,
            current_price=current_credit,
            pnl_pct=pnl_pct,
            state=state,
        )
        exit_reason = self._exit_reason(
            Position(
                symbol=spread_symbol,
                asset_class=AssetClass.OPTION,
                qty=qty,
                market_value=current_credit * OPTION_MULTIPLIER * qty,
                avg_entry_price=entry_debit,
                current_price=current_credit,
                unrealized_pl=unrealized_pl,
            ),
            metrics,
        )
        if not exit_reason:
            return None

        return TradeCandidate(
            symbol=spread_symbol,
            asset_class=AssetClass.OPTION,
            side=OrderSide.SELL,
            strategy=f"spread_{exit_reason['strategy']}",
            score=exit_reason["score"],
            entry_price=current_credit,
            rationale=[
                *exit_reason["rationale"],
                (
                    f"Managing {strategy} as one spread: sell-to-close {long_position.symbol} "
                    f"and buy-to-close {short_position.symbol}."
                ),
            ],
            metadata={
                "exit": True,
                "spread_exit": True,
                "spread_strategy": strategy,
                "exit_qty": qty,
                "underlying": underlying,
                "paired_symbols": [long_position.symbol, short_position.symbol],
                "legs": [
                    {
                        "symbol": long_position.symbol,
                        "ratio_qty": "1",
                        "side": "sell",
                        "position_intent": "sell_to_close",
                    },
                    {
                        "symbol": short_position.symbol,
                        "ratio_qty": "1",
                        "side": "buy",
                        "position_intent": "buy_to_close",
                    },
                ],
                "positions": [
                    long_position.model_dump(mode="json"),
                    short_position.model_dump(mode="json"),
                ],
                "position_state": state,
                "metrics": metrics,
            },
        )

    def _spread_unrealized_pl(
        self,
        *,
        long_position: Position,
        short_position: Position,
        entry_debit: float,
        current_credit: float,
        qty: int,
    ) -> float:
        if long_position.unrealized_pl is not None and short_position.unrealized_pl is not None:
            return long_position.unrealized_pl + short_position.unrealized_pl
        return (current_credit - entry_debit) * OPTION_MULTIPLIER * qty

    def _spread_metrics(self, *, cost_basis: float, current_price: float, pnl_pct: float, state: dict) -> dict:
        first_seen_at = datetime.fromisoformat(state["first_seen_at"])
        holding_days = max((datetime.now(UTC) - first_seen_at).days, 0)
        peak_price = float(state["peak_price"])
        trough_price = float(state["trough_price"])
        trailing_drawdown_pct = 0.0
        trailing_runup_pct = 0.0
        if peak_price > 0:
            trailing_drawdown_pct = ((peak_price - current_price) / peak_price) * 100
        if trough_price > 0:
            trailing_runup_pct = ((current_price - trough_price) / trough_price) * 100
        return {
            "current_price": current_price,
            "cost_basis": cost_basis,
            "pnl_pct": pnl_pct,
            "holding_days": holding_days,
            "peak_price": peak_price,
            "trough_price": trough_price,
            "trailing_drawdown_pct": trailing_drawdown_pct,
            "trailing_runup_pct": trailing_runup_pct,
        }

    def _current_price(self, position: Position) -> float | None:
        if position.current_price:
            return abs(position.current_price)
        if position.qty and position.market_value:
            return abs(position.market_value / position.qty)
        return position.avg_entry_price

    def _update_state(self, position: Position, current_price: float) -> dict:
        if self.audit:
            return self.audit.update_position_state(
                symbol=position.symbol,
                asset_class=position.asset_class.value,
                current_price=current_price,
            )

        now = datetime.now(UTC).isoformat()
        state = self._memory_state.get(position.symbol)
        if not state:
            state = {
                "symbol": position.symbol,
                "asset_class": position.asset_class.value,
                "first_seen_at": now,
                "last_seen_at": now,
                "peak_price": current_price,
                "trough_price": current_price,
            }
        else:
            state["last_seen_at"] = now
            state["peak_price"] = max(float(state["peak_price"]), current_price)
            state["trough_price"] = min(float(state["trough_price"]), current_price)
        self._memory_state[position.symbol] = state
        return state

    def _metrics(self, position: Position, current_price: float, state: dict) -> dict:
        cost_basis = self._cost_basis(position)
        pnl_pct = None
        if position.unrealized_pl is not None and cost_basis > 0:
            pnl_pct = (position.unrealized_pl / cost_basis) * 100
        elif position.avg_entry_price:
            direction = 1 if position.qty > 0 else -1
            pnl_pct = ((current_price - position.avg_entry_price) / position.avg_entry_price) * 100 * direction

        first_seen_at = datetime.fromisoformat(state["first_seen_at"])
        holding_days = max((datetime.now(UTC) - first_seen_at).days, 0)
        peak_price = float(state["peak_price"])
        trough_price = float(state["trough_price"])
        trailing_drawdown_pct = 0.0
        trailing_runup_pct = 0.0
        if peak_price > 0:
            trailing_drawdown_pct = ((peak_price - current_price) / peak_price) * 100
        if trough_price > 0:
            trailing_runup_pct = ((current_price - trough_price) / trough_price) * 100

        return {
            "current_price": current_price,
            "cost_basis": cost_basis,
            "pnl_pct": pnl_pct,
            "holding_days": holding_days,
            "peak_price": peak_price,
            "trough_price": trough_price,
            "trailing_drawdown_pct": trailing_drawdown_pct,
            "trailing_runup_pct": trailing_runup_pct,
        }

    def _cost_basis(self, position: Position) -> float:
        multiplier = 100 if position.asset_class == AssetClass.OPTION else 1
        if position.avg_entry_price:
            return abs(position.avg_entry_price * position.qty * multiplier)
        return abs(position.market_value)

    def _exit_reason(self, position: Position, metrics: dict) -> dict | None:
        if position.asset_class == AssetClass.OPTION and position.qty < 0:
            return self._short_option_exit_reason(position, metrics)

        pnl_pct = metrics["pnl_pct"]
        if pnl_pct is None:
            return None

        stop_loss_pct = (
            self.settings.position_manager.option_stop_loss_pct
            if position.asset_class == AssetClass.OPTION
            else self.settings.position_manager.stop_loss_pct
        )
        take_profit_pct = (
            self.settings.position_manager.option_take_profit_pct
            if position.asset_class == AssetClass.OPTION
            else self.settings.position_manager.take_profit_pct
        )

        if pnl_pct <= -stop_loss_pct:
            return {
                "strategy": "stop_loss_exit",
                "score": 100,
                "rationale": [
                    f"Position P/L is {pnl_pct:.2f}%, below stop loss threshold {-stop_loss_pct:.2f}%."
                ],
            }
        if pnl_pct >= take_profit_pct:
            return {
                "strategy": "take_profit_exit",
                "score": 95,
                "rationale": [
                    f"Position P/L is {pnl_pct:.2f}%, above take profit threshold {take_profit_pct:.2f}%."
                ],
            }

        if position.asset_class != AssetClass.OPTION and position.qty > 0:
            trailing_stop_pct = self.settings.position_manager.trailing_stop_pct
            if metrics["trailing_drawdown_pct"] >= trailing_stop_pct:
                return {
                    "strategy": "trailing_stop_exit",
                    "score": 90,
                    "rationale": [
                        f"Position fell {metrics['trailing_drawdown_pct']:.2f}% from tracked peak price."
                    ],
                }

        max_holding_days = self.settings.position_manager.max_holding_days
        if max_holding_days and metrics["holding_days"] >= max_holding_days:
            return {
                "strategy": "time_exit",
                "score": 80,
                "rationale": [
                    f"Position has been tracked for {metrics['holding_days']} days, meeting max holding period."
                ],
            }
        return None

    def _short_option_exit_reason(self, position: Position, metrics: dict) -> dict | None:
        pnl_pct = metrics["pnl_pct"]
        if pnl_pct is None:
            return None
        stop_loss_pct = self.settings.position_manager.option_stop_loss_pct
        take_profit_pct = self.settings.position_manager.option_take_profit_pct
        if pnl_pct <= -stop_loss_pct:
            return {
                "strategy": "short_option_stop_loss_exit",
                "score": 100,
                "rationale": [
                    f"Short option P/L is {pnl_pct:.2f}%, below stop loss threshold {-stop_loss_pct:.2f}%."
                ],
            }
        if pnl_pct >= take_profit_pct:
            return {
                "strategy": "short_option_take_profit_exit",
                "score": 95,
                "rationale": [
                    f"Short option P/L is {pnl_pct:.2f}%, above take profit threshold {take_profit_pct:.2f}%."
                ],
            }
        return None

    def _parse_option_symbol(self, symbol: str) -> dict | None:
        for index, char in enumerate(symbol):
            if not char.isdigit():
                continue
            if len(symbol) < index + 15:
                return None
            date_part = symbol[index : index + 6]
            option_type = symbol[index + 6 : index + 7]
            strike_part = symbol[index + 7 : index + 15]
            if not date_part.isdigit() or option_type not in {"C", "P"} or not strike_part.isdigit():
                return None
            try:
                expiration = datetime.strptime(date_part, "%y%m%d").date().isoformat()
            except ValueError:
                expiration = None
            return {
                "underlying": symbol[:index],
                "expiration": expiration,
                "type": option_type,
                "strike": int(strike_part) / 1000,
            }
        return None
