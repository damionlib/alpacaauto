from __future__ import annotations

from datetime import UTC, datetime

from trading_agent.audit import AuditStore
from trading_agent.config import Settings
from trading_agent.models import AssetClass, OrderSide, Position, TradeCandidate


class PositionManager:
    def __init__(self, settings: Settings, audit: AuditStore | None = None) -> None:
        self.settings = settings
        self.audit = audit
        self._memory_state: dict[str, dict] = {}

    def evaluate(self, positions: list[Position]) -> list[TradeCandidate]:
        if not self.settings.position_manager.enabled:
            return []

        candidates: list[TradeCandidate] = []
        for position in positions:
            if position.qty == 0:
                continue
            if position.asset_class == AssetClass.OPTION and not self.settings.position_manager.manage_options:
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
