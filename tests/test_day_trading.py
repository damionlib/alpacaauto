from datetime import UTC, datetime, timedelta

from trading_agent.catalyst.models import CatalystPrediction
from trading_agent.config import Settings
from trading_agent.day_trading import DayTradingEngine
from trading_agent.models import AccountSnapshot, AssetClass, MarketSnapshot, NewsItem, OrderSide, Position, ResearchSnapshot
from trading_agent.risk import RiskEngine


def _prediction(score: float = 88, direction: str = "bullish") -> CatalystPrediction:
    return CatalystPrediction(
        symbol="AAPL",
        asset_class="equity",
        direction=direction,
        prediction_score=score,
        bullish_score=score if direction == "bullish" else 100 - score,
        confidence="high",
        entry_allowed=True,
        market_regime="risk_on",
    )


def _market(price: float = 130) -> MarketSnapshot:
    return MarketSnapshot(
        symbol="AAPL",
        asset_class=AssetClass.EQUITY,
        price=price,
        closes=[90 + index for index in range(60)],
        metadata={"vwap": 125, "relative_volume": 2.0, "spread_pct": 0.05},
    )


def _research() -> ResearchSnapshot:
    return ResearchSnapshot(
        symbol="AAPL",
        news=[
            NewsItem(title="AAPL announces earnings beat and raises guidance"),
            NewsItem(title="Analyst upgrade follows record profit"),
        ],
        sec_summary={"latest_net_income": {"value": 1_000_000}, "recent_filings": [{"form": "10-Q"}]},
    )


def test_day_trading_engine_generates_bullish_entry() -> None:
    settings = Settings.model_validate({"day_trading": {"enabled": True}})
    engine = DayTradingEngine(settings)

    candidate, signal = engine.evaluate_entry(_market(price=150), _research(), _prediction(), [], trades_used_today=0)

    assert candidate is not None
    assert candidate.strategy == "day_trade_entry"
    assert candidate.side == OrderSide.BUY
    assert candidate.metadata["day_trade"] is True
    assert signal["status"] == "generated"


def test_day_trading_engine_blocks_weak_intraday_signal() -> None:
    settings = Settings.model_validate({"day_trading": {"enabled": True}})
    engine = DayTradingEngine(settings)
    market = _market(price=100).model_copy(update={"metadata": {"vwap": 125, "relative_volume": 0.7, "spread_pct": 0.05}})

    candidate, signal = engine.evaluate_entry(market, _research(), _prediction(), [], trades_used_today=0)

    assert candidate is None
    assert signal["status"] == "blocked"
    assert "Intraday score" in signal["reason"]


def test_day_trading_exit_triggers_on_stop_loss() -> None:
    settings = Settings.model_validate({"day_trading": {"enabled": True}})
    engine = DayTradingEngine(settings)
    entry_event = {"id": 10, "created_at": (datetime.now(UTC) - timedelta(minutes=20)).isoformat()}
    position = Position(
        symbol="AAPL",
        asset_class=AssetClass.EQUITY,
        qty=10,
        market_value=1_280,
        avg_entry_price=130,
        current_price=128,
        unrealized_pl=-20,
    )

    candidate, signal = engine.evaluate_exit(
        position,
        _market(price=128),
        _research(),
        _prediction(),
        entry_event=entry_event,
    )

    assert candidate is not None
    assert candidate.strategy == "day_trade_exit"
    assert candidate.metadata["exit"] is True
    assert "floor" in signal["reason"].lower() or "stop" in signal["reason"].lower()


def test_day_trading_exit_trailing_stop_from_peak() -> None:
    settings = Settings.model_validate({
        "day_trading": {"enabled": True, "profit_protect_pct": 0.4, "profit_trail_pct": 0.5},
    })
    engine = DayTradingEngine(settings)
    entry_event = {"id": 10, "created_at": (datetime.now(UTC) - timedelta(minutes=50)).isoformat()}
    position = Position(
        symbol="AAPL", asset_class=AssetClass.EQUITY, qty=10,
        market_value=1_344, avg_entry_price=130, current_price=131.1,
    )
    engine.evaluate_exit(position, _market(price=131.1), _research(), _prediction(), entry_event=entry_event)
    assert engine._peak_pnl["AAPL"] >= 0.84

    position_drop = Position(
        symbol="AAPL", asset_class=AssetClass.EQUITY, qty=10,
        market_value=1_303, avg_entry_price=130, current_price=130.3,
    )
    candidate, signal = engine.evaluate_exit(
        position_drop, _market(price=130.3), _research(), _prediction(), entry_event=entry_event,
    )
    assert candidate is not None
    assert "Trailing stop" in signal["reason"]
    assert "AAPL" not in engine._peak_pnl


def test_day_trading_exit_trailing_stop_holds_when_above_floor() -> None:
    settings = Settings.model_validate({
        "day_trading": {"enabled": True, "profit_protect_pct": 0.4, "profit_trail_pct": 0.5},
    })
    engine = DayTradingEngine(settings)
    entry_event = {"id": 10, "created_at": (datetime.now(UTC) - timedelta(minutes=20)).isoformat()}
    position = Position(
        symbol="AAPL", asset_class=AssetClass.EQUITY, qty=10,
        market_value=1_310, avg_entry_price=130, current_price=130.9,
    )
    engine.evaluate_exit(position, _market(price=130.9), _research(), _prediction(), entry_event=entry_event)
    position_still_ok = Position(
        symbol="AAPL", asset_class=AssetClass.EQUITY, qty=10,
        market_value=1_306, avg_entry_price=130, current_price=130.6,
    )
    candidate, signal = engine.evaluate_exit(
        position_still_ok, _market(price=130.6), _research(), _prediction(), entry_event=entry_event,
    )
    assert candidate is None
    assert signal["status"] == "held"


def test_day_trading_exit_profit_reversal() -> None:
    settings = Settings.model_validate({"day_trading": {"enabled": True}})
    engine = DayTradingEngine(settings)
    entry_event = {"id": 10, "created_at": (datetime.now(UTC) - timedelta(minutes=40)).isoformat()}
    position_up = Position(
        symbol="AAPL", asset_class=AssetClass.EQUITY, qty=10,
        market_value=1_306, avg_entry_price=130, current_price=130.5,
    )
    engine.evaluate_exit(position_up, _market(price=130.5), _research(), _prediction(), entry_event=entry_event)

    position_down = Position(
        symbol="AAPL", asset_class=AssetClass.EQUITY, qty=10,
        market_value=1_298, avg_entry_price=130, current_price=129.9,
    )
    candidate, signal = engine.evaluate_exit(
        position_down, _market(price=129.9), _research(), _prediction(), entry_event=entry_event,
    )
    assert candidate is not None
    assert "Profit reversal" in signal["reason"]


def test_day_trading_exit_stale_negative() -> None:
    settings = Settings.model_validate({
        "day_trading": {"enabled": True, "stale_negative_minutes": 30},
    })
    engine = DayTradingEngine(settings)
    entry_event = {"id": 10, "created_at": (datetime.now(UTC) - timedelta(minutes=35)).isoformat()}
    position = Position(
        symbol="AAPL", asset_class=AssetClass.EQUITY, qty=10,
        market_value=1_296, avg_entry_price=130, current_price=129.7,
    )
    candidate, signal = engine.evaluate_exit(
        position, _market(price=129.7), _research(), _prediction(), entry_event=entry_event,
    )
    assert candidate is not None
    assert "Stale negative" in signal["reason"]


def test_day_trading_exit_stale_negative_spares_if_was_profitable() -> None:
    settings = Settings.model_validate({
        "day_trading": {"enabled": True, "stale_negative_minutes": 30},
    })
    engine = DayTradingEngine(settings)
    entry_event = {"id": 10, "created_at": (datetime.now(UTC) - timedelta(minutes=35)).isoformat()}
    position_up = Position(
        symbol="AAPL", asset_class=AssetClass.EQUITY, qty=10,
        market_value=1_310, avg_entry_price=130, current_price=130.5,
    )
    engine.evaluate_exit(position_up, _market(price=130.5), _research(), _prediction(), entry_event=entry_event)

    position_down = Position(
        symbol="AAPL", asset_class=AssetClass.EQUITY, qty=10,
        market_value=1_298, avg_entry_price=130, current_price=129.9,
    )
    candidate, signal = engine.evaluate_exit(
        position_down, _market(price=129.9), _research(), _prediction(), entry_event=entry_event,
    )
    assert candidate is not None
    assert "Stale negative" not in signal["reason"]
    assert "Profit reversal" in signal["reason"]


def test_day_trading_exit_time_decay_60min() -> None:
    settings = Settings.model_validate({
        "day_trading": {"enabled": True, "time_decay_minutes": 60},
    })
    engine = DayTradingEngine(settings)
    entry_event = {"id": 10, "created_at": (datetime.now(UTC) - timedelta(minutes=65)).isoformat()}
    position_peak = Position(
        symbol="AAPL", asset_class=AssetClass.EQUITY, qty=10,
        market_value=1_306, avg_entry_price=130, current_price=130.5,
    )
    engine.evaluate_exit(position_peak, _market(price=130.5), _research(), _prediction(), entry_event=entry_event)
    position = Position(
        symbol="AAPL", asset_class=AssetClass.EQUITY, qty=10,
        market_value=1_302, avg_entry_price=130, current_price=130.2,
    )
    candidate, signal = engine.evaluate_exit(
        position, _market(price=130.2), _research(), _prediction(), entry_event=entry_event,
    )
    assert candidate is not None
    assert "Time decay" in signal["reason"]
    assert "need >= 0.3%" in signal["reason"]


def test_day_trading_exit_time_decay_90min() -> None:
    settings = Settings.model_validate({
        "day_trading": {"enabled": True, "time_decay_minutes": 60},
    })
    engine = DayTradingEngine(settings)
    entry_event = {"id": 10, "created_at": (datetime.now(UTC) - timedelta(minutes=95)).isoformat()}
    position = Position(
        symbol="AAPL", asset_class=AssetClass.EQUITY, qty=10,
        market_value=1_304, avg_entry_price=130, current_price=130.4,
    )
    candidate, signal = engine.evaluate_exit(
        position, _market(price=130.4), _research(), _prediction(), entry_event=entry_event,
    )
    assert candidate is not None
    assert "Time decay" in signal["reason"]
    assert "need >= 0.5%" in signal["reason"]


def test_day_trading_exit_time_decay_holds_when_profitable() -> None:
    settings = Settings.model_validate({
        "day_trading": {"enabled": True, "time_decay_minutes": 60},
    })
    engine = DayTradingEngine(settings)
    entry_event = {"id": 10, "created_at": (datetime.now(UTC) - timedelta(minutes=65)).isoformat()}
    position = Position(
        symbol="AAPL", asset_class=AssetClass.EQUITY, qty=10,
        market_value=1_310, avg_entry_price=130, current_price=130.7,
    )
    candidate, signal = engine.evaluate_exit(
        position, _market(price=130.7), _research(), _prediction(), entry_event=entry_event,
    )
    assert candidate is None
    assert signal["status"] == "held"


def test_day_trading_entry_computes_adr_ceiling_floor() -> None:
    settings = Settings.model_validate({
        "day_trading": {"enabled": True, "ceiling_adr_fraction": 0.55, "floor_adr_fraction": 0.40},
    })
    engine = DayTradingEngine(settings)
    market = _market(price=150)
    market.metadata["adr_pct"] = 2.5

    candidate, signal = engine.evaluate_entry(market, _research(), _prediction(), [], trades_used_today=0)

    assert candidate is not None
    assert candidate.metadata["ceiling_pct"] == round(2.5 * 0.55, 4)
    assert candidate.metadata["floor_pct"] == round(2.5 * 0.40, 4)
    assert candidate.take_profit_price > candidate.entry_price
    assert candidate.stop_price < candidate.entry_price


def test_day_trading_entry_caps_adr_targets_at_config_max() -> None:
    settings = Settings.model_validate({
        "day_trading": {
            "enabled": True,
            "ceiling_adr_fraction": 0.55,
            "floor_adr_fraction": 0.40,
            "take_profit_pct": 1.0,
            "stop_loss_pct": 0.5,
        },
    })
    engine = DayTradingEngine(settings)
    market = _market(price=150)
    market.metadata["adr_pct"] = 5.0

    candidate, _ = engine.evaluate_entry(market, _research(), _prediction(), [], trades_used_today=0)

    assert candidate.metadata["ceiling_pct"] == 1.0
    assert candidate.metadata["floor_pct"] == 0.5


def test_day_trading_exit_adr_ceiling_hit() -> None:
    settings = Settings.model_validate({
        "day_trading": {"enabled": True, "ceiling_adr_fraction": 0.55, "floor_adr_fraction": 0.40},
    })
    engine = DayTradingEngine(settings)
    entry_event = {
        "id": 10,
        "created_at": (datetime.now(UTC) - timedelta(minutes=30)).isoformat(),
        "payload": {"intent": {"metadata": {"ceiling_pct": 1.375, "floor_pct": 1.0}}},
    }
    position = Position(
        symbol="AAPL", asset_class=AssetClass.EQUITY, qty=10,
        market_value=1_320, avg_entry_price=130, current_price=131.8,
    )
    candidate, signal = engine.evaluate_exit(
        position, _market(price=131.8), _research(), _prediction(), entry_event=entry_event,
    )
    assert candidate is not None
    assert "ceiling" in signal["reason"].lower()


def test_day_trading_exit_adr_floor_hit() -> None:
    settings = Settings.model_validate({
        "day_trading": {"enabled": True, "ceiling_adr_fraction": 0.55, "floor_adr_fraction": 0.40},
    })
    engine = DayTradingEngine(settings)
    entry_event = {
        "id": 10,
        "created_at": (datetime.now(UTC) - timedelta(minutes=30)).isoformat(),
        "payload": {"intent": {"metadata": {"ceiling_pct": 1.375, "floor_pct": 1.0}}},
    }
    position = Position(
        symbol="AAPL", asset_class=AssetClass.EQUITY, qty=10,
        market_value=1_286, avg_entry_price=130, current_price=128.6,
    )
    candidate, signal = engine.evaluate_exit(
        position, _market(price=128.6), _research(), _prediction(), entry_event=entry_event,
    )
    assert candidate is not None
    assert "floor" in signal["reason"].lower()


def test_day_trading_exit_holds_patiently_within_range() -> None:
    settings = Settings.model_validate({
        "day_trading": {
            "enabled": True,
            "profit_protect_pct": 0,
            "stale_negative_minutes": 0,
            "time_decay_minutes": 0,
            "max_position_minutes": 480,
        },
    })
    engine = DayTradingEngine(settings)
    entry_event = {
        "id": 10,
        "created_at": (datetime.now(UTC) - timedelta(minutes=90)).isoformat(),
        "payload": {"intent": {"metadata": {"ceiling_pct": 1.375, "floor_pct": 1.0}}},
    }
    position = Position(
        symbol="AAPL", asset_class=AssetClass.EQUITY, qty=10,
        market_value=1_296, avg_entry_price=130, current_price=129.7,
    )
    candidate, signal = engine.evaluate_exit(
        position, _market(price=129.7), _research(), _prediction(), entry_event=entry_event,
    )
    assert candidate is None
    assert signal["status"] == "held"


def _partial_entry_event(qty: float = 10, minutes_ago: int = 60) -> dict:
    return {
        "id": 10,
        "created_at": (datetime.now(UTC) - timedelta(minutes=minutes_ago)).isoformat(),
        "payload": {
            "intent": {
                "qty": qty,
                "metadata": {"ceiling_pct": 1.375, "floor_pct": 1.0, "partial_pct": 0.9},
            }
        },
    }


def test_day_trading_partial_exit_at_target() -> None:
    settings = Settings.model_validate({"day_trading": {"enabled": True}})
    engine = DayTradingEngine(settings)
    position = Position(
        symbol="AAPL", asset_class=AssetClass.EQUITY, qty=10,
        market_value=1_313, avg_entry_price=130, current_price=131.3,
    )

    candidate, signal = engine.evaluate_exit(
        position, _market(price=131.3), _research(), _prediction(),
        entry_event=_partial_entry_event(qty=10),
    )

    assert candidate is not None
    assert candidate.metadata["partial_exit"] is True
    assert candidate.metadata["exit_qty"] == 5
    assert candidate.metadata["remainder_qty"] == 5
    assert candidate.metadata["remainder_stop_price"] == 130.0
    assert candidate.metadata["remainder_take_profit_price"] == round(130 * 1.01375, 2)
    assert "Partial profit-take" in signal["reason"]
    # position stays open, so peak tracking must survive the partial
    assert "AAPL" in engine._peak_pnl


def test_day_trading_partial_remainder_breakeven_stop() -> None:
    settings = Settings.model_validate({"day_trading": {"enabled": True}})
    engine = DayTradingEngine(settings)
    # qty 5 vs entry qty 10 -> the partial already banked; remainder is guarded at breakeven
    position = Position(
        symbol="AAPL", asset_class=AssetClass.EQUITY, qty=5,
        market_value=649, avg_entry_price=130, current_price=129.9,
    )

    candidate, signal = engine.evaluate_exit(
        position, _market(price=129.9), _research(), _prediction(),
        entry_event=_partial_entry_event(qty=10),
    )

    assert candidate is not None
    assert candidate.metadata.get("partial_exit") is None
    assert "Breakeven stop" in signal["reason"]


def test_day_trading_partial_remainder_rides_to_ceiling() -> None:
    settings = Settings.model_validate({"day_trading": {"enabled": True}})
    engine = DayTradingEngine(settings)
    position = Position(
        symbol="AAPL", asset_class=AssetClass.EQUITY, qty=5,
        market_value=660, avg_entry_price=130, current_price=131.9,
    )

    candidate, signal = engine.evaluate_exit(
        position, _market(price=131.9), _research(), _prediction(),
        entry_event=_partial_entry_event(qty=10),
    )

    assert candidate is not None
    assert "ceiling" in signal["reason"].lower()
    assert candidate.metadata["exit_qty"] == 5


def test_day_trading_partial_exit_disabled_by_config() -> None:
    settings = Settings.model_validate({
        "day_trading": {"enabled": True, "partial_exit_size": 0},
    })
    engine = DayTradingEngine(settings)
    position = Position(
        symbol="AAPL", asset_class=AssetClass.EQUITY, qty=10,
        market_value=1_313, avg_entry_price=130, current_price=131.3,
    )

    candidate, signal = engine.evaluate_exit(
        position, _market(price=131.3), _research(), _prediction(),
        entry_event=_partial_entry_event(qty=10),
    )

    assert candidate is None
    assert signal["status"] == "held"


def test_day_trading_partial_exit_skipped_for_single_share() -> None:
    settings = Settings.model_validate({"day_trading": {"enabled": True}})
    engine = DayTradingEngine(settings)
    position = Position(
        symbol="AAPL", asset_class=AssetClass.EQUITY, qty=1,
        market_value=131, avg_entry_price=130, current_price=131.3,
    )

    candidate, signal = engine.evaluate_exit(
        position, _market(price=131.3), _research(), _prediction(),
        entry_event=_partial_entry_event(qty=1),
    )

    assert candidate is None
    assert signal["status"] == "held"


def test_day_trade_setup_risk_multiplier_halves_orb_size() -> None:
    settings = Settings.model_validate({"day_trading": {"enabled": True}})
    engine = RiskEngine(settings)
    account = AccountSnapshot(equity=100_000, cash=100_000, buying_power=100_000, last_equity=100_000)

    def _candidate(setup: str) -> "TradeCandidate":
        from trading_agent.models import TradeCandidate
        return TradeCandidate(
            symbol="GE",
            asset_class=AssetClass.EQUITY,
            side=OrderSide.BUY,
            strategy="day_trade_entry",
            score=85,
            entry_price=150,
            stop_price=148.5,
            metadata={"day_trade": True, "day_trade_entry": True, "setup": setup},
        )

    trend = engine.evaluate(_candidate("trend_continuation"), account, [])
    orb = engine.evaluate(_candidate("opening_range_break"), account, [])

    assert trend.approved and orb.approved
    assert trend.intent is not None and orb.intent is not None
    assert orb.intent.qty <= trend.intent.qty / 2 + 1
    assert orb.intent.qty >= 1


def test_day_trade_risk_uses_tighter_position_sizing() -> None:
    settings = Settings.model_validate(
        {
            "day_trading": {
                "enabled": True,
                "risk_per_trade_pct": 0.25,
                "max_position_pct": 4.0,
            }
        }
    )
    engine = DayTradingEngine(settings)
    candidate, _signal = engine.evaluate_entry(_market(price=150), _research(), _prediction(), [], trades_used_today=0)

    decision = RiskEngine(settings).evaluate(
        candidate,
        AccountSnapshot(equity=100_000, cash=100_000, buying_power=100_000, last_equity=100_000),
        [],
    )

    assert decision.approved
    assert decision.intent is not None
    assert decision.intent.qty <= 250
