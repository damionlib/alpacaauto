from datetime import UTC, date, datetime, timedelta

import pytest

from trading_agent.agent import TradingAgent
from trading_agent.audit import AuditStore
from trading_agent.brokers.alpaca import AlpacaBroker
from trading_agent.catalyst.service import CatalystEngine
from trading_agent.config import Settings
from trading_agent.indicators import atr
from trading_agent.models import (
    AssetClass,
    MarketSnapshot,
    NewsItem,
    OrderIntent,
    OrderSide,
    OrderType,
    Position,
    ResearchSnapshot,
    TimeInForce,
)
from trading_agent.position_manager import PositionManager, swing_stop_level, trading_days_between
from trading_agent.strategies.momentum import MomentumStrategy
from trading_agent.text_terms import count_terms, matched_terms

DATA = "https://data.alpaca.markets/v2/stocks/bars"


# --- indicators / text matching ---------------------------------------------------


def test_atr_is_mean_true_range_including_gaps() -> None:
    highs = [10, 11, 12, 13]
    lows = [9, 10, 11, 12]
    closes = [9.5, 10.5, 11.5, 12.5]
    # TRs over the last 2 bars: max(1, |12-10.5|, |11-10.5|)=1.5 and 1.5
    assert atr(highs, lows, closes, period=2) == pytest.approx(1.5)
    assert atr(highs, lows, closes, period=4) is None


def test_whole_word_matching_ignores_embedded_words() -> None:
    text = "Software firm will accelerate federal execution; admission fees rise"
    assert count_terms(text, {"war", "rate", "fed", "cut", "miss"}) == 0
    assert matched_terms("Talks of war and a rate hike", {"war", "rate hike"}) == ["rate hike", "war"]


def _market(price: float = 130, *, with_hl: bool = True) -> MarketSnapshot:
    closes = [90 + index * 0.66 for index in range(60)]
    metadata = {"volumes": [1_000_000] * 60}
    if with_hl:
        metadata["highs"] = [close + 1.0 for close in closes]
        metadata["lows"] = [close - 1.0 for close in closes]
    return MarketSnapshot(symbol="AAPL", asset_class=AssetClass.EQUITY, price=price, closes=closes, metadata=metadata)


def _research(*titles: str) -> ResearchSnapshot:
    return ResearchSnapshot(
        symbol="AAPL",
        news=[NewsItem(title=title) for title in titles],
        sec_summary={"latest_net_income": {"value": 1_000_000}, "latest_revenue": {"value": 5}},
    )


def test_catalyst_does_not_flag_software_or_corporate_as_macro_risk() -> None:
    engine = CatalystEngine(Settings())
    prediction = engine.evaluate(
        _market(),
        _research(
            "Software company announces record results and buyback",
            "Corporate partnership could accelerate growth",
            "Company beats estimates on strong software demand",
        ),
    )
    assert "event_risk_macro_or_geopolitical" not in prediction.risks
    assert "rumor_only_catalyst" not in prediction.risks


def test_catalyst_flags_unconfirmed_rumor_and_heavy_macro() -> None:
    engine = CatalystEngine(Settings())
    rumor = engine.evaluate(_market(), _research("AAPL reportedly in talks to buy a startup, people familiar say"))
    assert "rumor_only_catalyst" in rumor.risks
    macro = engine.evaluate(
        _market(), _research("Fed weighs rate hike as inflation and tariffs rattle markets; war fears grow")
    )
    assert "event_risk_macro_or_geopolitical" in macro.risks


# --- strategy stops ---------------------------------------------------------------


def test_momentum_uses_atr_stop_and_no_fixed_target() -> None:
    settings = Settings.model_validate({"strategy": {"enabled": ["equity_momentum"], "min_signal_score": 0}})
    market = _market(price=130)
    candidates = MomentumStrategy(settings).evaluate(market, _research("Company announces results"))
    assert len(candidates) == 1
    candidate = candidates[0]
    expected_atr = atr(market.metadata["highs"], market.metadata["lows"], market.closes, 14)
    assert candidate.metadata["atr"] == pytest.approx(expected_atr, rel=1e-3)
    assert candidate.take_profit_price is None
    stop_pct = (1 - candidate.stop_price / 130) * 100
    assert stop_pct == pytest.approx(max(3.0, min(12.0, 2.5 * expected_atr / 130 * 100)), abs=0.01)
    assert candidate.metadata["initial_stop"] == candidate.stop_price


def test_momentum_falls_back_to_vol_stop_without_high_low_history() -> None:
    settings = Settings.model_validate({"strategy": {"enabled": ["equity_momentum"], "min_signal_score": 0}})
    candidate = MomentumStrategy(settings).evaluate(_market(with_hl=False), _research("Company announces results"))[0]
    assert "atr" not in candidate.metadata
    assert candidate.take_profit_price is not None


def test_swing_entry_without_target_is_a_plain_limit_order() -> None:
    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker.trading_base_url = "https://paper-api.alpaca.markets"
    sent = {}

    async def fake_request(method, url, *, params=None, json=None):
        sent.update(json or {})
        return {"id": "1"}

    broker._request = fake_request
    import asyncio

    asyncio.run(
        broker.submit_order(
            OrderIntent(
                symbol="AAPL", asset_class=AssetClass.EQUITY, side=OrderSide.BUY, qty=10,
                order_type=OrderType.LIMIT, time_in_force=TimeInForce.DAY, limit_price=130.5,
                stop_loss_price=124.0, take_profit_price=None,
            )
        )
    )
    # A DAY bracket's stop leg is cancelled at the close; swing protection is a
    # separate GTC stop placed by the reconciler instead.
    assert "order_class" not in sent
    assert "stop_loss" not in sent


# --- broker data layer ------------------------------------------------------------


@pytest.mark.anyio
async def test_daily_bars_follow_pagination_and_use_adjusted_sip(respx_mock) -> None:
    def bar(close):
        return {"o": close, "h": close + 1, "l": close - 1, "c": close, "v": 100, "t": "2026-01-01T05:00:00Z"}

    route = respx_mock.get(DATA).mock(
        side_effect=[
            __import__("httpx").Response(200, json={"bars": {"AAA": [bar(1), bar(2)]}, "next_page_token": "p2"}),
            __import__("httpx").Response(200, json={"bars": {"AAA": [bar(3)], "BBB": [bar(9)]}, "next_page_token": None}),
        ]
    )
    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker.settings = Settings()
    broker.headers = {}

    series = await broker.get_daily_bars(["AAA", "BBB"])

    assert series["AAA"]["closes"] == [1.0, 2.0, 3.0]
    assert series["BBB"]["highs"] == [10.0]
    first, second = route.calls[0].request.url.params, route.calls[1].request.url.params
    assert first["feed"] == "sip" and first["adjustment"] == "all" and "end" in first
    assert "page_token" not in first and second["page_token"] == "p2"


# --- position manager: ratcheting ATR stop ----------------------------------------


def test_swing_stop_level_ratchets_initial_breakeven_chandelier() -> None:
    kwargs = {"entry_price": 100, "atr_value": 2, "initial_stop": 95,
              "breakeven_trigger_atr": 1.5, "chandelier_atr_multiple": 3}
    assert swing_stop_level(peak_price=101, **kwargs) == (95, "initial_atr_stop")
    assert swing_stop_level(peak_price=103, **kwargs) == (100, "breakeven_stop")
    assert swing_stop_level(peak_price=110, **kwargs) == (104, "chandelier_stop")


def test_trading_days_between_skips_weekends() -> None:
    assert trading_days_between(date(2026, 11, 6), date(2026, 11, 9)) == 1  # Fri -> Mon
    assert trading_days_between(date(2026, 11, 2), date(2026, 11, 16)) == 10


def _store_with_entry(tmp_path, *, atr_value=2.0, initial_stop=95.0, peak=100.0) -> AuditStore:
    store = AuditStore(tmp_path / "audit.sqlite3")
    store.record_event(
        cycle_id=None, event_type="order",
        payload={"intent": {"symbol": "AAPL", "metadata": {"atr": atr_value, "initial_stop": initial_stop}}},
        symbol="AAPL", strategy="equity_momentum", status="submitted",
    )
    now = datetime.now(UTC).isoformat()
    with store._connect() as connection:
        connection.execute(
            "insert into position_state (symbol, asset_class, first_seen_at, last_seen_at, peak_price, trough_price)"
            " values (?, ?, ?, ?, ?, ?)",
            ("AAPL", "equity", now, now, peak, 90),
        )
    return store


def _position(price: float) -> Position:
    return Position(
        symbol="AAPL", asset_class=AssetClass.EQUITY, qty=10, market_value=price * 10,
        avg_entry_price=100, current_price=price, unrealized_pl=(price - 100) * 10,
    )


def test_atr_mode_holds_through_gain_beyond_old_take_profit(tmp_path) -> None:
    manager = PositionManager(Settings(), _store_with_entry(tmp_path, peak=112))
    # +11% would have hit the old fixed 8% take-profit; the trail is at 106.
    assert manager.evaluate([_position(111)]) == []


def test_atr_mode_exits_on_chandelier_trail(tmp_path) -> None:
    manager = PositionManager(Settings(), _store_with_entry(tmp_path, peak=112))
    [candidate] = manager.evaluate([_position(105.5)])
    assert candidate.strategy == "chandelier_stop_exit"


def test_atr_mode_exits_on_initial_atr_stop(tmp_path) -> None:
    manager = PositionManager(Settings(), _store_with_entry(tmp_path))
    [candidate] = manager.evaluate([_position(94.9)])
    assert candidate.strategy == "atr_stop_exit"


def test_positions_without_entry_atr_use_legacy_percent_rules(tmp_path) -> None:
    manager = PositionManager(Settings(), AuditStore(tmp_path / "audit.sqlite3"))
    [candidate] = manager.evaluate([_position(113)])  # default take_profit_pct is 12
    assert candidate.strategy == "take_profit_exit"


# --- agent: broker-side protection reconciler and entry window --------------------


class ProtectionBroker:
    def __init__(self, open_orders=None) -> None:
        self.open_orders = open_orders or []
        self.submitted: list[OrderIntent] = []
        self.replaced: list[dict] = []

    async def get_open_orders(self):
        return self.open_orders

    async def submit_order(self, intent):
        self.submitted.append(intent)
        return {"id": "stop-1", "status": "accepted"}

    async def replace_order(self, order_id, *, qty=None, stop_price=None, limit_price=None):
        self.replaced.append({"id": order_id, "qty": qty, "stop_price": stop_price})
        return {"id": "stop-2", "status": "accepted"}


class SilentConsole:
    def print(self, *args, **kwargs) -> None:
        return None


def _agent(broker, store) -> TradingAgent:
    agent = TradingAgent.__new__(TradingAgent)
    agent.settings = Settings()
    agent.console = SilentConsole()
    agent.audit = store
    agent.broker = broker
    agent.position_manager = PositionManager(agent.settings, store)
    return agent


def _resting_stop(stop: float, qty: int = 10) -> dict:
    return {"id": "rest-1", "symbol": "AAPL", "side": "sell", "type": "stop", "qty": str(qty), "stop_price": str(stop)}


@pytest.mark.anyio
async def test_reconciler_places_gtc_stop_for_unprotected_swing_position(tmp_path) -> None:
    broker = ProtectionBroker()
    await _agent(broker, _store_with_entry(tmp_path))._reconcile_swing_protection(
        [_position(101)], None, skip_symbols=set()
    )
    [intent] = broker.submitted
    assert intent.order_type == OrderType.STOP and intent.time_in_force == TimeInForce.GTC
    assert intent.stop_price == 95 and intent.qty == 10


@pytest.mark.anyio
async def test_reconciler_raises_stop_with_the_trail_but_never_lowers(tmp_path) -> None:
    broker = ProtectionBroker([_resting_stop(95)])
    await _agent(broker, _store_with_entry(tmp_path, peak=112))._reconcile_swing_protection(
        [_position(111)], None, skip_symbols=set()
    )
    assert broker.replaced == [{"id": "rest-1", "qty": None, "stop_price": 106}]

    higher = ProtectionBroker([_resting_stop(107)])
    await _agent(higher, _store_with_entry_dir(tmp_path, "b", 112))._reconcile_swing_protection(
        [_position(111)], None, skip_symbols=set()
    )
    assert higher.replaced == [] and higher.submitted == []


def _store_with_entry_dir(tmp_path, name: str, peak: float) -> AuditStore:
    directory = tmp_path / name
    directory.mkdir()
    return _store_with_entry(directory, peak=peak)


@pytest.mark.anyio
async def test_reconciler_resizes_stop_after_quantity_change(tmp_path) -> None:
    broker = ProtectionBroker([_resting_stop(95, qty=20)])
    await _agent(broker, _store_with_entry(tmp_path))._reconcile_swing_protection(
        [_position(101)], None, skip_symbols=set()
    )
    assert broker.replaced == [{"id": "rest-1", "qty": 10, "stop_price": None}]


@pytest.mark.anyio
async def test_reconciler_skips_day_trades_and_working_exits(tmp_path) -> None:
    store = _store_with_entry(tmp_path)
    skipped = ProtectionBroker()
    await _agent(skipped, store)._reconcile_swing_protection([_position(101)], None, skip_symbols={"AAPL"})
    assert skipped.submitted == []

    working_exit = {"id": "x", "symbol": "AAPL", "side": "sell", "type": "market", "qty": "10"}
    busy = ProtectionBroker([working_exit])
    await _agent(busy, store)._reconcile_swing_protection([_position(101)], None, skip_symbols=set())
    assert busy.submitted == [] and busy.replaced == []


@pytest.mark.anyio
async def test_reconciler_does_not_place_stop_above_market(tmp_path) -> None:
    broker = ProtectionBroker()
    await _agent(broker, _store_with_entry(tmp_path))._reconcile_swing_protection(
        [_position(95.1)], None, skip_symbols=set()
    )
    assert broker.submitted == []


def test_swing_entry_window(tmp_path) -> None:
    agent = _agent(ProtectionBroker(), None)
    agent.settings = Settings.model_validate({"strategy": {"swing_entry_window_minutes": 60}})
    close = datetime(2026, 11, 5, 21, 0, tzinfo=UTC)
    clock = {"is_open": True, "next_close": close.isoformat()}
    assert agent._evaluate_swing_entry_window(clock, close - timedelta(minutes=120))[0] is False
    assert agent._evaluate_swing_entry_window(clock, close - timedelta(minutes=45))[0] is True
    assert agent._evaluate_swing_entry_window(clock, close - timedelta(minutes=5))[0] is False
    assert agent._evaluate_swing_entry_window({"is_open": False}, close)[0] is False
    agent.settings = Settings()
    assert agent._evaluate_swing_entry_window(None, close)[0] is True


def test_screener_orders_saturated_scores_by_unclamped_strength() -> None:
    from trading_agent.screener.service import MarketScreener

    screener = MarketScreener(Settings(), broker=None)

    def snap(symbol: str, slope: float) -> MarketSnapshot:
        closes = [100 * (1 + slope) ** index for index in range(60)]
        return MarketSnapshot(
            symbol=symbol, asset_class=AssetClass.EQUITY, price=closes[-1], closes=closes,
            metadata={"volumes": [5_000_000] * 60},
        )

    mild, strong = screener.score_snapshot(snap("MILD", 0.008)), screener.score_snapshot(snap("STRONG", 0.012))
    assert mild.score == strong.score == 100.0
    assert strong.rank_score > mild.rank_score


class AcceptedStatusBroker(ProtectionBroker):
    """Broker that refuses replaces, as Alpaca does for orders still 'accepted' (market closed)."""

    def __init__(self, open_orders) -> None:
        super().__init__(open_orders)
        self.cancelled: list[str] = []

    async def replace_order(self, order_id, *, qty=None, stop_price=None, limit_price=None):
        raise RuntimeError("PATCH failed with 422: 42210000: cannot replace order in accepted status")

    async def cancel_order(self, order_id):
        self.cancelled.append(order_id)


@pytest.mark.anyio
async def test_reconciler_cancels_and_replaces_when_broker_refuses_replace(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("trading_agent.agent._EXIT_RETRY_DELAY_SECONDS", 0)
    broker = AcceptedStatusBroker([_resting_stop(95)])
    await _agent(broker, _store_with_entry(tmp_path, peak=112))._reconcile_swing_protection(
        [_position(111)], None, skip_symbols=set()
    )
    assert broker.cancelled == ["rest-1"]
    [intent] = broker.submitted
    assert intent.stop_price == 106 and intent.time_in_force == TimeInForce.GTC


# --- swing / day-trade separation --------------------------------------------------


def test_adopted_position_ignores_stale_swing_entry_metadata(tmp_path) -> None:
    """A position the swing engine did not open (e.g. a day trade held overnight)
    must not inherit the ATR/stop of an older, unrelated swing trade."""
    store = AuditStore(tmp_path / "audit.sqlite3")
    store.record_event(
        cycle_id=None, event_type="order",
        payload={"intent": {"symbol": "AAPL", "limit_price": 150.0, "metadata": {"atr": 2.0, "initial_stop": 140.0}}},
        symbol="AAPL", strategy="equity_momentum", status="submitted",
    )
    manager = PositionManager(Settings(), store)
    # Position bought at 100, but the latest swing entry was priced at 150.
    assert manager.swing_stop(_position(101)) is None


def test_stale_by_age_swing_metadata_is_ignored(tmp_path) -> None:
    store = AuditStore(tmp_path / "audit.sqlite3")
    store.record_event(
        cycle_id=None, event_type="order",
        payload={"intent": {"symbol": "AAPL", "limit_price": 100.5, "metadata": {"atr": 2.0, "initial_stop": 95.0}}},
        symbol="AAPL", strategy="equity_momentum", status="submitted",
    )
    old = (datetime.now(UTC) - timedelta(days=120)).isoformat()
    with store._connect() as connection:
        connection.execute("update audit_events set created_at = ?", (old,))
    manager = PositionManager(Settings(), store)
    assert manager.swing_stop(_position(101)) is None


def test_matching_recent_swing_entry_is_used(tmp_path) -> None:
    store = AuditStore(tmp_path / "audit.sqlite3")
    store.record_event(
        cycle_id=None, event_type="order",
        payload={"intent": {"symbol": "AAPL", "limit_price": 100.5, "metadata": {"atr": 2.0, "initial_stop": 95.0}}},
        symbol="AAPL", strategy="equity_momentum", status="submitted",
    )
    stop = PositionManager(Settings(), store).swing_stop(_position(101))
    assert stop is not None and stop.atr == 2.0


def test_day_trade_exit_rules_never_touch_a_swing_position(tmp_path) -> None:
    """Only symbols with a day-trade entry today are managed by the day-trade engine,
    and the swing position manager skips exactly those symbols."""
    from trading_agent.audit import start_of_trading_day

    store = AuditStore(tmp_path / "audit.sqlite3")
    store.record_event(
        cycle_id=None, event_type="order",
        payload={"intent": {"symbol": "DT", "metadata": {"day_trade": True}}},
        symbol="DT", strategy="day_trade_entry", status="submitted",
    )
    agent = _agent(ProtectionBroker(), store)
    agent.settings = Settings.model_validate({"day_trading": {"enabled": True}})
    assert agent._day_trade_symbols_today() == {"DT"}
    assert agent._day_trade_entry_event_today("AAPL") is None
    assert store.latest_day_trade_entry_for_symbol("DT", start_of_trading_day()) is not None
