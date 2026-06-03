from trading_agent.catalyst.service import CatalystEngine
from trading_agent.config import Settings
from trading_agent.models import AssetClass, MarketSnapshot, NewsItem, OrderSide, ResearchSnapshot, TradeCandidate


def test_catalyst_scores_verified_bullish_setup() -> None:
    settings = Settings()
    engine = CatalystEngine(settings)
    prediction = engine.evaluate(
        MarketSnapshot(
            symbol="GOOD",
            asset_class=AssetClass.EQUITY,
            price=130,
            closes=[90 + index for index in range(60)],
        ),
        ResearchSnapshot(
            symbol="GOOD",
            news=[
                NewsItem(title="GOOD announces earnings beat and raises guidance"),
                NewsItem(title="GOOD secures new AI partnership contract"),
                NewsItem(title="Analyst upgrade follows record profit"),
            ],
            sec_summary={
                "latest_revenue": {"value": 1_000_000_000},
                "latest_net_income": {"value": 200_000_000},
                "recent_filings": [{"form": "10-Q"}],
            },
        ),
    )

    assert prediction.direction == "bullish"
    assert prediction.prediction_score >= settings.catalyst.min_trade_score
    assert prediction.confidence in {"medium", "high"}
    assert prediction.entry_allowed
    assert prediction.evidence


def test_catalyst_blocks_rumor_and_macro_event_risk() -> None:
    engine = CatalystEngine(Settings())
    prediction = engine.evaluate(
        MarketSnapshot(
            symbol="RISK",
            asset_class=AssetClass.EQUITY,
            price=85,
            closes=[100 - index * 0.2 for index in range(60)],
        ),
        ResearchSnapshot(
            symbol="RISK",
            news=[
                NewsItem(title="RISK reportedly in talks amid tariff and war uncertainty"),
                NewsItem(title="Rumor says President may announce sanctions"),
            ],
            sec_summary={
                "latest_revenue": {"value": 1_000_000},
                "latest_net_income": {"value": -50_000_000},
            },
        ),
    )

    assert not prediction.entry_allowed
    assert prediction.block_reason is not None
    assert "rumor_only_catalyst" in prediction.risks


def test_catalyst_blocks_misaligned_bullish_candidate() -> None:
    engine = CatalystEngine(Settings())
    prediction = engine.evaluate(
        MarketSnapshot(
            symbol="WEAK",
            asset_class=AssetClass.EQUITY,
            price=70,
            closes=[120 - index for index in range(60)],
        ),
        ResearchSnapshot(
            symbol="WEAK",
            news=[
                NewsItem(title="WEAK misses earnings and cuts guidance"),
                NewsItem(title="Analyst downgrade follows investigation"),
                NewsItem(title="Company announces 10-Q filing"),
            ],
            sec_summary={
                "latest_revenue": {"value": 100_000_000},
                "latest_net_income": {"value": -10_000_000},
                "recent_filings": [{"form": "10-Q"}],
            },
        ),
    )
    candidate = TradeCandidate(
        symbol="WEAK",
        asset_class=AssetClass.EQUITY,
        side=OrderSide.BUY,
        strategy="equity_momentum",
        score=90,
        entry_price=70,
    )

    accepted, blocked = engine.apply_to_candidates([candidate], prediction)

    assert accepted == []
    assert blocked
    assert "not bullish" in blocked[0][1]


def test_catalyst_allows_bearish_put_when_prediction_is_bearish() -> None:
    engine = CatalystEngine(Settings.model_validate({"catalyst": {"block_low_confidence": False}}))
    prediction = engine.evaluate(
        MarketSnapshot(
            symbol="WEAK",
            asset_class=AssetClass.EQUITY,
            price=70,
            closes=[120 - index for index in range(60)],
        ),
        ResearchSnapshot(
            symbol="WEAK",
            news=[
                NewsItem(title="WEAK misses earnings and cuts guidance"),
                NewsItem(title="Analyst downgrade follows fraud investigation"),
                NewsItem(title="Company announces 10-Q filing"),
            ],
            sec_summary={
                "latest_revenue": {"value": 100_000_000},
                "latest_net_income": {"value": -10_000_000},
                "recent_filings": [{"form": "10-Q"}],
            },
        ),
    )
    candidate = TradeCandidate(
        symbol="WEAK260618P00070000",
        asset_class=AssetClass.OPTION,
        side=OrderSide.BUY,
        strategy="long_put",
        score=71,
        entry_price=4,
        metadata={"contract": {"strike_price": "70"}},
    )

    accepted, blocked = engine.apply_to_candidates([candidate], prediction)

    assert not blocked
    assert accepted[0].metadata["catalyst"]["direction"] == "bearish"


def test_catalyst_can_generate_spot_entry_candidate() -> None:
    settings = Settings.model_validate({"catalyst": {"min_entry_score": 70}})
    engine = CatalystEngine(settings)
    market = MarketSnapshot(
        symbol="GOOD",
        asset_class=AssetClass.EQUITY,
        price=130,
        closes=[90 + index for index in range(60)],
    )
    prediction = engine.evaluate(
        market,
        ResearchSnapshot(
            symbol="GOOD",
            news=[
                NewsItem(title="GOOD announces earnings beat and raises guidance"),
                NewsItem(title="GOOD secures partnership contract"),
                NewsItem(title="Analyst upgrade follows record profit"),
            ],
            sec_summary={
                "latest_revenue": {"value": 1_000_000_000},
                "latest_net_income": {"value": 200_000_000},
                "recent_filings": [{"form": "10-Q"}],
            },
        ),
    )

    candidate = engine.entry_candidate(market, prediction, [])

    assert candidate is not None
    assert candidate.strategy == "equity_catalyst"
    assert candidate.metadata["catalyst"]["direction"] == "bullish"
