from trading_agent.brokers.alpaca import AlpacaBroker


def test_option_quote_midpoint_uses_bid_ask() -> None:
    broker = AlpacaBroker.__new__(AlpacaBroker)

    assert broker.option_quote_midpoint({"bp": 1.0, "ap": 1.4}) == 1.2


def test_option_quote_midpoint_falls_back_to_ask() -> None:
    broker = AlpacaBroker.__new__(AlpacaBroker)

    assert broker.option_quote_midpoint({"bp": 0, "ap": 0.25}) == 0.25


def test_option_quote_bid_ask_accepts_verbose_keys() -> None:
    broker = AlpacaBroker.__new__(AlpacaBroker)

    assert broker.option_quote_bid_ask({"bid_price": 2.0, "ask_price": 2.5}) == (2.0, 2.5)


def test_asset_class_detects_option_symbol_even_if_broker_reports_equity() -> None:
    broker = AlpacaBroker.__new__(AlpacaBroker)

    assert broker._asset_class_from_alpaca("equity", "AAPL260612C00322500").value == "option"
