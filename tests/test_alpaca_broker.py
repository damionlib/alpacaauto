import pytest

from trading_agent.brokers.alpaca import AlpacaBroker


@pytest.mark.anyio
async def test_close_all_positions_calls_alpaca_endpoint() -> None:
    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker.trading_base_url = "https://paper-api.alpaca.markets"
    calls = []

    async def fake_request(method, url, *, params=None, json=None):
        calls.append((method, url, params, json))
        return [{"symbol": "SPY"}]

    broker._request = fake_request

    result = await broker.close_all_positions(cancel_orders=True)

    assert result == [{"symbol": "SPY"}]
    assert calls == [
        (
            "DELETE",
            "https://paper-api.alpaca.markets/v2/positions",
            {"cancel_orders": "true"},
            None,
        )
    ]


@pytest.mark.anyio
async def test_close_position_calls_alpaca_endpoint() -> None:
    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker.trading_base_url = "https://paper-api.alpaca.markets"
    calls = []

    async def fake_request(method, url, *, params=None, json=None):
        calls.append((method, url, params, json))
        return {"symbol": "AAPL"}

    broker._request = fake_request

    result = await broker.close_position("AAPL")

    assert result == {"symbol": "AAPL"}
    assert calls == [
        (
            "DELETE",
            "https://paper-api.alpaca.markets/v2/positions/AAPL",
            None,
            None,
        )
    ]


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


@pytest.mark.anyio
async def test_cancel_order_handles_204_no_content(respx_mock) -> None:
    respx_mock.delete("https://paper-api.alpaca.markets/v2/orders/abc-123").respond(204)
    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker.trading_base_url = "https://paper-api.alpaca.markets"
    broker.headers = {"APCA-API-KEY-ID": "k", "APCA-API-SECRET-KEY": "s"}

    # Must not raise: Alpaca returns 204 with an empty body on cancel, and an
    # exception here makes the caller believe the cancel failed when it worked.
    await broker.cancel_order("abc-123")


@pytest.mark.anyio
async def test_request_raises_on_error_status(respx_mock) -> None:
    respx_mock.delete("https://paper-api.alpaca.markets/v2/orders/bad-1").respond(
        403, json={"code": 40310000, "message": "insufficient qty available"}
    )
    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker.trading_base_url = "https://paper-api.alpaca.markets"
    broker.headers = {"APCA-API-KEY-ID": "k", "APCA-API-SECRET-KEY": "s"}

    with pytest.raises(RuntimeError, match="insufficient qty"):
        await broker.cancel_order("bad-1")
