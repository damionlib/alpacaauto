from __future__ import annotations


NASDAQ100_CORE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "AVGO", "GOOGL", "GOOG", "TSLA", "COST",
    "NFLX", "AMD", "ADBE", "PEP", "CSCO", "TMUS", "INTU", "AMAT", "QCOM", "TXN",
    "AMGN", "ISRG", "HON", "BKNG", "VRTX", "SBUX", "MU", "LRCX", "PANW", "ADP",
    "GILD", "MDLZ", "MELI", "ADI", "KLAC", "CRWD", "REGN", "CDNS", "SNPS", "MAR",
    "PYPL", "ASML", "ORLY", "CSX", "ABNB", "MRVL", "FTNT", "NXPI", "ROP", "CHTR",
]

SP500_CORE = [
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "BRK.B",
    "LLY", "JPM", "AVGO", "XOM", "UNH", "V", "MA", "PG", "COST", "HD",
    "JNJ", "NFLX", "ABBV", "WMT", "BAC", "KO", "PM", "CRM", "ORCL", "CVX",
    "MRK", "AMD", "PEP", "TMO", "LIN", "ADBE", "CSCO", "ACN", "MCD", "ABT",
    "GE", "IBM", "DIS", "CAT", "QCOM", "VZ", "INTU", "NOW", "AMAT", "TXN",
]

CRYPTO_MAJOR = [
    "BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "LTC/USD", "BCH/USD", "LINK/USD", "AVAX/USD",
]


UNIVERSES = {
    "nasdaq100": NASDAQ100_CORE,
    "sp500_core": SP500_CORE,
    "crypto_major": CRYPTO_MAJOR,
}


def symbols_for_universes(universes: list[str]) -> list[str]:
    seen: set[str] = set()
    symbols: list[str] = []
    for universe in universes:
        for symbol in UNIVERSES.get(universe, []):
            if symbol not in seen:
                seen.add(symbol)
                symbols.append(symbol)
    return symbols
