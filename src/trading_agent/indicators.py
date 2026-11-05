from __future__ import annotations


def sma(values: list[float], window: int) -> float | None:
    if len(values) < window:
        return None
    return sum(values[-window:]) / window


def pct_change(values: list[float], periods: int) -> float | None:
    if len(values) <= periods or values[-periods - 1] == 0:
        return None
    return ((values[-1] - values[-periods - 1]) / values[-periods - 1]) * 100


def realized_volatility(values: list[float], periods: int = 20) -> float | None:
    if len(values) <= periods:
        return None
    returns = []
    window = values[-periods - 1 :]
    for previous, current in zip(window, window[1:], strict=False):
        if previous:
            returns.append((current - previous) / previous)
    if len(returns) < 2:
        return None
    mean = sum(returns) / len(returns)
    variance = sum((ret - mean) ** 2 for ret in returns) / (len(returns) - 1)
    return (variance**0.5) * (252**0.5) * 100


def atr(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> float | None:
    """Average true range (simple mean of the last `period` true ranges)."""
    n = min(len(highs), len(lows), len(closes))
    if n <= period:
        return None
    highs, lows, closes = highs[-n:], lows[-n:], closes[-n:]
    true_ranges = []
    for index in range(n - period, n):
        previous_close = closes[index - 1]
        true_ranges.append(
            max(
                highs[index] - lows[index],
                abs(highs[index] - previous_close),
                abs(lows[index] - previous_close),
            )
        )
    return sum(true_ranges) / period
