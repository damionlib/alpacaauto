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
