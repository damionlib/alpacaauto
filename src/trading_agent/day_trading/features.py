"""Intraday features for the day-trade engine, computed from one-minute bars.

Pure functions shared by the live broker adapter and the minute-bar backtest, so
both see exactly the same feature definitions.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, time
from typing import Any
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")
SESSION_OPEN = time(9, 30)
SESSION_CLOSE = time(16, 0)
OPENING_RANGE_MINUTES = 15


def bar_time(bar: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(str(bar["t"]).replace("Z", "+00:00")).astimezone(EASTERN)


def minute_of_session(when: datetime) -> int:
    """Minutes since 09:30 ET (0 for the 09:30 bar)."""
    return (when.hour - 9) * 60 + when.minute - 30


def regular_session_bars(bars: list[dict[str, Any]], day: date) -> list[dict[str, Any]]:
    rows = []
    for bar in bars:
        if not bar.get("t"):
            continue
        when = bar_time(bar)
        if when.date() == day and SESSION_OPEN <= when.time() < SESSION_CLOSE:
            rows.append(bar)
    return rows


def cumulative_volume_profile(history_bars: list[dict[str, Any]], before_day: date, days: int = 10) -> list[float]:
    """Average cumulative volume by minute-of-session over the last `days` sessions.

    Element m is the mean volume traded from 09:30 through minute m. Comparing
    today's cumulative volume with this at the same minute gives a relative
    volume that respects the U-shaped intraday volume curve (the old version
    assumed volume accrues evenly, overstating it early in the day).
    """
    per_day: dict[date, list[float]] = defaultdict(lambda: [0.0] * 390)
    for bar in history_bars:
        if not bar.get("t"):
            continue
        when = bar_time(bar)
        if when.date() >= before_day or not (SESSION_OPEN <= when.time() < SESSION_CLOSE):
            continue
        per_day[when.date()][minute_of_session(when)] += float(bar.get("v") or 0)
    sessions = sorted(per_day)[-days:]
    if not sessions:
        return []
    profile = [0.0] * 390
    for session in sessions:
        running = 0.0
        volumes = per_day[session]
        for minute in range(390):
            running += volumes[minute]
            profile[minute] += running
    return [value / len(sessions) for value in profile]


def intraday_features(
    today_bars: list[dict[str, Any]],
    *,
    now: datetime,
    volume_profile: list[float] | None = None,
    orb_valid_minutes: int = 120,
) -> dict[str, Any]:
    """VWAP, short-term trend, opening-range break and relative volume as of `now`.

    `today_bars` must be one session's regular-hours bars (any order). Bars
    stamped at or after `now` are ignored so a backtest cannot peek ahead.
    """
    now_et = now.astimezone(EASTERN)
    rows = sorted(
        (bar for bar in today_bars if bar_time(bar) < now_et),
        key=lambda bar: bar["t"],
    )
    if not rows:
        return {}
    features: dict[str, Any] = {}
    volumes = [float(bar.get("v") or 0) for bar in rows]
    traded = sum(volumes)
    if traded > 0:
        weighted = sum(float(bar.get("vw") or bar.get("c") or 0) * float(bar.get("v") or 0) for bar in rows)
        features["vwap"] = round(weighted / traded, 4)

    # Short-term trend: last close vs the close ~5 minutes earlier (by time,
    # since a single-venue feed skips minutes with no trades).
    last = rows[-1]
    last_minute = minute_of_session(bar_time(last))
    earlier = [bar for bar in rows if minute_of_session(bar_time(bar)) <= last_minute - 5]
    if earlier and float(earlier[-1]["c"]):
        base = float(earlier[-1]["c"])
        features["minute_trend_pct"] = round((float(last["c"]) - base) / base * 100, 4)

    minutes_elapsed = minute_of_session(now_et)
    opening = [bar for bar in rows if minute_of_session(bar_time(bar)) < OPENING_RANGE_MINUTES]
    if opening and minutes_elapsed >= OPENING_RANGE_MINUTES:
        opening_high = max(float(bar["h"]) for bar in opening)
        features["opening_range_high"] = opening_high
        # A break counts only while it is still an opening-range trade; price
        # above the morning high at 2 PM is not the same setup.
        features["opening_range_break"] = bool(
            float(last["c"]) > opening_high and minutes_elapsed <= orb_valid_minutes
        )

    if volume_profile:
        index = max(0, min(389, minutes_elapsed - 1))
        expected = volume_profile[index]
        if expected > 0:
            features["relative_volume"] = round(traded / expected, 3)
    features["minutes_since_open"] = minutes_elapsed
    return features
