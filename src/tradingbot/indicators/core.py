"""Indicator formulas; periods are supplied by the central configuration.

These functions never read configuration, accounts or brokers. ATR itself is
unshifted: the caller must shift it once when constructing a signal. Donchian
channels already exclude the current row.
"""

from numbers import Integral

import numpy as np
import pandas as pd


def _period(period: int) -> int:
    if isinstance(period, bool) or not isinstance(period, Integral) or period < 1:
        raise ValueError("period must be a positive integer")
    return int(period)


def _values(series: pd.Series, label: str) -> np.ndarray:
    if not isinstance(series, pd.Series):
        raise TypeError(f"{label} must be a pandas Series")
    try:
        values = series.to_numpy(dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain numeric values") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{label} must contain only finite values")
    return values


def _prices(candles: pd.DataFrame, *, close: bool) -> tuple[np.ndarray, ...]:
    if not isinstance(candles, pd.DataFrame):
        raise TypeError("candles must be a pandas DataFrame")
    columns = ("high", "low", "close") if close else ("high", "low")
    missing = set(columns) - set(candles.columns)
    if missing:
        raise ValueError(f"missing price columns: {', '.join(sorted(missing))}")
    values = tuple(_values(candles[column], column) for column in columns)
    high, low = values[:2]
    if (high < low).any():
        raise ValueError("high must be greater than or equal to low")
    if close and ((values[2] > high) | (values[2] < low)).any():
        raise ValueError("close must lie between low and high")
    return values


def _wilder_average(values: np.ndarray, period: int, start: int = 0) -> np.ndarray:
    """Seed with an arithmetic mean, then recur with alpha = 1 / period."""
    result = np.full(len(values), np.nan, dtype=float)
    seed_position = start + period - 1
    if seed_position >= len(values):
        return result
    result[seed_position] = np.mean(values[start : seed_position + 1])
    for position in range(seed_position + 1, len(values)):
        result[position] = (
            result[position - 1] * (period - 1) + values[position]
        ) / period
    return result


def ema(close: pd.Series, period: int) -> pd.Series:
    """EMA seeded by the first ``period`` closes' SMA at index period - 1.

    Subsequent values use alpha = 2 / (period + 1). Earlier rows are NaN;
    input NaNs and infinities are rejected, never interpolated or filled.
    """
    period = _period(period)
    values = _values(close, "close")
    result = np.full(len(values), np.nan, dtype=float)
    if len(values) >= period:
        position = period - 1
        result[position] = np.mean(values[:period])
        alpha = 2.0 / (period + 1)
        for position in range(period, len(values)):
            result[position] = (
                alpha * values[position] + (1.0 - alpha) * result[position - 1]
            )
    return pd.Series(result, index=close.index, name="ema")


def true_range(candles: pd.DataFrame) -> pd.Series:
    """max(high-low, abs(high-prev_close), abs(low-prev_close)).

    The first row uses high-low because no previous close is available.
    """
    high, low, close = _prices(candles, close=True)
    result = high - low
    if len(result) > 1:
        result[1:] = np.maximum.reduce(
            (result[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
        )
    return pd.Series(result, index=candles.index, name="true_range")


def atr(candles: pd.DataFrame, period: int) -> pd.Series:
    """Wilder ATR, SMA seeded from TR rows 0..period-1 (inclusive).

    The first TR is high-low. The first ATR is at index period-1;
    subsequent values are (previous * (period-1) + current TR) / period.
    The returned ATR includes its own row; signals must use ``atr.shift(1)``.
    """
    period = _period(period)
    ranges = true_range(candles).to_numpy()
    return pd.Series(
        _wilder_average(ranges, period), index=candles.index, name="atr"
    )


def adx(candles: pd.DataFrame, period: int) -> pd.Series:
    """Wilder ADX using directional movements between successive bars.

    Initial TR/+DM/-DM averages use rows 1..period (row 0 has no movement).
    The first DX is at index period. The first ADX averages DX rows
    period..2*period-1 and appears at index 2*period-1. Each subsequent
    smoothing uses Wilder's alpha 1/period. Tied upward/downward movements
    contribute to neither DM; zero TR or zero total DI gives DX zero.
    """
    period = _period(period)
    high, low, _ = _prices(candles, close=True)
    count = len(candles)
    result = np.full(count, np.nan, dtype=float)
    if count <= period:
        return pd.Series(result, index=candles.index, name="adx")

    upward = np.diff(high, prepend=high[0])
    downward = -np.diff(low, prepend=low[0])
    plus_dm = np.where((upward > downward) & (upward > 0), upward, 0.0)
    minus_dm = np.where((downward > upward) & (downward > 0), downward, 0.0)
    ranges = true_range(candles).to_numpy()
    smoothed_tr = _wilder_average(ranges, period, start=1)
    smoothed_plus = _wilder_average(plus_dm, period, start=1)
    smoothed_minus = _wilder_average(minus_dm, period, start=1)

    plus_di = np.zeros(count, dtype=float)
    minus_di = np.zeros(count, dtype=float)
    np.divide(100.0 * smoothed_plus, smoothed_tr, out=plus_di, where=smoothed_tr > 0)
    np.divide(100.0 * smoothed_minus, smoothed_tr, out=minus_di, where=smoothed_tr > 0)
    total_di = plus_di + minus_di
    dx = np.zeros(count, dtype=float)
    np.divide(100.0 * np.abs(plus_di - minus_di), total_di, out=dx, where=total_di > 0)
    result = _wilder_average(dx, period, start=period)
    return pd.Series(result, index=candles.index, name="adx")


def donchian(candles: pd.DataFrame, period: int) -> pd.DataFrame:
    """Highest high / lowest low of the previous period rows, shifted by one.

    The current candle is always excluded, even during warmup. Both channels
    remain NaN until a complete window of prior observations is available.
    """
    period = _period(period)
    high, low = _prices(candles, close=False)
    upper = pd.Series(high, index=candles.index).rolling(period).max().shift(1)
    lower = pd.Series(low, index=candles.index).rolling(period).min().shift(1)
    return pd.DataFrame({"upper": upper, "lower": lower}, index=candles.index)
