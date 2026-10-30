"""Conservative temporal alignment shared by future backtest and paper modes."""

from math import isfinite
from typing import Literal

import numpy as np
import pandas as pd

from tradingbot.config import BotConfig, StrategyConfig
from tradingbot.data import build_timeframes
from tradingbot.indicators import adx, atr, donchian, ema

Direction = Literal["long", "short", "none"]


def trend_filter(
    close: float, ema_fast: float, ema_slow: float, adx_value: float, threshold: float,
) -> tuple[Direction, str]:
    """Apply precisely the D1 strict price/EMA comparisons and inclusive ADX gate."""
    if not isfinite(threshold) or not 0 <= threshold <= 100:
        raise ValueError("ADX threshold must be finite and in [0, 100]")
    if not all(isfinite(v) for v in (close, ema_fast, ema_slow, adx_value)):
        return "none", "insufficient_d1_history"
    if adx_value < threshold:
        return "none", "adx_below_threshold"
    if close > ema_slow and ema_fast > ema_slow:
        return "long", "long_trend_allowed"
    if close < ema_slow and ema_fast < ema_slow:
        return "short", "short_trend_allowed"
    return "none", "trend_conditions_not_met"


def breakout_signal(
    close: float, upper: float, lower: float, allowed: Direction,
) -> tuple[Direction, str]:
    """A close equal to a channel is not a breakout."""
    if allowed not in {"long", "short", "none"}:
        raise ValueError("allowed must be long, short or none")
    if not all(isfinite(v) for v in (close, upper, lower)):
        return "none", "insufficient_h4_history"
    if lower > upper:
        raise ValueError("Donchian lower cannot exceed upper")
    if allowed == "long" and close > upper:
        return "long", "long_breakout"
    if allowed == "short" and close < lower:
        return "short", "short_breakout"
    if allowed == "none":
        return "none", "trend_conditions_not_met"
    return "none", "no_breakout"


def _closed_candles(frame: pd.DataFrame, timeframe: str, as_of: pd.Timestamp) -> pd.DataFrame:
    """Check the resampled-frame contract and omit candles still in progress."""
    if not isinstance(frame.index, pd.DatetimeIndex) or str(frame.index.tz) != "UTC":
        raise ValueError(f"{timeframe} candles need a UTC DatetimeIndex of open times")
    if frame.index.hasnans or frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{timeframe} timestamps must be ordered, unique and nonmissing")
    frequency = {"H4": "4h", "D1": "1D"}[timeframe]
    if not frame.index.equals(frame.index.floor(frequency)):
        raise ValueError(f"{timeframe} candle open times must align to UTC boundaries")
    columns = ["open", "high", "low", "close", "volume"]
    if not set(columns).issubset(frame.columns):
        raise ValueError(f"{timeframe} candles require OHLCV columns")
    closed = frame.loc[frame.index + pd.Timedelta(frequency) <= as_of, columns].copy()
    if not np.isfinite(closed.to_numpy(dtype=float)).all():
        raise ValueError(f"{timeframe} OHLCV must be finite")
    if ((closed[["open", "high", "low", "close"]] <= 0).any().any()
            or (closed.volume < 0).any()
            or (closed.high < closed[["open", "close", "low"]].max(axis=1)).any()
            or (closed.low > closed[["open", "close", "high"]].min(axis=1)).any()):
        raise ValueError(f"{timeframe} invalid OHLCV")
    if len(closed) > 1 and not (closed.index.to_series().diff().iloc[1:] == pd.Timedelta(frequency)).all():
        raise ValueError(f"{timeframe} candles contain gaps")
    return closed


def evaluate_candles(
    h4: pd.DataFrame, d1: pd.DataFrame, config: StrategyConfig, *, as_of: pd.Timestamp,
) -> pd.DataFrame:
    """Low-level pure evaluator of internally resampled candles.

    Use generate_signals for base-data inputs. D1 close time must be STRICTLY
    earlier than the H4 OPEN, including at midnight. ATR and channels exclude the
    signal candle. Each result records its decision and intended next-open time;
    it is a candidate signal, not an instruction to place an order.
    """
    as_of = pd.Timestamp(as_of)
    if as_of.tzinfo is None or str(as_of.tzinfo) != "UTC":
        raise ValueError("as_of must be explicitly UTC")
    if pd.isna(as_of):
        raise ValueError("as_of must not be NaT")
    h4 = _closed_candles(h4, "H4", as_of)
    d1 = _closed_candles(d1, "D1", as_of)
    entry = donchian(h4, config.donchian_entry_period)
    exit_channel = donchian(h4, config.donchian_exit_period)
    h4_values = pd.DataFrame({
        "candle_open": h4.index,
        "close": h4.close.to_numpy(),
        "atr": atr(h4, config.atr_period).shift(1).to_numpy(),
        "entry_upper": entry.upper.to_numpy(),
        "entry_lower": entry.lower.to_numpy(),
        "exit_upper": exit_channel.upper.to_numpy(),
        "exit_lower": exit_channel.lower.to_numpy(),
    })
    daily = pd.DataFrame({
        "d1_open": d1.index,
        "d1_close_time": d1.index + pd.Timedelta(days=1),
        "d1_close": d1.close.to_numpy(),
        "d1_ema_fast": ema(d1.close, config.ema_fast_period).to_numpy(),
        "d1_ema_slow": ema(d1.close, config.ema_slow_period).to_numpy(),
        "d1_adx": adx(d1, config.adx_period).to_numpy(),
    })
    result = pd.merge_asof(
        h4_values, daily, left_on="candle_open", right_on="d1_close_time",
        direction="backward", allow_exact_matches=False,
    )
    result["signal_time"] = result.candle_open + pd.Timedelta(hours=4)
    result["entry_time"] = result.signal_time
    decisions: list[tuple[Direction, str]] = []
    trends: list[str] = []
    for row in result.itertuples(index=False):
        allowed, reason = trend_filter(
            row.d1_close, row.d1_ema_fast, row.d1_ema_slow, row.d1_adx, config.adx_threshold,
        )
        trends.append(reason)
        if not all(isfinite(v) for v in (row.atr, row.entry_upper, row.entry_lower)):
            decisions.append(("none", "insufficient_h4_history"))
        elif allowed == "none":
            decisions.append(("none", reason))
        else:
            decisions.append(breakout_signal(row.close, row.entry_upper, row.entry_lower, allowed))
    result["trend_reason"] = pd.Series(trends, index=result.index, dtype="string")
    result["direction"] = pd.Series([v[0] for v in decisions], index=result.index, dtype="string")
    result["reason"] = pd.Series([v[1] for v in decisions], index=result.index, dtype="string")
    return result.set_index("candle_open")


def generate_signals(
    base: pd.DataFrame, config: BotConfig, *, as_of: pd.Timestamp,
) -> pd.DataFrame:
    """Build both timeframes from M15/H1 base data and return closed H4 decisions.

    No wall clock, I/O, account access, randomness, filters, risk or execution.
    All decision reasons and the underlying aligned indicator values are returned.
    """
    frames = build_timeframes(base, config.data.base_timeframe, as_of=as_of)
    return evaluate_candles(frames["H4"], frames["D1"], config.strategy, as_of=as_of)
