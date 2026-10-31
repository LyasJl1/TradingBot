from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from tradingbot.config import StrategyConfig, load_config
from tradingbot.indicators import atr, donchian
from tradingbot.signals import breakout_signal, evaluate_candles, generate_signals, trend_filter


def small_strategy() -> StrategyConfig:
    config = load_config(Path(__file__).parents[1] / "config.yaml")
    return replace(config.strategy, ema_fast_period=2, ema_slow_period=3,
                   adx_period=2, atr_period=2, donchian_entry_period=2,
                   donchian_exit_period=2)


def candles(start: str, count: int, frequency: str, descending: bool = False) -> pd.DataFrame:
    index = pd.date_range(start, periods=count, freq=frequency, tz="UTC", name="timestamp")
    close = 100.0 + np.arange(count) * (-0.5 if descending else 0.5)
    return pd.DataFrame({"open": close, "high": close + .1, "low": close - .1,
                         "close": close, "volume": 1.0}, index=index)


@pytest.mark.parametrize("close,fast,slow,adx_value,expected,reason", [
    (12, 11, 10, 18, "long", "long_trend_allowed"),
    (8, 9, 10, 18, "short", "short_trend_allowed"),
    (12, 11, 10, 17.999, "none", "adx_below_threshold"),
    (8, 9, 10, 17.999, "none", "adx_below_threshold"),
    (10, 11, 10, 19, "none", "trend_conditions_not_met"),
    (12, 10, 10, 19, "none", "trend_conditions_not_met"),
    (8, 10, 10, 19, "none", "trend_conditions_not_met"),
    (10, 9, 10, 19, "none", "trend_conditions_not_met"),
    (12, 9, 10, 19, "none", "trend_conditions_not_met"),
    (8, 11, 10, 19, "none", "trend_conditions_not_met"),
    (12, 11, 10, np.nan, "none", "insufficient_d1_history"),
    (12, np.nan, 10, 19, "none", "insufficient_d1_history"),
])
def test_long_short_filter_conditions(close, fast, slow, adx_value, expected, reason) -> None:
    assert trend_filter(close, fast, slow, adx_value, 18) == (expected, reason)


@pytest.mark.parametrize("close,allowed,expected", [
    (12.01, "long", "long"), (12, "long", "none"),
    (7.99, "short", "short"), (8, "short", "none"),
    (7.99, "long", "none"), (12.01, "short", "none"),
    (12.01, "none", "none"), (7.99, "none", "none"),
])
def test_breakout_comparisons_are_strict_and_trend_direction_specific(close, allowed, expected) -> None:
    assert breakout_signal(close, 12, 8, allowed)[0] == expected


def test_d1_alignment_is_strictly_before_h4_open_including_midnight() -> None:
    d1 = candles("2024-01-01", 9, "D")
    h4 = candles("2024-01-01", 54, "4h")
    result = evaluate_candles(h4, d1, small_strategy(), as_of=pd.Timestamp("2024-01-10T00:00Z"))
    midnight = result.loc[pd.Timestamp("2024-01-07T00:00Z")]
    four_am = result.loc[pd.Timestamp("2024-01-07T04:00Z")]
    assert midnight.d1_open == pd.Timestamp("2024-01-05T00:00Z")
    assert midnight.d1_close_time == pd.Timestamp("2024-01-06T00:00Z")
    assert four_am.d1_open == pd.Timestamp("2024-01-06T00:00Z")
    assert four_am.d1_close_time == pd.Timestamp("2024-01-07T00:00Z")
    matched = result.dropna(subset=["d1_close_time"])
    assert (matched.d1_close_time < matched.index).all()
    assert (result.signal_time == result.index + pd.Timedelta(hours=4)).all()
    assert (result.entry_time == result.signal_time).all()


@pytest.mark.parametrize("descending,direction", [(False, "long"), (True, "short")])
def test_complete_signal_contains_reason_prior_atr_and_channels(descending, direction) -> None:
    d1 = candles("2024-01-01", 9, "D", descending)
    h4 = candles("2024-01-01", 54, "4h", descending)
    before_h4, before_d1 = h4.copy(deep=True), d1.copy(deep=True)
    config = small_strategy()
    result = evaluate_candles(h4, d1, config, as_of=pd.Timestamp("2024-01-10T00:00Z"))
    last = result.iloc[-1]
    assert last.direction == direction
    assert last.reason == f"{direction}_breakout"
    assert last.trend_reason == f"{direction}_trend_allowed"
    assert last.atr == atr(h4, config.atr_period).iloc[-2]
    assert last.entry_upper == donchian(h4, config.donchian_entry_period).upper.iloc[-1]
    assert last.exit_lower == donchian(h4, config.donchian_exit_period).lower.iloc[-1]
    assert_frame_equal(h4, before_h4)
    assert_frame_equal(d1, before_d1)
    assert result.reason.notna().all()
    assert result.direction.iloc[0] == "none"


def test_forming_h4_and_d1_candles_never_enter_decisions() -> None:
    h4 = candles("2024-01-01", 54, "4h")
    d1 = candles("2024-01-01", 9, "D")
    cutoff = pd.Timestamp("2024-01-07T13:00Z")
    expected = evaluate_candles(h4.iloc[:39], d1.iloc[:6], small_strategy(), as_of=cutoff)
    actual = evaluate_candles(h4, d1, small_strategy(), as_of=cutoff)
    assert_frame_equal(actual, expected)
    assert actual.signal_time.max() == pd.Timestamp("2024-01-07T12:00Z")


def test_no_completed_candles_returns_empty_decision_table() -> None:
    config = load_config(Path(__file__).parents[1] / "config.yaml")
    result = generate_signals(candles("2024-01-01", 1, "h"), config,
                              as_of=pd.Timestamp("2024-01-01T01:00Z"))
    assert result.empty
    assert {"direction", "reason", "signal_time", "entry_time", "atr"}.issubset(result.columns)


def test_signal_evaluator_volume_is_optional_and_unused() -> None:
    h4 = candles("2024-01-01", 54, "4h")
    d1 = candles("2024-01-01", 9, "D")
    as_of = pd.Timestamp("2024-01-10T00:00Z")
    expected = evaluate_candles(h4, d1, small_strategy(), as_of=as_of)
    assert_frame_equal(
        evaluate_candles(h4.drop(columns="volume"), d1.drop(columns="volume"),
                         small_strategy(), as_of=as_of), expected,
    )
    assert_frame_equal(
        evaluate_candles(h4.assign(volume="unused"), d1.assign(volume=np.nan),
                         small_strategy(), as_of=as_of), expected,
    )


def test_signal_evaluator_accepts_h4_and_d1_market_gaps_without_filling() -> None:
    h4 = candles("2024-01-01", 54, "4h").drop(columns="volume")
    d1 = candles("2024-01-01", 9, "D").drop(columns="volume")
    # Remove Saturday and Sunday entirely from each internally built timeframe.
    h4 = h4.loc[h4.index.dayofweek < 5]
    d1 = d1.loc[d1.index.dayofweek < 5]
    result = evaluate_candles(h4, d1, small_strategy(), as_of=pd.Timestamp("2024-01-10T00:00Z"))
    assert result.index.equals(h4.index)
    assert not (result.index.dayofweek >= 5).any()
    assert (result.direction == "long").any()
    assert result.loc[pd.Timestamp("2024-01-08T00:00Z"), "d1_open"] == pd.Timestamp("2024-01-05T00:00Z")


@pytest.mark.parametrize("cutoff", ["2024-01-10", "2024-01-10T00:00+01:00", pd.NaT])
def test_evaluation_rejects_ambiguous_cutoff(cutoff) -> None:
    with pytest.raises(ValueError, match="UTC|NaT"):
        evaluate_candles(candles("2024-01-01", 6, "4h"), candles("2024-01-01", 1, "D"),
                         small_strategy(), as_of=pd.Timestamp(cutoff))
