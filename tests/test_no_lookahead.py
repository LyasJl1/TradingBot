"""Causality checks run the entire default-parameter base-data signal pipeline.

The synthetic history deliberately lasts well beyond EMA(200)'s warmup and
contains both upward and downward regimes. Comparing only warmup ``none``
decisions would not establish temporal integrity.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from tradingbot.config import BotConfig, load_config
from tradingbot.signals import generate_signals


@pytest.fixture(scope="module")
def config() -> BotConfig:
    return load_config(Path(__file__).resolve().parents[1] / "config.yaml")


@pytest.fixture(scope="module")
def hourly(config: BotConfig) -> pd.DataFrame:
    hours = np.arange(720 * 24, dtype=float)
    turn = 330 * 24
    rng = np.random.default_rng(config.random_seed)
    close = (
        1.1
        + 0.0002 * np.where(hours < turn, hours, 2 * turn - hours)
        + 0.002 * np.sin(hours * 2 * np.pi / 72)
        + rng.normal(0, 0.000001, len(hours))
    )
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + 0.0001,
            "low": np.minimum(open_, close) - 0.0001,
            "close": close,
            "volume": np.full(len(hours), 10.0),
        },
        index=pd.date_range(
            "2022-01-01", periods=len(hours), freq="h", tz="UTC", name="timestamp",
        ),
    )


@pytest.fixture(scope="module")
def full(hourly: pd.DataFrame, config: BotConfig) -> pd.DataFrame:
    end = hourly.index[-1] + pd.Timedelta(hours=1)
    decisions = generate_signals(hourly, config, as_of=end)
    # Require mature indicators and actual decisions in BOTH directions, rather
    # than letting causality tests pass on all-NaN indicators or all-none signals.
    assert (decisions.direction == "long").any()
    assert (decisions.direction == "short").any()
    mature = decisions.loc[decisions.direction != "none"]
    columns = ["atr", "entry_upper", "entry_lower", "exit_upper", "exit_lower",
               "d1_close", "d1_ema_fast", "d1_ema_slow", "d1_adx"]
    assert mature[columns].notna().all().all()
    assert (mature.d1_adx >= config.strategy.adx_threshold).all()
    return decisions


@pytest.mark.parametrize(
    "cutoff_text",
    [
        "2022-09-08T00:00:00Z",  # Midnight: exact D1 close is not yet eligible.
        "2022-09-08T04:00:00Z",  # Next H4: the preceding D1 is now eligible.
        "2022-09-08T07:30:00Z",  # Both an H1 and an H4 candle are still forming.
        "2023-05-16T00:00:00Z",  # Mature bearish regime, UTC midnight.
        "2023-05-16T16:00:00Z",  # Mature bearish H4 boundary.
        "2023-05-16T19:30:00Z",  # Bearish incomplete-candle cutoff.
    ],
)
def test_truncating_future_preserves_every_earlier_signal_and_input(
    hourly: pd.DataFrame, full: pd.DataFrame, config: BotConfig, cutoff_text: str,
) -> None:
    cutoff = pd.Timestamp(cutoff_text)
    truncated = hourly.loc[hourly.index < cutoff]
    actual = generate_signals(truncated, config, as_of=cutoff)
    expected = full.loc[full.signal_time <= cutoff]
    # Includes exact directions, reasons, channels, ATR, aligned D1 indicators,
    # the daily source timestamps, and intended next-open entry timestamps.
    assert_frame_equal(actual, expected, check_exact=True)
    assert actual.atr.notna().any()
    assert actual.d1_ema_slow.notna().any()
    assert (actual.direction != "none").any()
    assert (actual.signal_time <= cutoff).all()
    aligned = actual.loc[actual.d1_close_time.notna()]
    assert (aligned.d1_close_time.to_numpy() < aligned.index.to_numpy()).all()


def test_mutating_all_future_prices_cannot_change_historical_decisions(
    hourly: pd.DataFrame, full: pd.DataFrame, config: BotConfig,
) -> None:
    cutoff = pd.Timestamp("2023-05-16T16:00:00Z")
    mutated = hourly.copy(deep=True)
    future = mutated.index >= cutoff
    mutated.loc[future, ["open", "high", "low", "close"]] = [150.0, 200.0, 100.0, 150.0]
    actual = generate_signals(
        mutated, config, as_of=hourly.index[-1] + pd.Timedelta(hours=1),
    )
    assert_frame_equal(
        actual.loc[actual.signal_time <= cutoff],
        full.loc[full.signal_time <= cutoff],
        check_exact=True,
    )
    # The change really affects the future, so this is not a no-op mutation.
    assert not actual.loc[actual.index > cutoff, "atr"].equals(
        full.loc[full.index > cutoff, "atr"],
    )


def test_appended_forming_base_candle_is_not_used(
    hourly: pd.DataFrame, full: pd.DataFrame, config: BotConfig,
) -> None:
    end = hourly.index[-1] + pd.Timedelta(hours=1)
    forming = pd.DataFrame(
        {"open": [50.0], "high": [100.0], "low": [0.1], "close": [75.0], "volume": [999.0]},
        index=pd.DatetimeIndex([end], name="timestamp"),
    )
    appended = pd.concat([hourly, forming])
    actual = generate_signals(appended, config, as_of=end + pd.Timedelta(minutes=30))
    assert_frame_equal(actual, full, check_exact=True)


def test_signal_candle_extremes_excluded_from_atr_and_both_channels(
    hourly: pd.DataFrame, full: pd.DataFrame, config: BotConfig,
) -> None:
    signal_open = pd.Timestamp("2023-05-16T04:00:00Z")
    mutated = hourly.copy(deep=True)
    in_signal = (mutated.index >= signal_open) & (
        mutated.index < signal_open + pd.Timedelta(hours=4)
    )
    # Keep the signal close identical while changing its range drastically.
    mutated.loc[in_signal, "high"] = 100.0
    mutated.loc[in_signal, "low"] = 0.1
    actual = generate_signals(mutated, config, as_of=signal_open + pd.Timedelta(hours=8))
    assert_frame_equal(
        actual.loc[[signal_open]], full.loc[[signal_open]], check_exact=True,
    )
    next_open = signal_open + pd.Timedelta(hours=4)
    for column in ("atr", "entry_upper", "entry_lower", "exit_upper", "exit_lower"):
        assert actual.loc[next_open, column] != full.loc[next_open, column]


def test_m15_and_h1_share_the_same_resampling_and_causality(
    hourly: pd.DataFrame, full: pd.DataFrame, config: BotConfig,
) -> None:
    # Four sub-bars reproduce each original H1 OHLC exactly; their volumes sum.
    quarters = hourly.loc[hourly.index.repeat(4)].copy()
    quarters.index = pd.date_range(
        hourly.index[0], periods=len(quarters), freq="15min", name="timestamp",
    )
    quarters["volume"] /= 4
    m15_config = replace(config, data=replace(config.data, base_timeframe="M15"))
    result = generate_signals(
        quarters, m15_config, as_of=hourly.index[-1] + pd.Timedelta(hours=1),
    )
    assert_frame_equal(result, full, check_exact=True)
    cutoff = pd.Timestamp("2023-05-16T19:37:00Z")
    truncated = generate_signals(
        quarters.loc[quarters.index < cutoff], m15_config, as_of=cutoff,
    )
    assert_frame_equal(
        truncated, result.loc[result.signal_time <= cutoff], check_exact=True,
    )


@pytest.fixture(scope="module")
def gappy(hourly: pd.DataFrame) -> pd.DataFrame:
    """BID-like history with real closures, outages, flat bars and no volume.

    Missing periods remain missing; no padded or artificial bars are introduced.
    The same long history is retained so default EMA/ADX warmup is exercised.
    """
    frame = hourly.drop(columns="volume").copy(deep=True)
    # Entire weekends and individual weekday observations are absent.
    keep = (frame.index.dayofweek < 5) & (frame.index.hour != 13)
    # Some 08:00 H4 buckets contain only one observation, below the minimum.
    sparse_h4 = (frame.index.dayofweek == 2) & frame.index.hour.isin([9, 10, 11])
    keep &= ~sparse_h4
    # Some whole weekdays retain fewer than the D1 minimum, while their earliest
    # H4 buckets still have enough data to be valid H4 candles.
    sparse_d1 = (frame.index.dayofyear % 31 == 0) & (frame.index.hour >= 11)
    frame = frame.loc[keep & ~sparse_d1].copy()
    # Flat BID observations are present in the source, but must not count toward
    # either minimum. Some H4 buckets have ONLY flat source observations.
    flat = (frame.index.hour == 17) | (
        (frame.index.dayofweek == 0) & (frame.index.hour >= 20)
    )
    for column in ("open", "high", "low"):
        frame.loc[flat, column] = frame.loc[flat, "close"]
    return frame


@pytest.fixture(scope="module")
def gappy_full(gappy: pd.DataFrame, config: BotConfig) -> pd.DataFrame:
    assert config.data.min_base_candles_per_h4 == 2
    assert config.data.min_base_candles_per_d1 == 12
    end = gappy.index[-1].floor("D") + pd.Timedelta(days=1)
    result = generate_signals(gappy, config, as_of=end)
    assert (result.direction == "long").any()
    assert (result.direction == "short").any()
    mature = result.loc[result.direction != "none"]
    assert mature[["atr", "d1_ema_fast", "d1_ema_slow", "d1_adx"]].notna().all().all()
    return result


def _nonflat(frame: pd.DataFrame) -> pd.DataFrame:
    flat = frame[["open", "high", "low", "close"]].eq(frame["close"], axis=0).all(axis=1)
    return frame.loc[~flat]


def test_gaps_flat_bars_and_sparse_buckets_are_not_padded_or_counted(
    gappy: pd.DataFrame, gappy_full: pd.DataFrame, config: BotConfig,
) -> None:
    assert "volume" not in gappy
    assert (gappy.index.dayofweek < 5).all()
    assert not (gappy.index.hour == 13).any()
    retained = _nonflat(gappy)
    assert len(retained) < len(gappy)
    h4_groups = retained.groupby(retained.index.floor("4h"))
    h4_counts = h4_groups.size()
    expected_index = h4_counts.index[h4_counts >= config.data.min_base_candles_per_h4]
    expected_index = expected_index.rename("candle_open")
    # Every emitted candle consists of retained observations. Missing buckets
    # must not be generated by filling, and sparse/flat rows cannot meet minima.
    pd.testing.assert_index_equal(gappy_full.index, expected_index)
    expected_close = h4_groups.close.last().loc[expected_index]
    pd.testing.assert_series_equal(
        gappy_full.close, expected_close.rename_axis("candle_open"), check_exact=True,
    )
    sparse_h4 = h4_counts.index[h4_counts < config.data.min_base_candles_per_h4]
    assert len(sparse_h4) > 0
    assert not gappy_full.index.isin(sparse_h4).any()
    raw_counts = gappy.groupby(gappy.index.floor("4h")).size()
    flat_only_buckets = raw_counts.index[
        (raw_counts >= config.data.min_base_candles_per_h4)
        & (h4_counts.reindex(raw_counts.index, fill_value=0) == 0)
    ]
    assert len(flat_only_buckets) > 0
    assert not gappy_full.index.isin(flat_only_buckets).any()
    d1_counts = retained.groupby(retained.index.floor("D")).size()
    sparse_d1 = d1_counts.index[d1_counts < config.data.min_base_candles_per_d1]
    assert len(sparse_d1) > 0
    assert not gappy_full.d1_open.isin(sparse_d1).any()
    assert (gappy_full.index.dayofweek < 5).all()


@pytest.mark.parametrize(
    "cutoff_text",
    [
        "2022-11-08T00:00:00Z",
        "2022-11-08T06:00:00Z",  # Already >= minimum H1 bars; H4 is not closed.
        "2022-11-08T14:30:00Z",  # Already >= minimum daily bars; D1 is not closed.
        "2023-06-13T00:00:00Z",
        "2023-06-13T06:00:00Z",
        "2023-06-13T14:30:00Z",
    ],
)
def test_gappy_history_full_vs_truncated_includes_all_indicator_alignment_values(
    gappy: pd.DataFrame, gappy_full: pd.DataFrame, config: BotConfig, cutoff_text: str,
) -> None:
    cutoff = pd.Timestamp(cutoff_text)
    actual = generate_signals(gappy.loc[gappy.index < cutoff], config, as_of=cutoff)
    assert_frame_equal(
        actual, gappy_full.loc[gappy_full.signal_time <= cutoff], check_exact=True,
    )
    assert actual.d1_ema_slow.notna().any()
    assert (actual.direction != "none").any()
    assert (actual.signal_time <= cutoff).all()
    aligned = actual.loc[actual.d1_close_time.notna()]
    assert (aligned.d1_close_time.to_numpy() < aligned.index.to_numpy()).all()
    if cutoff.hour == 6:
        forming_open = cutoff.floor("4h")
        closed_base = _nonflat(gappy.loc[gappy.index + pd.Timedelta(hours=1) <= cutoff])
        assert (closed_base.index >= forming_open).sum() >= config.data.min_base_candles_per_h4
        assert forming_open not in actual.index
    if cutoff.hour == 14:
        closed_base = _nonflat(gappy.loc[gappy.index + pd.Timedelta(hours=1) <= cutoff])
        assert (closed_base.index >= cutoff.floor("D")).sum() >= config.data.min_base_candles_per_d1


def test_gappy_future_mutation_and_flat_candle_removal_are_causal(
    gappy: pd.DataFrame, gappy_full: pd.DataFrame, config: BotConfig,
) -> None:
    cutoff = pd.Timestamp("2023-06-13T12:00:00Z")
    mutated = gappy.copy(deep=True)
    future = mutated.index >= cutoff
    mutated.loc[future, ["open", "high", "low", "close"]] = [150.0, 200.0, 100.0, 150.0]
    end = gappy.index[-1].floor("D") + pd.Timedelta(days=1)
    actual = generate_signals(mutated, config, as_of=end)
    assert_frame_equal(
        actual.loc[actual.signal_time <= cutoff],
        gappy_full.loc[gappy_full.signal_time <= cutoff],
        check_exact=True,
    )
    # Removing every flat source row ourselves must be equivalent to passing
    # them through the pipeline, including sample counts and subsequent ATR.
    without_flat = generate_signals(_nonflat(gappy), config, as_of=end)
    assert_frame_equal(without_flat, gappy_full, check_exact=True)
