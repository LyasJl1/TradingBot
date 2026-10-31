from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from tradingbot.data import (
    CandleValidationError,
    build_timeframes as _build_timeframes,
    load_candles,
    resample_candles as _resample_candles,
    load_bid_ask,
    find_gaps,
    validate_base_candles,
)


def build_timeframes(frame, base_timeframe, *, as_of):
    from tradingbot.config import load_config
    from pathlib import Path
    config = load_config(Path(__file__).parents[1] / "config.yaml")
    return _build_timeframes(frame, base_timeframe, as_of=as_of,
        min_base_candles_per_h4=config.data.min_base_candles_per_h4,
        min_base_candles_per_d1=config.data.min_base_candles_per_d1)


def resample_candles(frame, base_timeframe, target_timeframe, *, as_of):
    from tradingbot.config import load_config
    from pathlib import Path
    config = load_config(Path(__file__).parents[1] / "config.yaml")
    minimum = (config.data.min_base_candles_per_d1 if target_timeframe == "D1"
               else config.data.min_base_candles_per_h4)
    return _resample_candles(frame, base_timeframe, target_timeframe, as_of=as_of, min_base_candles=minimum)


def candles(periods: int = 48, frequency: str = "h", start: str = "2024-01-01") -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq=frequency, tz="UTC", name="timestamp")
    price = np.arange(periods, dtype=float) + 100
    return pd.DataFrame(
        {"open": price, "high": price + 2, "low": price - 1, "close": price + 1, "volume": np.arange(periods) + 1},
        index=index,
    )


def test_h4_and_daily_resampling_use_utc_open_labels_and_ohlcv() -> None:
    base = candles()
    result = build_timeframes(base, "H1", as_of=pd.Timestamp("2024-01-03T00:00Z"))
    assert len(result["H4"]) == 12
    assert len(result["D1"]) == 2
    assert result["H4"].iloc[0].to_dict() == {"open": 100, "high": 105, "low": 99, "close": 104}
    assert result["D1"].iloc[0].to_dict() == {"open": 100, "high": 125, "low": 99, "close": 124}
    assert result["D1"].index[0] == pd.Timestamp("2024-01-01T00:00Z")


def test_m15_builds_same_h4_boundaries() -> None:
    base = candles(96, "15min")
    result = build_timeframes(base, "M15", as_of=pd.Timestamp("2024-01-02T00:00Z"))
    assert len(result["H4"]) == 6
    assert result["H4"].iloc[0]["close"] == base.iloc[15]["close"]
    assert "volume" not in result["D1"]


def test_only_fully_closed_candles_are_emitted_and_future_is_ignored() -> None:
    base = candles()
    cutoff = pd.Timestamp("2024-01-01T09:30Z")
    full = resample_candles(base, "H1", "H4", as_of=cutoff)
    truncated = resample_candles(base.iloc[:9], "H1", "H4", as_of=cutoff)
    assert_frame_equal(full, truncated)
    assert list(full.index.hour) == [0, 4]
    assert (full.index + pd.Timedelta(hours=4) <= cutoff).all()


def test_as_of_exact_target_close_includes_that_candle() -> None:
    base = candles()
    before = resample_candles(base, "H1", "H4", as_of=pd.Timestamp("2024-01-01T03:59:59Z"))
    at_close = resample_candles(base, "H1", "H4", as_of=pd.Timestamp("2024-01-01T04:00Z"))
    assert before.empty
    assert len(at_close) == 1


def test_partial_boundary_groups_are_omitted_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    base = candles(28, start="2024-01-01T01:00")
    with caplog.at_level(logging.INFO, logger="tradingbot.data.candles"):
        result = resample_candles(base, "H1", "D1", as_of=pd.Timestamp("2024-01-02T05:00Z"))
    assert len(result) == 1
    assert result.attrs["resample_quality"] == {"built": 1, "dropped": 1, "dropped_below_minimum": 0, "dropped_not_closed": 1}
    assert "target_candle_omitted" in caplog.text
    assert "reason=not_closed" in caplog.text


def test_no_closed_base_candles_returns_typed_empty_frame() -> None:
    base = candles()
    result = resample_candles(base, "H1", "H4", as_of=pd.Timestamp("2023-12-31T23:00Z"))
    assert result.empty
    assert result.index.tz is not None
    assert list(result.columns) == ["open", "high", "low", "close"]


def test_validate_returns_copy_without_mutating_input() -> None:
    base = candles()
    copy = base.copy(deep=True)
    validated = validate_base_candles(base, "H1")
    validated.iloc[0, 0] = 999
    assert_frame_equal(base, copy)


def test_timestamp_resolution_does_not_affect_grid_validation() -> None:
    base = candles()
    base.index = base.index.as_unit("us")
    validated = validate_base_candles(base, "H1")
    assert len(validated) == len(base)


@pytest.mark.parametrize("change, message", [
    (lambda f: f.drop(columns="close"), "missing OHLC"),
    (lambda f: f.iloc[::-1], "ordered"),
    (lambda f: pd.concat([f.iloc[:1], f]), "duplicate"),
    (lambda f: f.set_axis(f.index.tz_localize(None)), "explicitly UTC"),
    (lambda f: f.set_axis(f.index.tz_convert("Europe/Paris")), "explicitly UTC"),
    (lambda f: f.set_axis(f.index + pd.Timedelta(minutes=1)), "aligned"),
    (lambda f: f.assign(high=f["open"] - 1), "inconsistent OHLC"),
    (lambda f: f.assign(low=f["close"] + 1), "inconsistent OHLC"),
    (lambda f: f.assign(close=np.nan), "nonfinite"),
    (lambda f: f.assign(spread=-1), "nonnegative"),
    (lambda f: f.assign(spread=np.inf), "nonfinite"),
    (lambda f: f.assign(close="bad"), "numeric"),
    (lambda f: f.assign(close=1 + 2j), "numeric"),
    (lambda f: f.iloc[:0], "empty"),
])
def test_validation_fails_explicitly(change, message: str) -> None:
    with pytest.raises(CandleValidationError, match=message):
        validate_base_candles(change(candles()), "H1")


def test_missing_weekend_is_allowed_without_filling() -> None:
    base = candles(96, start="2024-01-05")
    weekend = base.index.dayofweek >= 5
    result = validate_base_candles(base.loc[~weekend], "H1")
    assert result.index.equals(base.loc[~weekend].index)
    gaps = find_gaps(result, "H1")
    assert len(gaps) == 1
    assert gaps[0]["duration_hours"] == 48
    assert gaps[0]["warning"] is False


@pytest.mark.parametrize("timeframe", ["H4", "D1", "M30", "h1"])
def test_broker_h4_daily_or_unsupported_base_timeframe_is_rejected(timeframe: str) -> None:
    with pytest.raises(CandleValidationError, match="M15 or H1"):
        validate_base_candles(candles(), timeframe)


@pytest.mark.parametrize("as_of", [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-03", tz="Europe/Paris"), pd.NaT])
def test_as_of_must_be_explicit_utc(as_of: pd.Timestamp) -> None:
    with pytest.raises(CandleValidationError, match="as_of"):
        resample_candles(candles(), "H1", "H4", as_of=as_of)


def test_spread_is_optional_and_last_observation_is_preserved() -> None:
    base = candles().assign(spread=np.arange(48, dtype=float))
    with_spread = resample_candles(base, "H1", "H4", as_of=pd.Timestamp("2024-01-03T00:00Z"))
    assert with_spread.iloc[0]["spread"] == 3
    no_spread = resample_candles(base.drop(columns="spread"), "H1", "H4", as_of=pd.Timestamp("2024-01-03T00:00Z"))
    assert "spread" not in no_spread


def test_csv_roundtrip(tmp_path) -> None:
    source = candles().assign(spread=0.00012)
    path = tmp_path / "candles.csv"
    source.to_csv(path)
    loaded = load_candles(path, "H1")
    assert_frame_equal(loaded, source.drop(columns="volume"), check_freq=False)


@pytest.mark.parametrize("timezone", [None, "Europe/Paris"])
def test_csv_rejects_naive_or_non_utc_timestamps(tmp_path, timezone: str | None) -> None:
    source = candles()
    source.index = source.index.tz_localize(None) if timezone is None else source.index.tz_convert(timezone)
    path = tmp_path / "candles.csv"
    source.to_csv(path)
    with pytest.raises(CandleValidationError, match="UTC"):
        load_candles(path, "H1")


@pytest.mark.parametrize("index_as_column", [True, False])
def test_parquet_roundtrip(tmp_path, index_as_column: bool) -> None:
    source = candles().assign(spread=0.00012)
    path = tmp_path / "candles.parquet"
    (source.reset_index() if index_as_column else source).to_parquet(path)
    loaded = load_candles(path, "H1")
    assert_frame_equal(loaded, source.drop(columns="volume"), check_freq=False)


def test_csv_missing_timestamp_fails(tmp_path) -> None:
    path = tmp_path / "candles.csv"
    candles().to_csv(path, index=False)
    with pytest.raises(CandleValidationError, match="timestamp"):
        load_candles(path, "H1")


def test_unsupported_file_type_and_target_fail(tmp_path) -> None:
    with pytest.raises(CandleValidationError, match="CSV or Parquet"):
        load_candles(tmp_path / "candles.json", "H1")
    with pytest.raises(CandleValidationError, match="target_timeframe"):
        resample_candles(candles(), "H1", "H1", as_of=pd.Timestamp("2024-01-03T00:00Z"))


def write_pair(tmp_path, count=24):
    source = candles(count).drop(columns="volume")
    bid, ask = tmp_path / "bid.csv", tmp_path / "ask.csv"
    raw = source.reset_index()
    raw["timestamp"] = source.index.as_unit("ms").asi8
    raw.to_csv(bid, index=False)
    raw.loc[:, ["open", "high", "low", "close"]] += 0.25
    raw.to_csv(ask, index=False)
    return source, bid, ask


def test_epoch_milliseconds_are_explicitly_utc_and_iso_still_works(tmp_path):
    source, bid, _ = write_pair(tmp_path)
    result = load_candles(bid, "H1")
    assert_frame_equal(result, source, check_freq=False)
    mixed = pd.read_csv(bid)
    mixed["timestamp"] = mixed.timestamp.astype(str)
    mixed.loc[1, "timestamp"] = source.index[1].isoformat()
    mixed.to_csv(bid, index=False)
    assert_frame_equal(load_candles(bid, "H1"), source, check_freq=False)


@pytest.mark.parametrize("value", [1388534400000.5, True, "invalid", None])
def test_invalid_epoch_or_iso_timestamp_fails(tmp_path, value):
    _, bid, _ = write_pair(tmp_path)
    frame = pd.read_csv(bid).astype({"timestamp": object})
    frame.loc[0, "timestamp"] = value
    frame.to_csv(bid, index=False)
    with pytest.raises(CandleValidationError, match="timestamp|epoch"):
        load_candles(bid, "H1")


def test_volume_is_ignored_and_optional():
    expected = validate_base_candles(candles().drop(columns="volume"), "H1")
    for value in (np.nan, -1, "unused", True):
        assert_frame_equal(validate_base_candles(candles().assign(volume=value), "H1"), expected)


def test_bid_ask_prices_spread_and_source_counts(tmp_path):
    source, bid, ask = write_pair(tmp_path)
    actual = load_bid_ask(bid, ask, "H1")
    assert_frame_equal(actual.drop(columns="spread"), source, check_freq=False)
    assert (actual.spread == .25).all()
    assert actual.attrs["data_quality"]["bid_rows"] == 24
    assert actual.attrs["data_quality"]["ask_rows"] == 24


@pytest.mark.parametrize("mode", ["rows", "times"])
def test_bid_ask_alignment_is_checked_before_flat_filtering(tmp_path, mode):
    _, bid, ask = write_pair(tmp_path)
    frame = pd.read_csv(ask)
    if mode == "rows":
        frame = frame.iloc[:-1]
    else:
        frame.loc[len(frame)-1, "timestamp"] += 3600000
    frame.to_csv(ask, index=False)
    with pytest.raises(CandleValidationError, match="row count mismatch|timestamp mismatch"):
        load_bid_ask(bid, ask, "H1")


def test_negative_derived_spread_fails(tmp_path):
    _, bid, ask = write_pair(tmp_path)
    frame = pd.read_csv(ask)
    frame.loc[:, ["open", "high", "low", "close"]] -= 1
    frame.to_csv(ask, index=False)
    with pytest.raises(CandleValidationError, match="nonnegative"):
        load_bid_ask(bid, ask, "H1")


def test_flat_bid_rows_are_dropped_once_after_pairing(tmp_path):
    _, bid, ask = write_pair(tmp_path)
    frame = pd.read_csv(bid)
    frame.loc[3, ["open", "high", "low", "close"]] = frame.loc[3, "close"]
    frame.to_csv(bid, index=False)
    actual = load_bid_ask(bid, ask, "H1")
    assert len(actual) == 23
    assert actual.attrs["data_quality"]["flat_candles_dropped"] == 1
    again = validate_base_candles(actual, "H1")
    assert again.attrs["data_quality"]["flat_candles_dropped"] == 1
    assert again.attrs["data_quality"]["input_rows"] == 24


def test_flat_ask_does_not_drop_nonflat_bid(tmp_path):
    _, bid, ask = write_pair(tmp_path)
    frame = pd.read_csv(ask)
    frame.loc[3, ["open", "high", "low", "close"]] = frame.loc[3, "close"]
    frame.to_csv(ask, index=False)
    assert len(load_bid_ask(bid, ask, "H1")) == 24


@pytest.mark.parametrize("hours,weekday,warning", [(4, True, False), (5, True, True), (48, False, False)])
def test_gap_warning_threshold_and_start_convention(hours, weekday, warning):
    start = pd.Timestamp("2024-01-01T01:00Z" if weekday else "2024-01-06T00:00Z")
    frame = candles(2)
    frame.index = pd.DatetimeIndex([start-pd.Timedelta(hours=1), start+pd.Timedelta(hours=hours)])
    gaps = find_gaps(validate_base_candles(frame, "H1"), "H1")
    assert len(gaps) == 1
    assert gaps[0]["start"] == start.isoformat()
    assert gaps[0]["duration_hours"] == hours
    assert gaps[0]["missing_base_candles"] == hours
    assert gaps[0]["warning"] is warning


def test_sparse_closed_buckets_use_minimum_and_do_not_fill():
    frame = candles(24).iloc[[0, 2, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16]]
    actual = build_timeframes(frame, "H1", as_of=pd.Timestamp("2024-01-02T00:00Z"))
    assert list(actual["H4"].index.hour) == [0, 8, 12]
    assert actual["H4"].iloc[0].close == frame.loc[pd.Timestamp("2024-01-01T02:00Z"), "close"]
    assert actual["H4"].attrs["resample_quality"]["dropped_below_minimum"] == 2
    assert len(actual["D1"]) == 1


def test_all_flat_input_counts_original_buckets_and_keeps_bounds():
    raw = candles(24)
    for column in ("open", "high", "low"):
        raw[column] = raw.close
    clean = validate_base_candles(raw, "H1")
    assert clean.empty
    assert clean.attrs["data_quality"]["flat_candles_dropped"] == 24
    assert clean.attrs["data_quality"]["first_timestamp"] == raw.index[0].isoformat()
    frames = build_timeframes(clean, "H1", as_of=pd.Timestamp("2024-01-02T00:00Z"))
    assert frames["H4"].attrs["resample_quality"] == {"built": 0, "dropped": 6, "dropped_below_minimum": 6, "dropped_not_closed": 0}
    assert frames["D1"].attrs["resample_quality"]["dropped"] == 1


@pytest.mark.parametrize("minimum", [0, -1, 1.5, True, 5])
def test_invalid_minimum_fails(minimum):
    with pytest.raises(CandleValidationError, match="min_base_candles"):
        _resample_candles(candles(), "H1", "H4", as_of=pd.Timestamp("2024-01-03T00:00Z"), min_base_candles=minimum)
