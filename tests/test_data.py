from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from tradingbot.data import (
    CandleValidationError,
    build_timeframes,
    load_candles,
    resample_candles,
    validate_base_candles,
)


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
    assert result["H4"].iloc[0].to_dict() == {"open": 100, "high": 105, "low": 99, "close": 104, "volume": 10}
    assert result["D1"].iloc[0].to_dict() == {"open": 100, "high": 125, "low": 99, "close": 124, "volume": 300}
    assert result["D1"].index[0] == pd.Timestamp("2024-01-01T00:00Z")


def test_m15_builds_same_h4_boundaries() -> None:
    base = candles(96, "15min")
    result = build_timeframes(base, "M15", as_of=pd.Timestamp("2024-01-02T00:00Z"))
    assert len(result["H4"]) == 6
    assert result["H4"].iloc[0]["close"] == base.iloc[15]["close"]
    assert result["D1"].iloc[0]["volume"] == base["volume"].sum()


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
    assert result.empty
    assert "target_candle_omitted" in caplog.text
    assert "incomplete_or_not_closed" in caplog.text


def test_no_closed_base_candles_returns_typed_empty_frame() -> None:
    base = candles()
    result = resample_candles(base, "H1", "H4", as_of=pd.Timestamp("2023-12-31T23:00Z"))
    assert result.empty
    assert result.index.tz is not None
    assert list(result.columns) == list(base.columns)


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
    (lambda f: f.drop(columns="volume"), "missing OHLCV"),
    (lambda f: f.iloc[::-1], "ordered"),
    (lambda f: pd.concat([f.iloc[:1], f]), "duplicate"),
    (lambda f: f.drop(f.index[3]), "missing base candles"),
    (lambda f: f.set_axis(f.index.tz_localize(None)), "explicitly UTC"),
    (lambda f: f.set_axis(f.index.tz_convert("Europe/Paris")), "explicitly UTC"),
    (lambda f: f.set_axis(f.index + pd.Timedelta(minutes=1)), "aligned"),
    (lambda f: f.assign(high=f["open"] - 1), "inconsistent OHLC"),
    (lambda f: f.assign(low=f["close"] + 1), "inconsistent OHLC"),
    (lambda f: f.assign(close=np.nan), "nonfinite"),
    (lambda f: f.assign(volume=-1), "nonnegative"),
    (lambda f: f.assign(spread=-1), "nonnegative"),
    (lambda f: f.assign(spread=np.inf), "nonfinite"),
    (lambda f: f.assign(close="bad"), "numeric"),
    (lambda f: f.assign(volume=True), "numeric"),
    (lambda f: f.assign(close=1 + 2j), "numeric"),
    (lambda f: f.iloc[:0], "empty"),
])
def test_validation_fails_explicitly(change, message: str) -> None:
    with pytest.raises(CandleValidationError, match=message):
        validate_base_candles(change(candles()), "H1")


def test_missing_weekend_is_not_silently_assumed_to_be_a_market_closure() -> None:
    base = candles(96, start="2024-01-05")
    weekend = base.index.dayofweek >= 5
    with pytest.raises(CandleValidationError, match="no market calendar is assumed"):
        validate_base_candles(base.loc[~weekend], "H1")


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
    assert_frame_equal(loaded, source, check_freq=False)


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
    assert_frame_equal(loaded, source, check_freq=False)


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
