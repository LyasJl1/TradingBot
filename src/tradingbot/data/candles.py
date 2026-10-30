"""Data ingestion and conservative candle construction, with no broker access."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
_BASE_PERIODS = {"M15": pd.Timedelta(minutes=15), "H1": pd.Timedelta(hours=1)}
_TARGET_PERIODS = {"H4": pd.Timedelta(hours=4), "D1": pd.Timedelta(days=1)}
_OHLCV = ("open", "high", "low", "close", "volume")
_UTC_NAMES = {"UTC", "Etc/UTC", "Etc/GMT", "GMT", "UTC+00:00", "+00:00"}


class CandleValidationError(ValueError):
    """Input cannot safely represent the requested base candles."""


def _base_period(base_timeframe: str) -> pd.Timedelta:
    if base_timeframe not in _BASE_PERIODS:
        raise CandleValidationError("base_timeframe must be M15 or H1 to build H4/D1")
    return _BASE_PERIODS[base_timeframe]


def _validate_index(frame: pd.DataFrame) -> None:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise CandleValidationError("candles require a timezone-aware UTC DatetimeIndex")
    if frame.index.tz is None or str(frame.index.tz) not in _UTC_NAMES:
        raise CandleValidationError("candle timestamps must be explicitly UTC")
    if frame.index.hasnans:
        raise CandleValidationError("candle timestamps must not contain NaT")
    if frame.index.has_duplicates:
        raise CandleValidationError("duplicate candle timestamps")
    if not frame.index.is_monotonic_increasing:
        raise CandleValidationError("candle timestamps must be ordered, oldest first")


def validate_base_candles(frame: pd.DataFrame, base_timeframe: str) -> pd.DataFrame:
    """Return a validated copy without sorting, filling, or repairing input.

    A continuous base-timeframe grid is required. Even a scheduled market closure
    is reported as missing data, because no exchange calendar is assumed here.
    Volume and optional spread must be finite and nonnegative. Missing spread
    remains absent; this layer supplies no cost defaults.
    """
    period = _base_period(base_timeframe)
    if not isinstance(frame, pd.DataFrame):
        raise CandleValidationError("candles must be a pandas DataFrame")
    if frame.empty:
        raise CandleValidationError("base candle data is empty")
    _validate_index(frame)
    if frame.columns.has_duplicates:
        raise CandleValidationError("duplicate candle columns")
    missing = set(_OHLCV).difference(frame.columns)
    if missing:
        raise CandleValidationError(f"missing OHLCV columns: {', '.join(sorted(missing))}")
    if not frame.index.equals(frame.index.floor(period)):
        raise CandleValidationError(f"timestamps are not aligned to UTC {base_timeframe} boundaries")
    differences = frame.index.to_series().diff().iloc[1:]
    bad_gaps = differences[differences != period]
    if not bad_gaps.empty:
        first = bad_gaps.index[0]
        raise CandleValidationError(
            f"missing base candles before {first.isoformat()}: "
            f"expected {period}, found {bad_gaps.iloc[0]}; no market calendar is assumed"
        )
    columns = list(_OHLCV) + (["spread"] if "spread" in frame else [])
    for name in columns:
        series = frame[name]
        if (
            not pd.api.types.is_numeric_dtype(series)
            or pd.api.types.is_bool_dtype(series)
            or pd.api.types.is_complex_dtype(series)
        ):
            raise CandleValidationError(f"{name} must contain numeric values")
        if not np.isfinite(series.to_numpy(dtype=float)).all():
            raise CandleValidationError(f"{name} contains missing or nonfinite values")
    if (frame["volume"] < 0).any():
        raise CandleValidationError("volume must be nonnegative")
    if "spread" in frame and (frame["spread"] < 0).any():
        raise CandleValidationError("spread must be nonnegative when supplied")
    invalid_ohlc = (
        (frame["high"] < frame["low"])
        | (frame["high"] < frame["open"])
        | (frame["high"] < frame["close"])
        | (frame["low"] > frame["open"])
        | (frame["low"] > frame["close"])
    )
    if invalid_ohlc.any():
        timestamp = frame.index[invalid_ohlc][0]
        raise CandleValidationError(f"inconsistent OHLC at {timestamp.isoformat()}")
    result = frame.copy()
    result.index = result.index.tz_convert("UTC").rename("timestamp")
    return result


def _timestamp_index(values: pd.Series) -> pd.DatetimeIndex:
    parsed: list[pd.Timestamp] = []
    for value in values:
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError) as error:
            raise CandleValidationError("timestamp column contains an invalid timestamp") from error
        if pd.isna(timestamp) or timestamp.tzinfo is None:
            raise CandleValidationError("timestamp column must contain explicitly UTC timestamps")
        if timestamp.utcoffset() != pd.Timedelta(0):
            raise CandleValidationError("timestamp column contains a non-UTC offset")
        parsed.append(timestamp.tz_convert("UTC"))
    return pd.DatetimeIndex(parsed, tz="UTC", name="timestamp")


def load_candles(path: str | Path, base_timeframe: str) -> pd.DataFrame:
    """Load CSV or Parquet with timestamp and lowercase OHLCV columns.

    CSV timestamps must include Z or +00:00; naive timestamps are never assumed
    to be UTC. Parquet may alternatively store a named UTC DatetimeIndex.
    """
    _base_period(base_timeframe)
    source = Path(path)
    try:
        if source.suffix.lower() == ".csv":
            frame = pd.read_csv(source)
        elif source.suffix.lower() in {".parquet", ".pq"}:
            frame = pd.read_parquet(source)
        else:
            raise CandleValidationError("candle file must be CSV or Parquet")
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        raise CandleValidationError(f"cannot parse candle file {source.name}") from error
    if "timestamp" in frame.columns:
        frame.index = _timestamp_index(frame.pop("timestamp"))
    elif not isinstance(frame.index, pd.DatetimeIndex):
        raise CandleValidationError("candle file is missing the timestamp column")
    return validate_base_candles(frame, base_timeframe)


def resample_candles(
    frame: pd.DataFrame,
    base_timeframe: str,
    target_timeframe: str,
    *,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """Build complete closed H4 or D1 candles using UTC open labels.

    Incomplete leading/trailing groups and the currently forming candle are
    omitted with a reason in the standard logger. Supplied future data is ignored.
    Optional spread is the last base-candle observation, not an invented cost or
    a guarantee about a later execution price.
    """
    base_period = _base_period(base_timeframe)
    if target_timeframe not in _TARGET_PERIODS:
        raise CandleValidationError("target_timeframe must be H4 or D1")
    try:
        cutoff = pd.Timestamp(as_of)
    except (TypeError, ValueError) as error:
        raise CandleValidationError("as_of must be an explicit UTC timestamp") from error
    if pd.isna(cutoff) or cutoff.tzinfo is None or str(cutoff.tz) not in _UTC_NAMES:
        raise CandleValidationError("as_of must be an explicit UTC timestamp")
    if not isinstance(frame, pd.DataFrame):
        raise CandleValidationError("candles must be a pandas DataFrame")
    if frame.empty:
        raise CandleValidationError("base candle data is empty")
    _validate_index(frame)
    closed = frame.loc[frame.index + base_period <= cutoff]
    if len(closed) != len(frame):
        LOGGER.info("base_candles_omitted reason=not_closed_as_of count=%d as_of=%s", len(frame) - len(closed), cutoff.isoformat())
    # Validate the schema even when as_of precedes all observations.
    validated = validate_base_candles(closed if not closed.empty else frame.iloc[:1], base_timeframe)
    if closed.empty:
        return validated.iloc[:0].loc[:, list(_OHLCV) + (["spread"] if "spread" in validated else [])]
    period = _TARGET_PERIODS[target_timeframe]
    aggregation = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    if "spread" in validated:
        aggregation["spread"] = "last"
    groups = validated.resample(period, origin="epoch", closed="left", label="left")
    result = groups.agg(aggregation)
    counts = groups.size()
    expected_count = int(period / base_period)
    complete = (counts == expected_count) & (result.index + period <= cutoff)
    for timestamp in result.index[~complete]:
        LOGGER.info(
            "target_candle_omitted reason=incomplete_or_not_closed timeframe=%s timestamp=%s count=%d expected=%d as_of=%s",
            target_timeframe, timestamp.isoformat(), counts.loc[timestamp], expected_count, cutoff.isoformat(),
        )
    result = result.loc[complete].copy()
    result.index.name = "timestamp"
    return result


def build_timeframes(
    frame: pd.DataFrame, base_timeframe: str, *, as_of: pd.Timestamp
) -> dict[str, pd.DataFrame]:
    """Use the same resampler for both strategy timeframes in every mode."""
    return {
        target: resample_candles(frame, base_timeframe, target, as_of=as_of)
        for target in _TARGET_PERIODS
    }
