"""Validated UTC OHLC data, paired BID/ASK loading and unfilled resampling."""

from __future__ import annotations

import logging
from numbers import Real
from pathlib import Path
import re

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
_BASE_PERIODS = {"M15": pd.Timedelta(minutes=15), "H1": pd.Timedelta(hours=1)}
_TARGET_PERIODS = {"H4": pd.Timedelta(hours=4), "D1": pd.Timedelta(days=1)}
_OHLC = ["open", "high", "low", "close"]
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


def _validated_raw(frame: pd.DataFrame, base_timeframe: str, *, allow_empty: bool = False) -> pd.DataFrame:
    period = _base_period(base_timeframe)
    if not isinstance(frame, pd.DataFrame):
        raise CandleValidationError("candles must be a pandas DataFrame")
    if frame.empty and not allow_empty:
        raise CandleValidationError("base candle data is empty")
    _validate_index(frame)
    if frame.columns.has_duplicates:
        raise CandleValidationError("duplicate candle columns")
    missing = set(_OHLC).difference(frame.columns)
    if missing:
        raise CandleValidationError(f"missing OHLC columns: {', '.join(sorted(missing))}")
    if not frame.index.equals(frame.index.floor(period)):
        raise CandleValidationError(f"timestamps are not aligned to UTC {base_timeframe} boundaries")
    columns = _OHLC + (["spread"] if "spread" in frame else [])
    result = frame.loc[:, columns].copy()
    for name in columns:
        series = result[name]
        if (not pd.api.types.is_numeric_dtype(series)
                or pd.api.types.is_bool_dtype(series)
                or pd.api.types.is_complex_dtype(series)):
            raise CandleValidationError(f"{name} must contain numeric values")
        if not np.isfinite(series.to_numpy(dtype=float, na_value=np.nan)).all():
            raise CandleValidationError(f"{name} contains missing or nonfinite values")
    if "spread" in result and (result.spread < 0).any():
        raise CandleValidationError("spread must be nonnegative when supplied")
    invalid = ((result.high < result.low) | (result.high < result.open)
               | (result.high < result.close) | (result.low > result.open)
               | (result.low > result.close))
    if invalid.any():
        raise CandleValidationError(f"inconsistent OHLC at {result.index[invalid][0].isoformat()}")
    result.index = result.index.tz_convert("UTC").rename("timestamp")
    return result


def validate_base_candles(frame: pd.DataFrame, base_timeframe: str) -> pd.DataFrame:
    """Validate and drop flat BID rows. Gaps are allowed; volume is ignored.

    Metadata retains original row counts and timestamp bounds across repeated
    validation. No missing prices, rows, volume or spread are invented.
    """
    prior = dict(frame.attrs.get("data_quality", {})) if isinstance(frame, pd.DataFrame) else {}
    raw = _validated_raw(frame, base_timeframe, allow_empty=bool(prior.get("input_rows", 0)))
    metadata = prior or {
        "input_rows": len(raw), "flat_candles_dropped": 0,
        "first_timestamp": raw.index[0].isoformat() if len(raw) else None,
        "last_timestamp": raw.index[-1].isoformat() if len(raw) else None,
    }
    flat = raw[_OHLC].eq(raw.close, axis=0).all(axis=1)
    count = int(flat.sum())
    metadata["flat_candles_dropped"] = int(metadata.get("flat_candles_dropped", 0)) + count
    result = raw.loc[~flat].copy()
    result.attrs["data_quality"] = metadata
    if count:
        LOGGER.info("base_candles_dropped reason=flat_bid count=%d", count)
    return result


def _timestamp_index(values: pd.Series) -> pd.DatetimeIndex:
    parsed: list[pd.Timestamp] = []
    for value in values:
        try:
            numeric = isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
            numeric_text = isinstance(value, str) and bool(re.fullmatch(r"[+-]?\d+", value))
            if numeric or numeric_text:
                number = float(value)
                if not np.isfinite(number) or not number.is_integer():
                    raise CandleValidationError("epoch-millisecond timestamps must be finite integers")
                timestamp = pd.to_datetime(int(value), unit="ms", utc=True)
            else:
                if isinstance(value, (bool, np.bool_)):
                    raise CandleValidationError("boolean timestamp is invalid")
                timestamp = pd.Timestamp(value)
                if pd.isna(timestamp) or timestamp.tzinfo is None:
                    raise CandleValidationError("timestamp column must contain explicitly UTC timestamps")
                if timestamp.utcoffset() != pd.Timedelta(0):
                    raise CandleValidationError("timestamp column contains a non-UTC offset")
                timestamp = timestamp.tz_convert("UTC")
            if pd.isna(timestamp):
                raise CandleValidationError("timestamp column contains NaT")
            parsed.append(timestamp)
        except (TypeError, ValueError, OverflowError) as error:
            if isinstance(error, CandleValidationError):
                raise
            raise CandleValidationError("timestamp column contains an invalid UTC timestamp or epoch milliseconds") from error
    return pd.DatetimeIndex(parsed, tz="UTC", name="timestamp")


def _read_candles(path: str | Path, base_timeframe: str) -> pd.DataFrame:
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
    return _validated_raw(frame, base_timeframe)


def load_candles(path: str | Path, base_timeframe: str) -> pd.DataFrame:
    """Load UTC epoch-ms/ISO OHLC CSV or Parquet; drop flats, ignore volume."""
    return validate_base_candles(_read_candles(path, base_timeframe), base_timeframe)


def load_bid_ask(bid_path: str | Path, ask_path: str | Path, base_timeframe: str) -> pd.DataFrame:
    """Require identical RAW rows/times, use BID OHLC and ASK-close minus BID-close.

    Spread is a close-based estimate in price units. The BID flat mask alone is
    applied AFTER pairing; no unmatched dates are intersected or repaired.
    """
    bid = _read_candles(bid_path, base_timeframe)
    ask = _read_candles(ask_path, base_timeframe)
    if len(bid) != len(ask):
        raise CandleValidationError(f"Bid/Ask row count mismatch: bid={len(bid)}, ask={len(ask)}")
    if not bid.index.equals(ask.index):
        raise CandleValidationError("Bid/Ask timestamp mismatch: raw timestamp sequences must be identical")
    merged = bid.loc[:, _OHLC].copy()
    merged["spread"] = ask.close.to_numpy() - bid.close.to_numpy()
    result = validate_base_candles(merged, base_timeframe)
    result.attrs["data_quality"].update(bid_rows=len(bid), ask_rows=len(ask))
    return result


def find_gaps(frame: pd.DataFrame, base_timeframe: str) -> list[dict[str, object]]:
    """Report missing intervals after cleaning, without filling or rejecting.

    Start is the FIRST missing open; duration ends at the next observed open.
    Warn only when this missing duration exceeds four hours and starts Mon-Fri.
    """
    period = _base_period(base_timeframe)
    _validate_index(frame)
    result: list[dict[str, object]] = []
    for previous, following in zip(frame.index[:-1], frame.index[1:]):
        if following - previous > period:
            start = previous + period
            duration = following - start
            result.append({
                "previous_timestamp": previous.isoformat(), "next_timestamp": following.isoformat(),
                "start": start.isoformat(), "end": following.isoformat(),
                "duration_hours": duration.total_seconds() / 3600,
                "missing_base_candles": int(duration / period),
                "warning": bool(duration > pd.Timedelta(hours=4) and start.dayofweek < 5),
            })
    return result


def resample_candles(
    frame: pd.DataFrame, base_timeframe: str, target_timeframe: str, *,
    as_of: pd.Timestamp, min_base_candles: int,
) -> pd.DataFrame:
    """Aggregate existing closed base bars; minimum coverage never closes a bucket.

    Empty buckets inside original input bounds count as dropped. No OHLC rows are
    manufactured. Metadata distinguishes below-minimum and not-yet-closed drops.
    """
    base = _base_period(base_timeframe)
    if target_timeframe not in _TARGET_PERIODS:
        raise CandleValidationError("target_timeframe must be H4 or D1")
    period = _TARGET_PERIODS[target_timeframe]
    if type(min_base_candles) is not int or not 1 <= min_base_candles <= int(period / base):
        raise CandleValidationError("min_base_candles must be a positive integer within bucket capacity")
    cutoff = pd.Timestamp(as_of)
    if pd.isna(cutoff) or cutoff.tzinfo is None or str(cutoff.tzinfo) not in _UTC_NAMES:
        raise CandleValidationError("as_of must be an explicit UTC timestamp")
    if not isinstance(frame, pd.DataFrame):
        raise CandleValidationError("candles must be a pandas DataFrame")
    _validate_index(frame)
    metadata = frame.attrs.get("data_quality", {})
    if frame.empty and not metadata.get("input_rows", 0):
        raise CandleValidationError("base candle data is empty")
    # Filter before reading values so future/forming base prices cannot leak in.
    source = frame.loc[frame.index + base <= cutoff]
    if source.empty:
        cleaned = _validated_raw(source, base_timeframe, allow_empty=True)
    else:
        cleaned = validate_base_candles(source, base_timeframe)
    first = pd.Timestamp(metadata.get("first_timestamp") or frame.index[0]) if len(frame) or metadata else None
    last = pd.Timestamp(metadata.get("last_timestamp") or frame.index[-1]) if len(frame) or metadata else None
    last_closed = cutoff.floor(base) - base
    last = min(last, last_closed) if last is not None else None
    buckets = (pd.date_range(first.floor(period), last.floor(period), freq=period, name="timestamp")
               if first is not None and last is not None and last >= first else
               pd.DatetimeIndex([], tz="UTC", name="timestamp"))
    aggregation = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "spread" in cleaned:
        aggregation["spread"] = "last"
    groups = cleaned.resample(period, origin="epoch", closed="left", label="left")
    aggregated = groups.agg(aggregation).reindex(buckets)
    counts = groups.size().reindex(buckets, fill_value=0)
    closed = buckets + period <= cutoff
    enough = counts >= min_base_candles
    keep = closed & enough
    result = aggregated.loc[keep].copy()
    result.index.name = "timestamp"
    not_closed = int((~closed).sum())
    below = int((closed & ~enough).sum())
    result.attrs["resample_quality"] = {
        "built": len(result), "dropped": not_closed + below,
        "dropped_below_minimum": below, "dropped_not_closed": not_closed,
    }
    for timestamp in buckets[~keep]:
        reason = "not_closed" if timestamp + period > cutoff else "below_minimum"
        LOGGER.info("target_candle_omitted reason=%s timeframe=%s timestamp=%s count=%d minimum=%d",
                    reason, target_timeframe, timestamp.isoformat(), counts.loc[timestamp], min_base_candles)
    return result


def build_timeframes(
    frame: pd.DataFrame, base_timeframe: str, *, as_of: pd.Timestamp,
    min_base_candles_per_h4: int, min_base_candles_per_d1: int,
) -> dict[str, pd.DataFrame]:
    """Use one resampler and explicit configured minima in every mode."""
    return {
        target: resample_candles(frame, base_timeframe, target, as_of=as_of, min_base_candles=minimum)
        for target, minimum in (("H4", min_base_candles_per_h4), ("D1", min_base_candles_per_d1))
    }
