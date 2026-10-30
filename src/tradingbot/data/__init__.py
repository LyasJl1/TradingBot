"""Validated UTC base candles and shared H4/D1 resampling.

Timestamps label candle opens. Resampling requires an explicit UTC observation
time and emits only complete, closed target candles. Missing periods, including
weekends, are rejected: a market calendar has deliberately not been inferred.
"""

from .candles import (
    CandleValidationError,
    build_timeframes,
    load_candles,
    resample_candles,
    validate_base_candles,
)

__all__ = [
    "CandleValidationError",
    "build_timeframes",
    "load_candles",
    "resample_candles",
    "validate_base_candles",
]
