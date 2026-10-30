"""Pure closed-candle decisions; no account, position or broker access."""

from .core import breakout_signal, evaluate_candles, generate_signals, trend_filter

__all__ = ["breakout_signal", "evaluate_candles", "generate_signals", "trend_filter"]
