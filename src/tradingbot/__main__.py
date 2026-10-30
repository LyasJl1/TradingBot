"""Local Phase 1 CSV/Parquet -> signal-decision CSV command."""

import argparse
import sys

import pandas as pd

from tradingbot.config import load_config
from tradingbot.data import load_candles
from tradingbot.signals import generate_signals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True, help="Single-symbol UTC M15/H1 OHLCV CSV or Parquet")
    parser.add_argument("--as-of", required=True, help="Explicit UTC cutoff, e.g. 2026-01-01T00:00:00Z")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        data = load_candles(args.data, config.data.base_timeframe)
        decisions = generate_signals(data, config, as_of=pd.Timestamp(args.as_of))
        decisions.to_csv(sys.stdout)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
