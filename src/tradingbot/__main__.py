"""Local Phase 1 signal CSV and paired-source data-quality commands."""

import argparse
import json
import sys

import pandas as pd

from tradingbot.config import load_config
from tradingbot.data import load_bid_ask, load_candles
from tradingbot.data.quality import quality_reports
from tradingbot.signals import generate_signals


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "data-quality":
        _data_quality(sys.argv[2:])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", help="Single-symbol UTC M15/H1 OHLC CSV or Parquet")
    parser.add_argument("--bid", help="Dukascopy H1 BID CSV")
    parser.add_argument("--ask", help="Dukascopy H1 ASK CSV")
    parser.add_argument("--as-of", required=True, help="Explicit UTC cutoff, e.g. 2026-01-01T00:00:00Z")
    args = parser.parse_args()
    if bool(args.bid) != bool(args.ask):
        parser.error("--bid and --ask must be supplied together")
    if args.data and args.bid:
        parser.error("--data cannot be combined with --bid/--ask")
    if not args.data and not args.bid:
        parser.error("provide --data or both --bid and --ask")
    try:
        config = load_config(args.config)
        if args.bid:
            if config.data.base_timeframe != "H1":
                raise ValueError("Dukascopy paired signal input requires base_timeframe H1")
            data = load_bid_ask(args.bid, args.ask, "H1")
        else:
            data = load_candles(args.data, config.data.base_timeframe)
        decisions = generate_signals(data, config, as_of=pd.Timestamp(args.as_of))
        decisions.to_csv(sys.stdout)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


def _data_quality(arguments: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="tradingbot data-quality", description="Paired Dukascopy H1 quality report")
    parser.add_argument("--config", required=True)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--symbol", action="append", dest="symbols")
    parser.add_argument("--as-of", help="Explicit UTC cutoff; default last original H1 open plus one hour")
    args = parser.parse_args(arguments)
    try:
        config = load_config(args.config)
        reports = quality_reports(args.raw_dir, config, symbols=args.symbols,
                                  as_of=pd.Timestamp(args.as_of) if args.as_of is not None else None)
        json.dump(reports, sys.stdout, indent=2, allow_nan=False)
        sys.stdout.write("\n")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
