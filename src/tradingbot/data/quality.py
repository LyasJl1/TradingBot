"""Local data-quality reports for paired Dukascopy H1 files."""

from pathlib import Path
from typing import Sequence

import pandas as pd

from tradingbot.config import BotConfig
from .candles import build_timeframes, find_gaps, load_bid_ask


def quality_reports(
    raw_dir: str | Path,
    config: BotConfig,
    *,
    symbols: Sequence[str] | None = None,
    as_of: pd.Timestamp | None = None,
) -> list[dict[str, object]]:
    """Report raw/retained data and closed target buckets, without filling gaps.

    With no explicit cutoff, use the last original H1 open plus one hour. This
    closes that base candle but does not prematurely close its H4/D1 bucket.
    Raw timestamp bounds are kept separate from retained (nonflat BID) bounds.
    """
    if config.data.base_timeframe != "H1":
        raise ValueError("Dukascopy paired quality reports require base_timeframe H1")
    directory = Path(raw_dir)
    if not directory.is_dir():
        raise ValueError(f"raw data directory does not exist: {directory}")
    if symbols is None:
        discovered: set[str] = set()
        for side in ("bid", "ask"):
            suffix = f"_H1_{side}.csv"
            discovered.update(path.name[:-len(suffix)] for path in directory.glob(f"*{suffix}"))
        selected = sorted(discovered)
    else:
        selected = list(dict.fromkeys(symbols))
    if not selected:
        raise ValueError(f"no Dukascopy H1 Bid/Ask CSV files found in {directory}")
    reports: list[dict[str, object]] = []
    for symbol in selected:
        if not symbol or any(character in symbol for character in ("/", "\\")):
            raise ValueError("symbol must be a nonempty filename prefix without path separators")
        bid_path = directory / f"{symbol}_H1_bid.csv"
        ask_path = directory / f"{symbol}_H1_ask.csv"
        missing = [str(path) for path in (bid_path, ask_path) if not path.is_file()]
        if missing:
            raise ValueError("missing Bid/Ask file(s): " + ", ".join(missing))
        frame = load_bid_ask(bid_path, ask_path, "H1")
        metadata = frame.attrs["data_quality"]
        cutoff = pd.Timestamp(as_of) if as_of is not None else (
            pd.Timestamp(metadata["last_timestamp"]) + pd.Timedelta(hours=1)
        )
        frames = build_timeframes(
            frame, "H1", as_of=cutoff,
            min_base_candles_per_h4=config.data.min_base_candles_per_h4,
            min_base_candles_per_d1=config.data.min_base_candles_per_d1,
        )
        gaps = find_gaps(frame, "H1")
        warnings = [
            f"gap {gap['start']} to {gap['end']}: {gap['duration_hours']:g} missing hours "
            "starting Monday-Friday UTC"
            for gap in gaps if gap["warning"]
        ]
        reports.append({
            "symbol": symbol,
            "bid_rows": int(metadata["bid_rows"]),
            "ask_rows": int(metadata["ask_rows"]),
            "input_rows": int(metadata["input_rows"]),
            "rows_retained": len(frame),
            "first_timestamp": metadata["first_timestamp"],
            "last_timestamp": metadata["last_timestamp"],
            "first_retained_timestamp": frame.index[0].isoformat() if len(frame) else None,
            "last_retained_timestamp": frame.index[-1].isoformat() if len(frame) else None,
            "flat_candles_dropped": int(metadata["flat_candles_dropped"]),
            "gaps_found": len(gaps),
            "gaps": gaps,
            "warnings": warnings,
            "as_of": cutoff.isoformat(),
            "H4": frames["H4"].attrs["resample_quality"],
            "D1": frames["D1"].attrs["resample_quality"],
        })
    return reports
