# TradingBot

Phase 1 of [SPEC.md](SPEC.md): validated base data, UTC H4/D1 resampling,
indicators and pure Donchian breakout candidate signals. No backtest, risk or
execution code is included.

Requires Python 3.11+; tested on Python 3.12. Install and test:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest
```

All strategy periods and thresholds are in `config.yaml`. `paper_mode` defaults
to `true`; this phase has no order submission capability.

Produce decision records from a **single symbol's M15/H1 base data**:

```sh
.venv/bin/python -m tradingbot --config config.yaml \
  --data /path/to/eurusd_h1.csv --as-of 2026-01-01T00:00:00Z > decisions.csv
```

CSV requires `timestamp,open,high,low,close`, optionally `spread`.
Volume is optional and unused. Numeric timestamps are UTC Unix epoch
**milliseconds**; explicit UTC ISO candle **opens** remain supported. Parquet
is also supported. Both timeframes are built internally from closed base bars.

For Dukascopy H1, put separate BID/ASK files in `data/raw/` and report quality:

```sh
.venv/bin/python -m tradingbot data-quality --config config.yaml --raw-dir data/raw
```

Expected names: `{SYMBOL}_H1_bid.csv` and `{SYMBOL}_H1_ask.csv`. Timestamps and
row counts must match. Candles use BID prices; `ask close - bid close` is a
close-based spread estimate. For paired signal input, use `--bid FILE --ask FILE`
instead of `--data FILE`.

Weekend/holiday gaps are allowed and never filled. Flat BID candles are dropped.
Closed H4/D1 buckets require the configured minima (2/12 retained base candles);
insufficient buckets are dropped and counted. The report includes gap warnings.
Raw data and CSV files are ignored by Git.

Signal output contains raw candidates and reasons; later phases
must apply entry filters, position constraints, risk and execution checks.

See [Phase 1 notes](docs/phase1.md) for seeding, conservative D1 alignment,
assumptions, files and acceptance evidence.
