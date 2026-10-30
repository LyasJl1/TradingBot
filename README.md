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

CSV requires `timestamp,open,high,low,close,volume`, optionally `spread`.
Timestamps are explicit UTC candle **opens**. Parquet is also supported.
Both timeframes are built internally; partial/current candles are excluded.

**Current data limitation:** input must be continuous. Weekend/holiday gaps fail
explicitly because no verified market calendar is assumed. Do not fill gaps with
invented candles. Signal output contains raw candidates and reasons; later phases
must apply entry filters, position constraints, risk and execution checks.

See [Phase 1 notes](docs/phase1.md) for seeding, conservative D1 alignment,
assumptions, files and acceptance evidence.
