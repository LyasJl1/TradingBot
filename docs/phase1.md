# Phase 1 implementation notes

Scope: data, UTC candle construction, indicators and **raw breakout candidate
signals**. No positions, fills, PnL, stop management, risk, broker, persistence,
Telegram or live trading are implemented. `paper_mode: true` is the central
default; this package cannot submit orders in either mode. News/spread/session
entry gates, position constraints and execution checks must be applied by later
phases when turning a candidate into an intended order. These outputs must not
be treated as permission to trade or as a full-strategy backtest.

## Files

- `.gitignore`, `pyproject.toml`: local environment and Python packaging.
- `config.yaml`, `src/tradingbot/config.py`: central parameters and typed validation.
- `src/tradingbot/__init__.py`, `src/tradingbot/__main__.py`: package and local data-to-signals CLI.
- `src/tradingbot/data/__init__.py`, `src/tradingbot/data/candles.py`: CSV/Parquet validation and shared UTC resampling.
- `src/tradingbot/indicators/__init__.py`, `src/tradingbot/indicators/core.py`: pure indicator functions.
- `src/tradingbot/signals/__init__.py`, `src/tradingbot/signals/core.py`: pure candidate decisions, reasons and indicator provenance.
- `tests/test_config.py`, `tests/test_data.py`, `tests/test_indicators.py`, `tests/test_signals.py`, `tests/test_no_lookahead.py`, `tests/test_cli.py`.
- `README.md`, `docs/phase1.md`: usage, assumptions and phase boundaries.

`SPEC.md` is unchanged.

## Timestamp and data contract

- Every timestamp labels the candle **open**, not its close. CSV uses a
  `timestamp` column with an explicit `Z` or zero UTC offset. Parquet accepts that
  column or a UTC DatetimeIndex. Naive/local timestamps are rejected, not guessed.
- Required lowercase columns are `open,high,low,close,volume`; optional `spread`
  must be finite and nonnegative when present. Absence stays absent. This phase
  never assumes zero costs or applies a spread fallback.
- Base candles must be on a complete M15 or H1 UTC grid. The default is H1;
  both feed exactly the same H4/D1 resampler. Broker H4/D1 data cannot be supplied
  to the public base-data signal pipeline.
- The grid check is deliberately strict: **weekends and holidays are gaps too**.
  There is no verified broker market calendar in the repository. Ordinary FX
  history with weekend gaps therefore fails validation. Do not fill those gaps
  with invented prices. Explicit market-closure/calendar support needs a user
  decision before research on such datasets; no calendar is inferred here.
- H4 starts at 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC; D1 starts at
  00:00 UTC. Aggregation is first open, max high, min low, last close and summed
  volume. Optional spread is the **last base observation**, not a future
  execution spread. Downstream cost code must define its units/source explicitly.
- `as_of` is required and explicitly UTC; there is no hidden wall clock. Only
  closed base and complete closed target candles are used. Partial leading and
  trailing groups are omitted with a reason through the standard logger. Interior
  gaps, invalid OHLC, unordered/duplicate timestamps and nonfinite data fail.
- Input is one symbol at a time; callers can iterate the configured symbol list.
  No random operations occur in Phase 1. Tests use the central seed.

## Indicator initialization and timing

- EMA is seeded with an SMA of the first N closes at index N-1 and recurs with
  alpha `2/(N+1)`. This is documented because libraries differ in EMA seeding.
- True range starts with `high-low`; subsequent rows include the previous close.
  Wilder ATR seeds with the mean of the first N ranges at index N-1, then uses
  `(previous*(N-1)+TR)/N`. The indicator itself is unshifted; the signal consumes
  **ATR shifted once**, so the evaluated H4 candle never affects its ATR.
- ADX uses directional differences from bars 1..N for its initial Wilder
  TR/+DM/-DM averages. The first DX is at index N, and the first ADX is the
  average of N DX values at index `2*N-1`. Tied directional movements count as
  neither direction; flat markets yield zero DX/ADX after warmup. Missing warmup
  values remain NaN and block a candidate with a reason.
- Donchian entry/exit channels use exactly the previous N bars, shifted by one.
  Exit channels are exposed for later phases; no trailing stop is managed here.
- D1 indicators include their own fully closed daily candle. The as-of join
  requires `D1 close time < H4 open time`, **strictly**. At a 00:00 H4 open, the
  daily candle closing at that same instant is unavailable under this conservative
  interpretation of “BEFORE”; it becomes usable at 04:00. Thus all H4 decisions
  can lag the newest D1 close, intentionally. There is no backward-fill.
- A decision is timestamped at H4 close. `entry_time` is the intended next H4
  open (the same UTC instant), **not an executed fill or a known entry price**.
- Price/EMA and breakout comparisons are strict; `ADX >= threshold` is inclusive.
  Every H4 decision, including no-signal/warmup outcomes, returns a reason. Pure
  functions return decision records; the CLI serializes them as the Phase 1
  decision log. Stateful logging/alerts remain Phase 5.

## Acceptance evidence

Run from the repository root after editable installation:

```sh
.venv/bin/python -m pytest
```

Executed on Python 3.12.14 with pytest 8.4.2: **120 passed in 4.68s**,
zero failures and zero skips. The CSV and Parquet CLI smoke tests both execute
the installed package in a subprocess from outside the checkout. `git diff
--check` also passed. This records validation in the current environment only.

The suite covers:

| Phase 1 criterion | Status | Evidence |
| --- | --- | --- |
| CSV/Parquet validation and UTC M15/H1 resampling | PASSED | `test_data.py`, `test_cli.py` |
| Indicators versus fixed reference values | PASSED | `test_indicators.py`: EMA and hand-calculated rational fixtures |
| Wilder ATR/ADX, including mixed direction and flat prices | PASSED | `test_indicators.py`: periods 3 and 14, seed and recurrence checks |
| Donchian shift by one | PASSED | Indicator mutation test and signal-candle independence checks |
| Conservative D1/H4 alignment | PASSED | `test_signals.py`: midnight/exact boundary, warmup, forming candles |
| Long/short trend conditions and breakout comparisons | PASSED | `test_signals.py`: strict comparisons, threshold equality, both directions |
| Critical no-lookahead guarantee | PASSED | `test_no_lookahead.py`: full vs truncated history at six cutoffs, future mutation, H1/M15 parity |

The no-lookahead fixture uses 720 days of deterministic **synthetic continuous**
base data and the actual default periods. It requires mature D1/ATR values and
both long and short candidates, preventing an all-warmup/all-none false pass.
It compares every earlier output column, including channels, ATR, daily
indicators and their timestamps. Changes to the signal candle affect neither its
ATR nor its channels, but do affect subsequent values. These are correctness
tests, not historical performance results. No real broker data has been validated.

Later-phase requirements are NOT TESTED and intentionally unimplemented. No
environment configuration was published or changed as part of implementation.
