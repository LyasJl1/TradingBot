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
- `src/tradingbot/data/__init__.py`, `src/tradingbot/data/candles.py`, `src/tradingbot/data/quality.py`: CSV/Parquet and paired BID/ASK validation, shared UTC resampling and data-quality reports.
- `src/tradingbot/indicators/__init__.py`, `src/tradingbot/indicators/core.py`: pure indicator functions.
- `src/tradingbot/signals/__init__.py`, `src/tradingbot/signals/core.py`: pure candidate decisions, reasons and indicator provenance.
- `tests/test_config.py`, `tests/test_data.py`, `tests/test_indicators.py`, `tests/test_signals.py`, `tests/test_no_lookahead.py`, `tests/test_cli.py`.
- `README.md`, `docs/phase1.md`, `SPEC.md`: usage, data decisions and phase boundaries.

The Phase 1 real-data patch updates `SPEC.md` decisions D13-D18. Strategy
indicators, thresholds, alignment and all later-phase requirements are unchanged.

## Timestamp and data contract

- Every timestamp labels the candle **open**, not its close. Numeric timestamps
  (including numeric text) are Unix epoch **milliseconds**, parsed with explicit
  UTC units; `1388534400000` is `2014-01-01T00:00:00+00:00`. Explicit `Z` or
  zero-offset ISO timestamps remain supported. Parquet also accepts a UTC
  DatetimeIndex. Naive/local ISO times, fractional/invalid epoch values,
  unordered/duplicate timestamps and off-grid opens fail validation.
- Required lowercase columns are `open,high,low,close`. **Volume is optional
  and unused**: ignored on ingestion, never required, synthesized, aggregated
  or read by indicators/signals. OHLC must be finite and consistent. Optional
  `spread` must be finite/nonnegative if present; absence remains absent.
- Dukascopy H1 files are paired as `data/raw/{SYMBOL}_H1_bid.csv` and
  `data/raw/{SYMBOL}_H1_ask.csv`. Raw row counts and parsed timestamp sequences
  must match **before filtering**, or loading fails. OHLC uses **BID** prices;
  spread is `ask close - bid close` in price units, a **close-based estimate**.
  This is not the spread at a later execution time. No costs are assumed zero.
- Flat base BID candles (`open == high == low == close`) are dropped and counted.
  The same BID mask retains the corresponding ASK quotes; independent ASK flat
  candles do not remove a nonflat BID candle. Pair alignment is never repaired
  by silently intersecting dates or dropping unmatched rows.
- Base timestamps must lie on an M15 or H1 UTC grid, but **weekend/holiday and
  other gaps are allowed**. Nothing is forward-filled/interpolated. Both
  timeframes use the same resampler; indicators operate on retained observed
  candles. Broker H4/D1 candles cannot enter the public base-data pipeline.
- All gaps are reported. The gap starts at the **first missing expected base
  open** (`previous observed open + base period`) and ends at the next observed
  open. Its duration is this missing interval, not the distance between the two
  observed opens. Gaps longer than 4 hours with a Monday-Friday UTC start are
  listed as warnings, without failing. A gap of exactly 4 hours is not warned;
  weekend starts are reported but not warned. Flat removal can create a gap.
- H4 starts at 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC; D1 starts at
  00:00 UTC. Aggregation is first retained open, max high, min low and last
  retained close. Optional spread is the **last base observation**, not a future
  execution spread. Downstream cost code must define its units/source explicitly.
- `as_of` is required for signals and explicitly UTC; there is no hidden wall
  clock. Only closed base candles contribute, and a target bucket's nominal UTC
  close must be <= `as_of`, **even if its minimum count is already met**.
- `data.min_base_candles_per_h4: 2` and `data.min_base_candles_per_d1: 12` count
  retained base candles. Fully closed buckets meeting these minima are built
  from available rows; buckets below the minimum are dropped and counted. Empty
  buckets inside the observed span count as drops too. Nothing is manufactured
  outside the span of closed observations. An otherwise adequate partial
  leading/trailing bucket is allowed once nominally closed. Changing the source
  from H1 to M15 retains the configured absolute base-candle minimum counts.
- Reports distinguish built, dropped-below-minimum and dropped-not-closed
  buckets. Drop reasons are disjoint: a forming bucket is classified not-closed
  before checking its count. Every excluded group has an explicit reason.
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
- A decision is timestamped at H4 close. `entry_time` is the nominal next H4
  open (the same UTC instant), **not an executed fill, a known entry price or a
  guarantee that a market observation exists then**. No gap is filled to create
  that observation. Later execution work must handle market availability.
- Price/EMA and breakout comparisons are strict; `ADX >= threshold` is inclusive.
  Every H4 decision, including no-signal/warmup outcomes, returns a reason. Pure
  functions return decision records; the CLI serializes them as the Phase 1
  decision log. Stateful logging/alerts remain Phase 5.

## Acceptance evidence

Run from the repository root after editable installation:

```sh
.venv/bin/python -m pytest
```

The original Phase 1 run on Python 3.12.14 with pytest 8.4.2 passed 120 tests.
The real-data patch full suite (`.venv/bin/python -m pytest`) executed on the same
Python/pytest versions: **171 passed in 11.03s**, zero failures and zero skips.
This includes all **18 no-lookahead tests**, on continuous and gappy histories.
CSV and Parquet CLI smoke tests execute
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

The no-lookahead fixtures use 720 days of deterministic **synthetic continuous
and gappy** base data and the actual default periods. They require mature D1/ATR values and
both long and short candidates, preventing an all-warmup/all-none false pass.
It compares every earlier output column, including channels, ATR, daily
indicators and their timestamps. Changes to the signal candle affect neither its
ATR nor its channels, but do affect subsequent values. These are correctness
tests, not historical performance results. They also cover OHLC-only sources,
weekends, holidays/missing weekdays, sparse buckets and dropped flat candles.

## Real-data patch acceptance

Files changed by this patch: `.gitignore`, `README.md`, `SPEC.md`, `config.yaml`,
`docs/phase1.md`, `src/tradingbot/__main__.py`, `src/tradingbot/config.py`,
`src/tradingbot/data/__init__.py`, `src/tradingbot/data/candles.py`,
`src/tradingbot/signals/core.py`, and `tests/test_cli.py`, `tests/test_config.py`,
`tests/test_data.py`, `tests/test_no_lookahead.py`, `tests/test_signals.py`.
New file: `src/tradingbot/data/quality.py`. No dependency or indicator changes.

| Requested change | Status | Evidence |
| --- | --- | --- |
| Epoch-ms UTC parsing, with ISO support retained | PASSED | Synthetic parser tests and real Dukascopy file load |
| Volume optional and unused | PASSED | OHLC-only loaders, CLI and signal tests; irrelevant volume values ignored |
| Paired raw timestamp/count identity, BID prices, close-spread estimate | PASSED | Pair mismatch/flat-mask/spread tests and real files |
| Unfilled gaps, configurable 2/12 minima, dropped bucket counts and warnings | PASSED | Gap/boundary/minimum-count tests and real quality report |
| Flat BID candle removal and counts | PASSED | Isolated/all-flat/revalidation tests and real report |
| Quality report command executed on real input | PASSED | Command and actual output below; exit status 0 |
| Full suite, including no-lookahead | PASSED | 171 passed, including 18 no-lookahead cases |
| Raw-data exclusion and updated data decisions | PASSED | `git check-ignore`, no CSVs added to patch; `SPEC.md` D13-D18 and this document |

## Data-quality command

From the repository root, discover and report every paired symbol in `data/raw/`:

```sh
.venv/bin/python -m tradingbot data-quality --config config.yaml --raw-dir data/raw
```

Use repeated `--symbol EURUSD --symbol GBPUSD` to select symbols explicitly.
`--as-of 2026-01-01T00:00:00Z` is optional for this report; without it, the cutoff
is the last original input open plus one H1 period. This is deterministic and
does not bypass target-bucket close times. The command prints JSON with per-symbol
raw BID/ASK and retained row counts, UTC timestamp bounds, flats dropped, gap
records/warnings and H4/D1 build/drop counts. Missing sides or absent real files
produce a clear nonzero error. No synthetic data is substituted for real files.

Signal CLI pairing is also supported:

```sh
.venv/bin/python -m tradingbot --config config.yaml \
  --bid data/raw/EURUSD_H1_bid.csv --ask data/raw/EURUSD_H1_ask.csv \
  --as-of 2026-01-01T00:00:00Z > decisions.csv
```

`data/raw/` and `*.csv` are ignored by Git. Small synthetic fixtures are generated
inside tests; no CSV data is committed. Indicator formulas and strategy numbers
are unchanged, and no Phase 2 code is added.

## Actual EURUSD quality report

The user-provided files were found at the root of `main`, commit `440b4b3`, and
copied unmodified to the requested ignored `data/raw/` paths. Existing upstream
uploads were not altered; this patch adds no CSV files to Git.

Executed successfully (exit status 0):

```sh
.venv/bin/python -m tradingbot data-quality --config config.yaml --raw-dir data/raw > data/raw/quality_report.json
```

The full JSON stdout is saved locally at `data/raw/quality_report.json`, including
all 681 gap records and warning messages. Its summary is:

```text
Symbol:                  EURUSD
Bid rows:                111682
Ask rows:                111682
First raw timestamp:     2014-01-01T00:00:00+00:00
Last raw timestamp:      2026-10-29T23:00:00+00:00
First retained:          2014-01-01T22:00:00+00:00
Last retained:           2026-10-29T23:00:00+00:00
Flat BID candles dropped: 31718
Rows retained:            79964
Gaps found:                 681
Gap warnings:               681
As of:                   2026-10-30T00:00:00+00:00
H4 built / dropped:       20224 / 7886
D1 built / dropped:        3328 / 1357
```

All H4/D1 drops were below-minimum buckets; not-closed drops were zero at this
cutoff. All gaps happened to qualify for warnings under the specified weekday
start rule. Friday-start weekend closures are allowed but still warned. First
warning (verbatim):

```text
gap 2014-01-03T22:00:00+00:00 to 2014-01-05T22:00:00+00:00: 48 missing hours starting Monday-Friday UTC
```

The complete real-data signal command also exited 0:

```sh
.venv/bin/python -m tradingbot --config config.yaml --bid data/raw/EURUSD_H1_bid.csv --ask data/raw/EURUSD_H1_ask.csv --as-of 2026-10-30T00:00:00Z > data/raw/EURUSD_H4_decisions.csv
```

It emitted 20,224 decision records, each with a reason. Every signal time was at
or before the cutoff, and every matched D1 close was strictly before its H4 open.
This is Phase 1 compatibility validation, not a backtest or performance report.

Later-phase requirements are NOT TESTED and intentionally unimplemented. No
environment configuration was published or changed as part of implementation.
