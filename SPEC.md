# SPEC.md: Donchian Breakout Trading Bot

## 0. MISSION AND RULES

You are a senior quant developer. Build a modular, testable, reproducible
Python trading bot implementing EXACTLY the strategy below.

Priority #1 is specification fidelity, not performance. Do not add
indicators, filters, rules or parameter changes. Do not optimize or tune
anything unless a section below explicitly says so. A technical safeguard
may block an unsafe order but must never alter the strategy's numbers.

WORKFLOW (mandatory): implement ONE phase, run its tests, report, and STOP.
Do not start the next phase until I approve.

Engineering rules:
- Python, type hints, pytest, one central config.yaml, no hardcoded
  strategy/risk numbers.
- Indicators and signals are pure functions with no broker or account access.
- Backtest and live/paper modes share the same signal and risk code.
- Deterministic, reproducible (fixed seed from config).
- Log every decision with its reason.
- Fail explicitly rather than silently assume. Never treat missing costs
  as zero.
- Never place an order without an attached stop loss.
- PAPER_MODE defaults to TRUE. Never enable live trading automatically.
- Never claim a test passed unless you actually ran it.

## 1. PROJECT DECISIONS (already made, do not re-ask)

D1. Candles: the bot builds ALL H4 and D1 candles itself from M15 or H1
    base data, using UTC boundaries, with the SAME resampling code in
    backtest and live. Never use broker-provided H4/D1 candles for signals.
D2. Research integrity: parameter sweeps and any optimization run ONLY on
    the in-sample segment (first 70%). The out-of-sample segment (last 30%)
    is evaluated once with fixed parameters. Never sweep on full data.
D3. Costs: spread, commission, slippage AND swap (long/short, per lot per
    night, including triple-swap day) are all configurable cost inputs.
D4. Correlated buckets: config option max_bucket_risk_pct = 1.0 (total
    open risk of instruments in one bucket). This is a user decision.
D5. Placeholder values (EURUSD), to be replaced by verified broker data:
    contract size 100000, lot step 0.01, min lot 0.01, spread from data
    (fallback 1.2 pips), commission 7 USD per round-turn lot, slippage
    0.3 pips per side, swap must be configured explicitly (no default 0).
D6. Sweep steps: Donchian 1, ATR multiplier 0.25, ADX threshold 1.
D7. Walk-forward: 5-year train, 1-year test, rolling by 1 year.
D8. Sharpe: daily equity returns, annualized with sqrt(252).
D9. Random seed: 42.
D10. Broker: MetaTrader 5 (Windows). Default symbol EURUSD; symbol list
     configurable. Design for testing 3-5 low-correlated liquid
     instruments with identical rules (sample size matters).
D11. Timestamps: all UTC. Day boundaries for risk limits are UTC.
D12. Account currency: configurable. Convert risk into account currency
     where needed.

## 2. STRATEGY (exact)

Timeframes: D1 = trend filter. H4 = entries, exits, ATR, Donchian.

Indicators:
- D1: EMA(50), EMA(200) on close; ADX(14) (Wilder).
- H4: ATR(14) (Wilder); Donchian entry channel 20; Donchian exit channel 10.
- Donchian highest high / lowest low of the PREVIOUS N candles, excluding
  the current candle (shift by 1).

No-lookahead rules (critical):
- Signals are evaluated only on a CLOSED H4 candle; order executes at the
  OPEN of the next H4 candle.
- D1 values used for an H4 decision must come from a D1 candle fully
  closed BEFORE the evaluated H4 candle begins (conservative; document
  the effect).
- ATR and channels must not include the signal candle's data.

Trend filter (D1):
- Long allowed: close > EMA200 AND EMA50 > EMA200 AND ADX >= 18.
- Short allowed: close < EMA200 AND EMA50 < EMA200 AND ADX >= 18.
- Otherwise no new trades.

Entry (H4):
- Long: H4 close > Donchian-20 upper AND long allowed.
- Short: H4 close < Donchian-20 lower AND short allowed.
- One position per symbol, no pyramiding, no same-bar reversal. Process
  exits first.

Initial stop: stop distance = 2.0 x ATR(14) at signal time, applied
relative to the actual next-open entry price (long: entry - distance,
short: entry + distance). Attached to the order. If the broker cannot
accept it, reject the entry and log why. Never widen the stop.

Trailing stop: Donchian-10 opposite channel (long: lowest low of previous
10 candles; short: highest high). Long stop only rises, short stop only
falls. Updates computed after a candle closes apply only to later price
action. No fixed take-profit.

Optional partial TP (config flag, default OFF): close 50% at +2R (R = initial
risk), trail the rest with the same rules.

Filters (each independently toggleable):
- News: no new entries within +/-30 min of events in a local CSV
  (datetime_utc, currency, impact). All events count (no impact threshold).
  Match event currency to the instrument's currencies. Evaluate against the
  intended entry execution time. Existing positions are not closed.
- Spread: skip entry if spread > 20% of ATR(14) (compatible units).
- Session: no new entries 21:00-23:00 UTC, or after Friday 20:00 UTC until
  the week resumes. Applies to entry execution time.

## 3. RISK (all mandatory)

- Risk per trade: 0.75% of current equity (realized + unrealized).
- Size = (equity x risk_pct) / (stop_distance x value_per_unit), rounded
  DOWN to lot step. Skip if below min lot. Do not change the formula.
- Max total open risk: 2.5% of equity, including the candidate trade.
  Position open risk = max(0, loss from entry to current stop) x remaining
  size; a stop beyond breakeven never offsets other positions' risk.
- Correlated groups (configurable membership) count as one bucket, capped
  by max_bucket_risk_pct (D4) in addition to the total cap.
- Daily loss limit: 2% of start-of-day equity (fixed reference, UTC day).
  When hit: block new entries until next UTC day; keep existing trades and
  stops managed.
- Max drawdown kill switch: 10% from peak equity. When hit: block all new
  entries, close nothing, keep stops, send alert, require MANUAL reset
  (flag file or config change). Persists across restarts. Never auto-resets
  (not on recovery, new day or reconnect). Daily reset never bypasses it.
- Max 3 new trades per UTC day (count opened trades, not signals).
- Risk module returns explicit approval/rejection reasons.

## 4. ARCHITECTURE

data/ (loaders CSV/Parquet, UTC validation, gap checks, resampling),
indicators/ (EMA, ATR, ADX, Donchian), signals/ (pure: data -> long/short/none
+ reason + timestamps), risk/, execution/ (broker interface + MT5 adapter),
state/ (SQLite), logging/ (structured logs + Telegram), backtest/, tests/.
Credentials come from environment variables referenced by config, never
committed or logged.

## 5. BACKTEST REQUIREMENTS

- Input: CSV/Parquet OHLCV (+ optional spread), UTC. Validate columns,
  ordering, duplicates, missing candles, OHLC consistency. Fail clearly if
  the base timeframe can't support H4/D1 resampling.
- Event order per H4 candle: process existing positions/stops -> apply
  gap and intrabar rules -> update fills, costs, equity -> evaluate the
  closed candle -> apply filters and risk -> schedule entry for next open.
- Pessimistic rules: if stop and target are both reachable in one candle,
  stop first. If price gaps through a stop, fill at the open (plus
  slippage). Costs apply to entries, exits and partial closes.
- Outputs: trade log CSV (symbol, direction, signal time, entry time/price,
  initial stop, ATR, initial risk, size, exit time/price/reason, PnL,
  costs, R multiple, partial exits), equity curve, and EXACTLY these
  metrics: net profit, profit factor, win rate, avg win/avg loss,
  expectancy (R), max drawdown, Sharpe, number of trades, longest losing
  streak. Handle empty/undefined metrics explicitly.
- 70/30 chronological split, reported separately. Never shuffle candles.
- Walk-forward per D7; results per window and combined; no train/test leakage.
- Parameter sweep (in-sample only, per D2): Donchian entry 15-25, ATR
  multiplier 1.5-3.0, ADX threshold 15-25, steps per D6. Output heatmaps
  to inspect plateau vs isolated peak. Never auto-select or deploy
  "best" parameters.
- Monte Carlo: reshuffle TRADE order 1000 times, fixed seed, report 95th
  percentile max drawdown; document how each equity path is built.

## 6. LIVE / PAPER REQUIREMENTS

- Loop triggers only when a new H4 candle is fully closed (built per D1).
  Persist the last processed candle to prevent duplicate signals/orders.
- PAPER_MODE=TRUE: connect to demo for data and metadata, simulate orders,
  submit nothing, same signal/risk code as backtest.
- Validate broker min stop distance before ordering; reject rather than
  alter the stop.
- Reconnect with retry/backoff; handle rejections and partial fills;
  distinguish UNKNOWN order status from confirmed rejection; retries must
  never create duplicate positions.
- Startup reconciliation of SQLite vs broker positions. Never auto-close
  positions on mismatch. If safe state can't be established, block new
  entries and alert.
- Heartbeat log hourly; alert if no data for > 1 candle period. Missing
  data never permits trading.
- Telegram alerts: trades, errors, kill switch, lost connection.

## 7. PHASES (stop after each for my approval)

PHASE 1: data + resampling + indicators + signals + tests.
  Tests: indicators vs known reference values; Wilder ATR/ADX; Donchian
  shift-by-1; D1/H4 alignment; long/short filter conditions; breakout
  comparisons. CRITICAL no-lookahead test: compute signals on full data,
  truncate future data after a chosen point, recompute; all earlier
  signals must be identical (cover channels, ATR, D1 filter, alignment).
  No backtest, risk or execution code yet.

PHASE 2: backtest engine, costs (incl. swap), metrics, reports, 70/30
  split, walk-forward, sweeps (in-sample only), heatmaps, Monte Carlo.
  Tests: next-open fills, stop-first, gap-through-stop, trailing
  monotonicity, partial TP, costs, no same-bar reversal, equity
  accounting, reproducibility, train/test chronology. Design around a risk
  interface; label results as INCOMPLETE until Phase 3. Do not present
  them as full-strategy results.

PHASE 3: full risk module integrated into the backtester (sizing, rounding
  down, min lot, open-risk cap, bucket cap, daily loss, kill switch with
  manual reset interface, 3 trades/day). Tests for each rule, including
  that existing positions stay open and stops stay managed. Re-run all
  prior tests, then produce full-strategy reports.

PHASE 4: broker interface, MT5 adapter, paper executor, closed-H4 loop,
  reliability controls. Mocked-broker tests: rejections, partial fills,
  min-stop rejection, disconnect/reconnect, timeout/unknown state,
  duplicate prevention, missing-stop prevention, paper mode submits
  nothing.

PHASE 5: SQLite persistence, startup reconciliation, structured logging,
  Telegram, monitoring, kill-switch persistence and manual reset. Restart
  tests: open position, trailing stop, partial close, daily-loss block,
  active kill switch, connection failure, uncertain order, broker/SQLite
  mismatch. End-to-end validation. Real-money trading is never enabled.

## 8. BEHAVIOR AT EVERY PHASE

Start: inspect the repo, list files you will create/modify, note
dependencies and unresolved assumptions. Do only the approved phase.
End: run pytest, report exact commands and results, list files changed,
mark each acceptance criterion PASSED / FAILED / NOT TESTED, list
limitations, then end with:
"Phase N is complete. Please review. I will not proceed to Phase N+1 until
you explicitly approve."
Record assumptions in docs; if a new material ambiguity appears, ask at the
phase boundary instead of guessing.

## START

Read this file. Inspect the repository. Implement PHASE 1 ONLY. Stop and
wait for approval.
