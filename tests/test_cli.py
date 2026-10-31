from io import StringIO
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]


def paired_files(directory: Path, index: pd.DatetimeIndex, symbol: str = "EURUSD", flat: int | None = None) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    price = 1.1 + np.arange(len(index)) * .00001
    frame = pd.DataFrame({"timestamp": index.as_unit("ms").asi8,
                          "open": price, "high": price + .000002,
                          "low": price - .000002, "close": price})
    if flat is not None:
        frame.loc[flat, ["open", "high", "low", "close"]] = price[flat]
    bid = directory / f"{symbol}_H1_bid.csv"
    ask = directory / f"{symbol}_H1_ask.csv"
    frame.to_csv(bid, index=False)
    frame.loc[:, ["open", "high", "low", "close"]] += .0002
    frame.to_csv(ask, index=False)
    return bid, ask


def quality_command(directory: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "tradingbot", "data-quality",
                           "--config", str(ROOT / "config.yaml"), "--raw-dir", str(directory),
                           *extra], cwd=directory.parent, capture_output=True, text=True, check=False)


def report(directory: Path, *extra: str) -> dict:
    result = quality_command(directory, *extra)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)[0]


@pytest.mark.parametrize("extension", ["csv", "parquet"])
def test_cli_emits_mature_decisions_from_base_data(tmp_path, extension) -> None:
    index = pd.date_range("2024-01-01", periods=230 * 24, freq="h", tz="UTC", name="timestamp")
    price = 1.1 + np.arange(len(index)) * 0.00001
    base = pd.DataFrame({"open": price, "high": price + .000002,
                         "low": price - .000002, "close": price, "volume": 1}, index=index)
    source = tmp_path / f"base.{extension}"
    if extension == "csv":
        base.to_csv(source)
    else:
        base.to_parquet(source)
    cutoff = index[-1] + pd.Timedelta(hours=1)
    command = [sys.executable, "-m", "tradingbot", "--config", str(ROOT / "config.yaml"),
               "--data", str(source), "--as-of", cutoff.isoformat()]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    decisions = pd.read_csv(StringIO(result.stdout))
    assert len(decisions) == 230 * 6
    mature = decisions.loc[decisions.direction == "long"]
    assert not mature.empty
    assert (mature.reason == "long_breakout").all()
    assert (mature.atr > 0).all()
    assert (pd.to_datetime(mature.d1_close_time) < pd.to_datetime(mature.candle_open)).all()
    assert (pd.to_datetime(decisions.signal_time) <= cutoff).all()


def test_cli_rejects_naive_time_instead_of_assuming_utc(tmp_path) -> None:
    source = tmp_path / "invalid.csv"
    source.write_text("timestamp,open,high,low,close,volume\n2024-01-01,1,2,0.5,1.5,1\n")
    result = subprocess.run(
        [sys.executable, "-m", "tradingbot", "--config", str(ROOT / "config.yaml"),
         "--data", str(source), "--as-of", "2024-01-02T00:00Z"],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "explicitly UTC" in result.stderr
    assert not result.stdout


def test_quality_cli_parses_epoch_no_volume_and_reports_counts(tmp_path) -> None:
    directory = tmp_path / "raw"
    index = pd.date_range("2014-01-01", periods=48, freq="h", tz="UTC")
    paired_files(directory, index)
    result = report(directory)
    assert result["bid_rows"] == result["ask_rows"] == result["input_rows"] == result["rows_retained"] == 48
    assert pd.Timestamp(result["first_timestamp"]) == index[0]
    assert pd.Timestamp(result["last_timestamp"]) == index[-1]
    assert result["first_retained_timestamp"] == result["first_timestamp"]
    assert result["flat_candles_dropped"] == result["gaps_found"] == 0
    assert result["warnings"] == []
    assert result["H4"] == {"built": 12, "dropped": 0, "dropped_below_minimum": 0, "dropped_not_closed": 0}
    assert result["D1"] == {"built": 2, "dropped": 0, "dropped_below_minimum": 0, "dropped_not_closed": 0}


def test_quality_cli_flat_candles_count_once_and_are_not_filled(tmp_path) -> None:
    directory = tmp_path / "raw"
    paired_files(directory, pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC"), flat=3)
    result = report(directory)
    assert result["input_rows"] == 48
    assert result["rows_retained"] == 47
    assert result["flat_candles_dropped"] == 1
    assert result["gaps_found"] == 1
    assert result["gaps"][0]["missing_base_candles"] == 1
    assert result["warnings"] == []
    assert result["H4"]["built"] == 12


def test_quality_cli_reports_weekday_long_gap_and_sparse_bucket_drop(tmp_path) -> None:
    directory = tmp_path / "raw"
    index = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC").delete(range(3, 10))
    paired_files(directory, index)
    result = report(directory)
    assert result["rows_retained"] == 17
    assert result["gaps_found"] == len(result["warnings"]) == 1
    assert "7 missing hours" in result["warnings"][0]
    assert result["gaps"][0]["warning"] is True
    assert result["H4"] == {"built": 5, "dropped": 1, "dropped_below_minimum": 1, "dropped_not_closed": 0}
    assert result["D1"]["built"] == 1


def test_quality_cli_reports_weekend_gap_without_warning(tmp_path) -> None:
    directory = tmp_path / "raw"
    index = pd.date_range("2024-01-05", periods=24, freq="h", tz="UTC").append(
        pd.date_range("2024-01-08", periods=24, freq="h", tz="UTC"))
    paired_files(directory, index)
    result = report(directory)
    assert result["gaps_found"] == 1
    assert result["gaps"][0]["duration_hours"] == 48
    assert result["warnings"] == []
    assert result["H4"]["built"] == result["H4"]["dropped_below_minimum"] == 12
    assert result["D1"]["built"] == result["D1"]["dropped_below_minimum"] == 2


def test_quality_cli_minimum_does_not_close_forming_bucket(tmp_path) -> None:
    directory = tmp_path / "raw"
    paired_files(directory, pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC"))
    result = report(directory, "--as-of", "2024-01-01T02:00:00Z")
    assert result["H4"] == {"built": 0, "dropped": 1, "dropped_below_minimum": 0, "dropped_not_closed": 1}
    assert result["D1"]["built"] == 0


def test_quality_cli_all_flat_preserves_original_bounds_and_reports_empty_builds(tmp_path) -> None:
    directory = tmp_path / "raw"
    bid, ask = paired_files(directory, pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC"))
    for path in (bid, ask):
        frame = pd.read_csv(path)
        for column in ("open", "high", "low"):
            frame[column] = frame["close"]
        frame.to_csv(path, index=False)
    result = report(directory)
    assert result["input_rows"] == result["flat_candles_dropped"] == 24
    assert result["rows_retained"] == 0
    assert result["first_retained_timestamp"] is None
    assert result["last_retained_timestamp"] is None
    assert pd.Timestamp(result["last_timestamp"]) == pd.Timestamp("2024-01-01T23:00Z")
    assert result["H4"]["built"] == result["D1"]["built"] == 0
    assert result["H4"]["dropped_below_minimum"] == 6
    assert result["D1"]["dropped_below_minimum"] == 1


@pytest.mark.parametrize("mode", ["directory_missing", "directory_empty", "ask_missing", "row_mismatch", "timestamp_mismatch"])
def test_quality_cli_missing_or_unaligned_input_fails_clearly(tmp_path, mode) -> None:
    directory = tmp_path / "raw"
    if mode != "directory_missing":
        directory.mkdir()
    if mode not in {"directory_missing", "directory_empty"}:
        _, ask = paired_files(directory, pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC"))
        if mode == "ask_missing":
            ask.unlink()
        elif mode == "row_mismatch":
            pd.read_csv(ask).iloc[:-1].to_csv(ask, index=False)
        else:
            frame = pd.read_csv(ask)
            frame.loc[len(frame) - 1, "timestamp"] += 3600000
            frame.to_csv(ask, index=False)
    result = quality_command(directory)
    assert result.returncode == 2
    assert result.stderr.startswith("error:")
    assert not result.stdout


def test_quality_cli_discovers_pairs_and_selects_symbols(tmp_path) -> None:
    directory = tmp_path / "raw"
    index = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
    paired_files(directory, index, "EURUSD")
    paired_files(directory, index, "GBPUSD")
    all_result = quality_command(directory)
    assert all_result.returncode == 0, all_result.stderr
    assert [item["symbol"] for item in json.loads(all_result.stdout)] == ["EURUSD", "GBPUSD"]
    assert report(directory, "--symbol", "GBPUSD")["symbol"] == "GBPUSD"


def test_paired_signal_cli_uses_epoch_input_without_volume(tmp_path) -> None:
    bid, ask = paired_files(tmp_path / "raw", pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC"))
    result = subprocess.run([sys.executable, "-m", "tradingbot", "--config", str(ROOT / "config.yaml"),
                             "--bid", str(bid), "--ask", str(ask), "--as-of", "2024-01-02T00:00Z"],
                            cwd=tmp_path, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert len(pd.read_csv(StringIO(result.stdout))) == 6


@pytest.mark.parametrize("extra", [["--bid", "bid.csv"], ["--ask", "ask.csv"],
                                    ["--data", "base.csv", "--bid", "bid.csv", "--ask", "ask.csv"]])
def test_signal_cli_rejects_incomplete_or_conflicting_input_options(tmp_path, extra) -> None:
    result = subprocess.run([sys.executable, "-m", "tradingbot", "--config", str(ROOT / "config.yaml"),
                             "--as-of", "2024-01-02T00:00Z", *extra],
                            cwd=tmp_path, capture_output=True, text=True, check=False)
    assert result.returncode == 2
    assert "error:" in result.stderr
