from io import StringIO
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]


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
