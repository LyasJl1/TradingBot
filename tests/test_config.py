from pathlib import Path

import pytest
import yaml

from tradingbot.config import load_config

CONFIG_PATH = Path(__file__).parents[1] / "config.yaml"


def test_spec_defaults_live_in_central_config() -> None:
    config = load_config(CONFIG_PATH)
    assert config.paper_mode is True
    assert config.random_seed == 42
    assert config.symbols == ("EURUSD",)
    assert config.data.base_timeframe == "H1"
    assert config.data.entry_timeframe == "H4"
    assert config.data.trend_timeframe == "D1"
    assert vars(config.strategy) == {
        "ema_fast_period": 50, "ema_slow_period": 200,
        "adx_period": 14, "adx_threshold": 18,
        "atr_period": 14, "donchian_entry_period": 20, "donchian_exit_period": 10,
    }


@pytest.mark.parametrize("section, field, value", [
    (None, "random_seed", True),
    (None, "random_seed", -1),
    (None, "paper_mode", "false"),
    (None, "symbols", "EURUSD"),
    (None, "symbols", []),
    (None, "symbols", ["EURUSD", "EURUSD"]),
    (None, "unknown_field", 1),
    ("data", "base_timeframe", "H4"),
    ("data", "entry_timeframe", "H1"),
    ("data", "trend_timeframe", "H4"),
    ("strategy", "atr_period", 0),
    ("strategy", "adx_period", True),
    ("strategy", "ema_fast_period", 200),
    ("strategy", "adx_threshold", float("nan")),
    ("strategy", "adx_threshold", True),
    ("strategy", "adx_threshold", -1),
])
def test_invalid_config_fails_explicitly(tmp_path, section, field, value) -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    (raw if section is None else raw[section])[field] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_config(path)


def test_missing_parameter_has_no_implicit_fallback(tmp_path) -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    del raw["strategy"]["adx_threshold"]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="missing configuration"):
        load_config(path)
