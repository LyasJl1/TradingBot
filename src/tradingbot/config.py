"""Typed, explicit configuration; strategy defaults live only in config.yaml."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    base_timeframe: str
    entry_timeframe: str
    trend_timeframe: str

    def __post_init__(self) -> None:
        if self.base_timeframe not in {"M15", "H1"}:
            raise ValueError("base_timeframe must be M15 or H1")
        if self.entry_timeframe != "H4" or self.trend_timeframe != "D1":
            raise ValueError("The specified strategy requires H4 entries and D1 trend candles")


@dataclass(frozen=True)
class StrategyConfig:
    ema_fast_period: int
    ema_slow_period: int
    adx_period: int
    adx_threshold: float
    atr_period: int
    donchian_entry_period: int
    donchian_exit_period: int

    def __post_init__(self) -> None:
        for name in (
            "ema_fast_period", "ema_slow_period", "adx_period", "atr_period",
            "donchian_entry_period", "donchian_exit_period",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.ema_fast_period >= self.ema_slow_period:
            raise ValueError("ema_fast_period must be less than ema_slow_period")
        if (isinstance(self.adx_threshold, bool)
                or not isinstance(self.adx_threshold, (int, float))
                or not isfinite(self.adx_threshold)
                or not 0 <= self.adx_threshold <= 100):
            raise ValueError("adx_threshold must be finite and in [0, 100]")


@dataclass(frozen=True)
class BotConfig:
    paper_mode: bool
    random_seed: int
    symbols: tuple[str, ...]
    data: DataConfig
    strategy: StrategyConfig

    def __post_init__(self) -> None:
        if type(self.paper_mode) is not bool:
            raise ValueError("paper_mode must be a boolean")
        if type(self.random_seed) is not int or self.random_seed < 0:
            raise ValueError("random_seed must be a nonnegative integer")
        if not self.symbols or any(not isinstance(s, str) or not s.strip() for s in self.symbols):
            raise ValueError("symbols must be a nonempty sequence of symbol names")
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("symbols must not contain duplicates")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise ValueError(f"{name} must be a mapping with string keys")
    return value.copy()


def load_config(path: str | Path) -> BotConfig:
    """Load one config; missing/unknown parameters fail rather than defaulting."""
    with Path(path).open(encoding="utf-8") as handle:
        raw = _mapping(yaml.safe_load(handle), "config")
    try:
        data = DataConfig(**_mapping(raw.pop("data"), "data"))
        strategy = StrategyConfig(**_mapping(raw.pop("strategy"), "strategy"))
        symbols = raw.pop("symbols")
        if not isinstance(symbols, list):
            raise ValueError("symbols must be a list")
        return BotConfig(data=data, strategy=strategy, symbols=tuple(symbols), **raw)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid or missing configuration field: {exc}") from exc
